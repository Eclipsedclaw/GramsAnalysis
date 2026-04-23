"""
01_Bin2HDF_FPB_Synced_EvtBuilder.py
Event Builder and Binary to HDF5 Converter for Synced CAEN Data
Designed for File per Board format where each CAEN yields its own binary file

Modifications:
- Absolute CAEN Indexing (reads 2-byte headers to place waveforms at true hardware index).
- Path-sniffing metadata extraction (Run9 Bandaid support).
- Automated hierarchical output sorting: Output > RunConfig > HV_State > Acq_Tag
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import struct
import numpy as np
import h5py
import glob
import time
import argparse
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import detector_config as dc
import grams_tools as gt

progress_queue = None

def init_worker(q):
    global progress_queue
    progress_queue = q

def extract_metadata(full_path):
    """Sniffs the entire absolute path to guarantee we catch parent-folder metadata."""
    meta = {}
    meta['run_config'] = dc.detect_run_mode(full_path) 
    
    full_path_lower = full_path.lower()
    
    # Sniff HV state or Pedestal
    if 'pedestal' in full_path_lower: 
        meta['hv_tag'] = 'Pedestal'
    else:
        hv_match = re.search(r'(?i)HV(\d+)', full_path)
        meta['hv_tag'] = f"HV{hv_match.group(1)}" if hv_match else 'Data'
        
    # Sniff Acquisition number
    acq_match = re.search(r'(?i)acq(\d+)', full_path)
    meta['acq_tag'] = f"acq{acq_match.group(1)}" if acq_match else "acq_unknown"
    
    return meta

def scan_board_metadata(filepath):
    header_format = "<IQIQi"
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(struct.calcsize(header_format))
            _, _, n_samples, samp_period, n_channels = struct.unpack(header_format, header_bytes)
            
            caen_ids = []
            for _ in range(n_channels):
                caen_ids.append(struct.unpack("<H", f.read(2))[0])
                f.seek(n_samples * 4, 1) 
                
            max_depth = max(caen_ids) + 1 if caen_ids else 64
            return n_channels, samp_period, n_samples, max_depth, caen_ids
    except: return 0, 0, 0, 64, []

def get_global_channel(bid, local_ch):
    if bid == '72': return local_ch
    elif bid == '85': return local_ch + 32  
    elif bid == '75': return local_ch + 96  
    return local_ch

def map_board_files(file_list, bid, desc_str, target_ts=None, limit=0):
    header_format = "<IQIQi"
    header_size = struct.calcsize(header_format)
    event_map = [] 
    
    with tqdm(total=len(file_list), desc=desc_str, unit='file') as pbar:
        for filepath in file_list:
            try:
                with open(filepath, 'rb') as f:
                    while True:
                        offset = f.tell()
                        h_bytes = f.read(header_size)
                        if len(h_bytes) < header_size: break
                        ev_num, ts, n_samp, _, n_chan = struct.unpack(header_format, h_bytes)
                        
                        event_map.append({'ts': ts, 'file': filepath, 'offset': offset, 'n_chan': n_chan, 'n_samp': n_samp})
                        f.seek((n_chan * 2) + (n_chan * n_samp * 4), 1)
                        
                        if limit > 0 and bid == '72' and len(event_map) >= limit: break
                        if target_ts and ts > target_ts: break
            except Exception as e: 
                print(f"Error mapping {filepath}: {e}")
                
            pbar.update(1)
            
            if limit > 0 and bid == '72' and len(event_map) >= limit: break
            if target_ts and len(event_map) > 0 and event_map[-1]['ts'] > target_ts: break
                
    event_map.sort(key=lambda x: x['ts'])
    return event_map

def fetch_waveform(event_info, max_depth):
    n_chan = event_info['n_chan']
    n_samp = event_info['n_samp']
    waveforms = np.full((max_depth, n_samp), np.nan, dtype=np.float32)
    with open(event_info['file'], 'rb') as f:
        f.seek(event_info['offset'] + struct.calcsize("<IQIQi")) 
        for i in range(n_chan):
            ch_id = struct.unpack("<H", f.read(2))[0]
            waveforms[ch_id, :] = np.frombuffer(f.read(n_samp * 4), dtype=np.float32)
    return waveforms

def process_chunk(chunk_idx, master_chunk, map_75, map_85, offsets, save_dir, meta, run_config, board_info, time_tol):
    out_file = os.path.join(save_dir, f"{meta['acq_tag']}_synced_part{chunk_idx}.h5")
    stats = {'golden': 0, 'miss_75': 0, 'miss_85': 0, 'orphaned': 0, 'bytes': 0}
    
    n_samp_75 = board_info['75']['n_samp']
    n_samp_85 = board_info['85']['n_samp']
    nan_75 = np.full((board_info['75']['max_depth'], n_samp_75), np.nan, dtype=np.float32)
    nan_85 = np.full((board_info['85']['max_depth'], n_samp_85), np.nan, dtype=np.float32)
    
    idx_75, idx_85 = 0, 0 
    
    with h5py.File(out_file, 'w') as hf:
        for k, v in meta.items(): hf.attrs[k] = v
        
        datasets = {}
        for bid in ['72', '75', '85']:
            grp = hf.create_group(f"Board_{bid}")
            grp.attrs['sampling_period_ns'] = board_info[bid]['samp_period']
            grp.attrs['n_channels'] = board_info[bid]['n_chan']
            grp.attrs['n_samples'] = board_info[bid]['n_samp']
            grp.attrs['array_depth'] = board_info[bid]['max_depth']
            
            det_types = []
            for ch in range(board_info[bid]['max_depth']):
                if ch in board_info[bid]['caen_ids']:
                    det_types.append(dc.get_channel_type(get_global_channel(bid, ch), run_config))
                else:
                    det_types.append("NC")
            grp.attrs['detector_config'] = np.array([s.encode('utf-8') for s in det_types])
            
            datasets[bid] = grp.create_dataset("waveforms", shape=(0, board_info[bid]['max_depth'], board_info[bid]['n_samp']), maxshape=(None, board_info[bid]['max_depth'], board_info[bid]['n_samp']), dtype='f4')
        
        for ev_72 in master_chunk:
            m_ts = ev_72['ts']
            t_75, t_85 = m_ts + offsets['75'], m_ts + offsets['85']
            match_75, match_85 = None, None
            
            while idx_75 < len(map_75) and map_75[idx_75]['ts'] < t_75 - time_tol: idx_75 += 1
            if idx_75 < len(map_75) and abs(map_75[idx_75]['ts'] - t_75) <= time_tol:
                match_75 = map_75[idx_75]; idx_75 += 1
                
            while idx_85 < len(map_85) and map_85[idx_85]['ts'] < t_85 - time_tol: idx_85 += 1
            if idx_85 < len(map_85) and abs(map_85[idx_85]['ts'] - t_85) <= time_tol:
                match_85 = map_85[idx_85]; idx_85 += 1

            if match_75 and match_85: stats['golden'] += 1
            elif match_85 and not match_75: stats['miss_75'] += 1
            elif match_75 and not match_85: stats['miss_85'] += 1
            else: stats['orphaned'] += 1

            w_72 = fetch_waveform(ev_72, board_info['72']['max_depth'])
            w_75 = fetch_waveform(match_75, board_info['75']['max_depth']) if match_75 else nan_75
            w_85 = fetch_waveform(match_85, board_info['85']['max_depth']) if match_85 else nan_85
            
            bytes_proc = (w_72.size + w_75.size + w_85.size) * 4
            stats['bytes'] += bytes_proc
            if progress_queue: progress_queue.put(bytes_proc)
            
            ns = datasets['72'].shape[0] + 1
            datasets['72'].resize((ns, datasets['72'].shape[1], datasets['72'].shape[2])); datasets['72'][-1] = w_72
            datasets['75'].resize((ns, datasets['75'].shape[1], datasets['75'].shape[2])); datasets['75'][-1] = w_75
            datasets['85'].resize((ns, datasets['85'].shape[1], datasets['85'].shape[2])); datasets['85'][-1] = w_85

    return stats, out_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help="Target Acq Directory")
    parser.add_argument('--output', '-o', default='./HDF5_Output', help="Destination Directory")
    parser.add_argument('--chunk_size', '-c', type=int, default=250, help="Events per HDF5 chunk")
    parser.add_argument('--limit', '-l', type=int, default=0, help="Max master events to process")
    parser.add_argument('--time_tol', '-t', type=int, default=50, help="Timestamp diff tolerance")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Concurrent workers")
    args = parser.parse_args()

    full_path = os.path.normpath(args.input_dir)
    meta = extract_metadata(full_path)
    all_files = glob.glob(os.path.join(args.input_dir, '*.bin'))
    
    board_files = {'72': [], '75': [], '85': []}
    for f in all_files:
        if '192.168.0.72' in f: board_files['72'].append(f)
        elif '192.168.0.75' in f: board_files['75'].append(f)
        elif '192.168.0.85' in f: board_files['85'].append(f)
        
    for k in board_files: board_files[k].sort()

    print(f"\n=== LArTPC Coincidence Event Builder ===")
    
    board_info = {}
    print("\n--- Digitizer Info ---")
    for bid in ['72', '75', '85']:
        if board_files[bid]:
            chans, samp_period, samps, max_depth, caen_ids = scan_board_metadata(board_files[bid][0])
            board_info[bid] = {'n_chan': chans, 'samp_period': samp_period, 'n_samp': samps, 'max_depth': max_depth, 'caen_ids': caen_ids}
            rate_mhz = 1000.0 / samp_period if samp_period > 0 else 0
            print(f"Board {bid}: {chans:02d} Active Channels (Depth: {max_depth:02d}) | {rate_mhz:05.1f} MS/s | {samps} Samples/Evt")
            
    offsets = gt.calculate_daq_offsets(board_files['72'][0], board_files['75'][0], board_files['85'][0])
    
    print(f"\n=== Phase 1: Metadata Mapping ===")
    t_map = time.time()
    
    map_72 = map_board_files(board_files['72'], '72', "Master (72)", limit=args.limit)
    
    target_ts = None
    if args.limit > 0 and map_72:
        target_ts = map_72[-1]['ts'] + 500000 
        
    map_85 = map_board_files(board_files['85'], '85', "Slave 1(85)", target_ts=target_ts)
    map_75 = map_board_files(board_files['75'], '75', "Slave 2(75)", target_ts=target_ts)
    
    print(f"Mapped {len(map_72)} Master events in {time.time() - t_map:.2f} s")

    chunks = [map_72[i:i + args.chunk_size] for i in range(0, len(map_72), args.chunk_size)]
    
    # --- AUTOMATED HIERARCHICAL FOLDER ROUTING ---
    save_dir = os.path.join(os.path.abspath(args.output), meta['run_config'], meta['hv_tag'], meta['acq_tag'])
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nRouting Data to: {save_dir}")

    print(f"\n=== Phase 2: Multithreaded Coincidence Building ===")
    m = mp.Manager()
    q = m.Queue()
    
    est_bytes = len(map_72) * ((board_info['72']['max_depth'] * board_info['72']['n_samp']) + 
                               (board_info['75']['max_depth'] * board_info['75']['n_samp']) + 
                               (board_info['85']['max_depth'] * board_info['85']['n_samp'])) * 4

    script_start = time.time()
    final_stats = {'golden': 0, 'miss_75': 0, 'miss_85': 0, 'orphaned': 0, 'bytes': 0}
    generated_files = []

    with tqdm(total=est_bytes, unit='B', unit_scale=True, desc="Building") as pbar:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(q,)) as exc:
            futures = [exc.submit(process_chunk, i, chunks[i], map_75, map_85, offsets, save_dir, meta, meta['run_config'], board_info, args.time_tol) for i in range(len(chunks))]
            
            while any(not f.done() for f in futures):
                while not q.empty(): pbar.update(q.get())
                time.sleep(0.1)
            while not q.empty(): pbar.update(q.get())
                
            for f in as_completed(futures):
                s, fname = f.result()
                generated_files.append(fname)
                for k in final_stats: final_stats[k] += s[k]

    t_total = time.time() - script_start
    mb_proc = final_stats['bytes'] / (1024*1024)
    rate_mb = mb_proc / t_total if t_total > 0 else 0
    rate_evt = len(map_72) / t_total if t_total > 0 else 0

    print("\n" + "="*60)
    print("      DAQ COINCIDENCE & PERFORMANCE REPORT")
    print("="*60)
    print(f"Total Master Triggers: {len(map_72)}")
    print(f"Golden Events:         {final_stats['golden']} ({(final_stats['golden']/len(map_72))*100:.1f}%)")
    print(f"Partial Drop (Board 75 Missed):   {final_stats['miss_75']} ({(final_stats['miss_75']/len(map_72))*100:.1f}%)")
    print(f"Partial Drop (Board 85 Missed):   {final_stats['miss_85']} ({(final_stats['miss_85']/len(map_72))*100:.1f}%)")
    print(f"Orphaned (Both Slaves Missed):    {final_stats['orphaned']} ({(final_stats['orphaned']/len(map_72))*100:.1f}%)")
    print("-" * 60)
    print(f"Total Conversion Time: {t_total:.2f} s")
    print(f"Effective Rate:        {rate_mb:.2f} MB/s ({rate_evt:.1f} Evts/s)")
    print("="*60)

if __name__ == "__main__":
    main()