"""
01_Bin2HDF_Standalone_FastCAEN.py
Streamlined Event Builder for Standalone Single-Digitizer Data
- Natively designed for the 32-channel Fast CAEN (Board 72).
- Bypasses all multi-board syncing for maximum conversion speed.
- Upgraded: Custom Output Directory routing (--out_dir) with auto 'acqX' sorting.
- Upgraded: Hard stop event limits for rapid pipeline testing (--max_events).
- Fix: Perfectly aligned to the <IQIQi custom binary header and float32 payloads.
- Fix: Mathematical Offset Mapping to completely bypass NAS I/O latency bottlenecks.
- Update: ALL COMPRESSION REMOVED. 1:1 Raw data footprint.
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

progress_queue = None

def init_worker(q):
    global progress_queue
    progress_queue = q

def extract_metadata(full_path):
    """Sniffs the entire absolute path to capture parent-folder metadata."""
    meta = {}
    meta['run_config'] = dc.detect_run_mode(full_path) 
    full_path_lower = full_path.lower()
    
    if 'pedestal' in full_path_lower: 
        meta['hv_tag'] = 'Pedestal'
    else:
        hv_match = re.search(r'(?i)HV(\d+)', full_path)
        meta['hv_tag'] = f"HV{hv_match.group(1)}" if hv_match else 'HV_Unknown'
        
    acq_match = re.search(r'(?i)acq(\d+)', full_path)
    meta['acq_tag'] = f"Acq{acq_match.group(1)}" if acq_match else os.path.basename(os.path.normpath(full_path))
    
    return meta

def locate_binary_files(target_dir):
    """Hunts for ALL custom .bin or .dat files in the target or immediate subdirectories."""
    all_files = []
    for ext in ["*.bin", "*.dat"]:
        all_files.extend(glob.glob(os.path.join(target_dir, ext)))
        
        sub_dirs = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        for d in sub_dirs:
            all_files.extend(glob.glob(os.path.join(d, ext)))
            
    return sorted(list(set(all_files)))

def map_binary_files(file_list, max_events=None):
    """
    Mathematically maps the offsets by checking only the first header of each file.
    This eliminates millions of network I/O requests over NAS storage.
    """
    header_format = "<IQIQi"
    header_size = struct.calcsize(header_format)
    offsets = []
    mapped_bytes = 0
    
    for filepath in file_list:
        if max_events is not None and len(offsets) >= max_events:
            break
            
        file_size = os.path.getsize(filepath)
        if file_size < header_size: continue
        
        # Read only the first 28 bytes to establish the geometry
        with open(filepath, 'rb') as f:
            h_bytes = f.read(header_size)
            
        if len(h_bytes) < header_size: continue
        
        _, _, n_samp, samp_period, n_chan = struct.unpack(header_format, h_bytes)
        payload_size = (n_chan * 2) + (n_chan * n_samp * 4)
        event_size = header_size + payload_size
        
        # Mathematically deduce event counts
        events_in_file = file_size // event_size
        
        for i in range(events_in_file):
            if max_events is not None and len(offsets) >= max_events:
                break
            
            # Record the absolute byte offset for this event
            absolute_offset = i * event_size
            offsets.append((filepath, absolute_offset, n_samp, samp_period, n_chan, payload_size))
            mapped_bytes += event_size
            
    return offsets, mapped_bytes

def process_chunk(task):
    chunk_events, out_filepath, meta, chunk_idx = task
    
    if not chunk_events:
        return ({'events': 0, 'bytes': 0}, out_filepath)
        
    _, _, n_samp, samp_period, n_chan, payload_size = chunk_events[0]
    n_events = len(chunk_events)
    header_format = "<IQIQi"
    header_size = struct.calcsize(header_format)
    
    waveforms = np.full((n_events, 32, n_samp), np.nan, dtype=np.float32)
    timestamps = np.zeros(n_events, dtype=np.uint64)
    bytes_processed = 0
    
    current_filepath = None
    f_bin = None
    
    for i, ev in enumerate(chunk_events):
        filepath, offset, ev_n_samp, ev_samp_period, ev_n_chan, ev_payload_size = ev
        
        if filepath != current_filepath:
            if f_bin is not None: f_bin.close()
            try:
                f_bin = open(filepath, 'rb')
                current_filepath = filepath
            except Exception as e:
                continue
                
        f_bin.seek(offset)
        h_bytes = f_bin.read(header_size)
        
        if len(h_bytes) == header_size:
            # The parallel worker now extracts the timestamp from the local header
            ev_num, ts, _, _, _ = struct.unpack(header_format, h_bytes)
            timestamps[i] = ts
            
            for ch_idx in range(ev_n_chan):
                ch_id = struct.unpack("<H", f_bin.read(2))[0]
                if ch_id < 32:
                    waveforms[i, ch_id, :] = np.frombuffer(f_bin.read(ev_n_samp * 4), dtype=np.float32)
                else:
                    f_bin.seek(ev_n_samp * 4, 1) 
                    
        bytes_processed += (header_size + ev_payload_size)
            
    if f_bin is not None:
        f_bin.close()
    
    with h5py.File(out_filepath, 'w') as h5f:
        h5f.attrs['run_config'] = meta['run_config']
        h5f.attrs['hv_state'] = meta['hv_tag']
        h5f.attrs['acq_time'] = meta['acq_tag']
        
        grp = h5f.create_group('Board_72')
        grp.attrs['sampling_period_ns'] = float(samp_period) if samp_period > 0 else 8.0 
        grp.attrs['n_samples'] = n_samp
        
        mapping = dc.get_channel_map(meta['run_config'])
        det_config = [dc.get_channel_type(ch, meta['run_config']) for ch in range(32)]
        grp.attrs['detector_config'] = [n.encode('utf-8') for n in det_config]
        
        grp.create_dataset('waveforms', data=waveforms)
        grp.create_dataset('timestamps', data=timestamps)
        
    if progress_queue:
        progress_queue.put(bytes_processed)
        
    return ({'events': n_events, 'bytes': bytes_processed}, out_filepath)

def main():
    parser = argparse.ArgumentParser(description="LArTPC Standalone Fast CAEN Event Builder")
    parser.add_argument('target_dir', help="Acquisition directory containing multiple .bin or .dat files")
    parser.add_argument('--events_per_file', '-e', type=int, default=1000, help="Events per HDF5 chunk")
    parser.add_argument('--workers', '-w', type=int, default=6, help="CPU cores for conversion")
    parser.add_argument('--out_dir', '-o', type=str, default=None, help="Custom output directory for HDF5 files")
    parser.add_argument('--max_events', '-m', type=int, default=None, help="Maximum total events to extract")
    args = parser.parse_args()

    bin_files = locate_binary_files(args.target_dir)
    if not bin_files:
        print(f"[FAIL] No binary .bin or .dat files found in {args.target_dir} or subdirectories.")
        return
        
    meta = extract_metadata(args.target_dir)
    
    print(f"\nStandalone Target: {args.target_dir} ({len(bin_files)} files found)")
    print(f"Detected Config:   {meta['run_config']} | {meta['hv_tag']} | {meta['acq_tag']}")
    
    if args.max_events:
        print(f"Testing Mode:      Hunting for first {args.max_events} events across all files...")
    else:
        print("Mapping binary file offsets... (Mathematically mapped for NAS acceleration)")
        
    script_start = time.time()
    event_map, total_bytes = map_binary_files(bin_files, max_events=args.max_events)
    
    if not event_map:
        print("[FAIL] Could not parse any events from binaries.")
        return
        
    print(f"Map Built: Found {len(event_map)} events.")
    
    base_out_dir = args.out_dir if args.out_dir else os.path.join(args.target_dir, "Converted_Standalone")
    acq_subfolder = meta['acq_tag'].lower()
    
    out_dir = os.path.join(base_out_dir, acq_subfolder)
    os.makedirs(out_dir, exist_ok=True)
    
    chunks = [event_map[i:i + args.events_per_file] for i in range(0, len(event_map), args.events_per_file)]
    
    tasks = []
    for i, chunk in enumerate(chunks):
        out_name = os.path.join(out_dir, f"{meta['acq_tag']}_standalone_part{i}.h5")
        tasks.append((chunk, out_name, meta, i))

    final_stats = {'events': 0, 'bytes': 0}
    m = mp.Manager()
    q = m.Queue()
    
    print(f"Deploying {args.workers} workers for HDF5 Conversion...")
    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Converting") as pbar:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(q,)) as executor:
            futures = [executor.submit(process_chunk, t) for t in tasks]
            
            while any(not f.done() for f in futures):
                while not q.empty(): pbar.update(q.get())
                time.sleep(0.1)
            while not q.empty(): pbar.update(q.get())
                
            for f in as_completed(futures):
                s, _ = f.result()
                for k in final_stats: final_stats[k] += s[k]

    t_total = time.time() - script_start
    mb_proc = final_stats['bytes'] / (1024*1024)
    rate_mb = mb_proc / t_total if t_total > 0 else 0
    rate_evt = final_stats['events'] / t_total if t_total > 0 else 0

    print("\n" + "="*60)
    print("      STANDALONE CONVERSION & PERFORMANCE REPORT")
    print("="*60)
    print(f"Total Events Converted: {final_stats['events']}")
    print(f"Total Data Processed:   {mb_proc:.1f} MB")
    print(f"Processing Time:        {t_total:.1f} seconds")
    print(f"Effective Rate:         {rate_mb:.1f} MB/s  ({rate_evt:.1f} Evt/s)")
    print(f"Output Directory:       {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()