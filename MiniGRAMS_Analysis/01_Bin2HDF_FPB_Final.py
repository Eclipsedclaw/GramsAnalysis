"""
01_Bin2HDF_FPB_Final.py
Binary to HDF5 Converter for MicroGRAMS/MiniGRAMS
-------------------------------------------------
- Upfront Fast-Scan: Instant Event Counting & Hardware Specs
- ABSOLUTE CAEN INDEXING: HDF5 Array Index == True CAEN Channel ID.
- Missing/Unrecorded channels are padded with NaNs.
- Fixes NAS Locking Errors (HDF5_USE_FILE_LOCKING = FALSE).
- Flattened WV_Tucson directory output.
- Bridged directly to TUCSON_WV mapping.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import struct
import numpy as np
import h5py
import re
import argparse
import time
from multiprocessing import Pool, cpu_count, Value, Lock, Manager
from tqdm import tqdm
import detector_config as dc

counter = None
total_files = None
print_lock = None

def init_globals(c, t, l):
    global counter, total_files, print_lock
    counter = c
    total_files = t
    print_lock = l

def extract_metadata(filename):
    base = os.path.basename(filename)
    base_lower = base.lower()
    meta = {}
    
    # --- Identify acquisition tag ---
    acq_match = re.search(r'(?i)acq(\d+)', base)
    type_match = re.search(r'(?i)(pedestal\d+|sig\d+|calib\d+|run\d+)', base)
    if acq_match: meta['acq_tag'] = f"acq{acq_match.group(1)}"
    elif type_match: meta['acq_tag'] = type_match.group(1).lower()
    else:
        ts_match = re.search(r'(\d{14})', base)
        meta['acq_tag'] = f"acq_{ts_match.group(1)}" if ts_match else f"acq_unknown_{base}"

    # --- Identify Run Config ---
    if 'osaka' in base_lower and 'lightstudy' not in base_lower:
        meta['run_config'] = 'OSAKA'
        hv_match = re.search(r'TPCHV(\d+)', base)
        meta['hv_tag'] = f"TPCHV{hv_match.group(1)}" if hv_match else ('PedestalDataConverted' if 'pedestal' in base_lower else 'Unknown_Osaka')
        meta['field_on'] = bool(hv_match)
    
    # BRIDGE FOR BOSTON DATA
    elif 'lightstudy' in base_lower:
        meta['run_config'] = 'MICROG_PURITY_STUDY'
        hv_match = re.search(r'(\d+)tpchv', base_lower) 
        if hv_match:
            meta['hv_tag'] = f"TPCHV{hv_match.group(1)}"
            meta['field_on'] = int(hv_match.group(1)) > 0
        else:
            meta['hv_tag'] = 'UnknownHV'
            meta['field_on'] = False

    # BRIDGE FOR TUCSON DATA
    elif 'wv_' in base_lower:
        meta['run_config'] = 'TUCSON_WV'
        meta['hv_tag'] = 'WV_Tucson'
        meta['field_on'] = False

    elif 'ups' in base_lower:
        meta['run_config'] = 'UPS2' if 'ups2' in base_lower else 'UPS1'
        hv_match = re.search(r'TPCHV(\d+)', base)
        meta['hv_tag'] = f"TPCHV{hv_match.group(1)}" if hv_match else ('PedestalDataConverted' if 'pedestal' in base_lower else 'Unknown_UPS')
        meta['field_on'] = bool(hv_match)

    elif 'light-pedestal' in base_lower or 'light-sig' in base_lower or 'combo' in base_lower:
        meta['run_config'] = 'combo' if 'combo' in base_lower else 'light_only'
        hv_match = re.search(r'TPCHV(\d+)', base)
        meta['hv_tag'] = f"TPCHV{hv_match.group(1)}" if hv_match else ('PedestalDataConverted' if 'pedestal' in base_lower else 'ScintDataConverted')
        meta['field_on'] = bool(hv_match)
    else:
        meta['run_config'] = 'light_only'
        meta['hv_tag'] = 'UnknownConfig'; meta['field_on'] = False

    dig_match = re.search(r'(dig\d+)', base)
    meta['dig_id'] = dig_match.group(1) if dig_match else ('dig2' if 'dig2' in base else 'unknown_dig')
    ts_match = re.search(r'(\d{14})', base)
    meta['timestamp'] = ts_match.group(1) if ts_match else "00000000000000"

    return meta

def convert_single_file(task):
    input_path, output_root, file_limit, pt_window_us, shared_dict = task
    filename = os.path.basename(input_path)
    file_start_time = time.time()
    
    try:
        file_size_bytes = os.path.getsize(input_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
    except:
        file_size_mb = 0

    meta = extract_metadata(filename)
    
    # --- FLATTEN DIRECTORY FOR TUCSON DATA ---
    if meta['hv_tag'] == 'WV_Tucson':
        save_dir = os.path.join(output_root, meta['hv_tag'])
    else:
        save_dir = os.path.join(output_root, meta['hv_tag'], meta['acq_tag'])
        
    os.makedirs(save_dir, exist_ok=True)
    output_name = f"{meta['timestamp']}_{meta['run_config']}_{meta['hv_tag']}_{meta['acq_tag']}.h5"
    save_path = os.path.join(save_dir, output_name)

    events_processed = 0

    try:
        with h5py.File(save_path, 'w') as hf:
            dset_waveforms = None
            dset_ids = None
            dset_timestamps = None
            
            for k, v in meta.items():
                hf.attrs[k] = v
                
            if pt_window_us > 0:
                hf.attrs['baseline_pre_trigger_us'] = pt_window_us
            
            with open(input_path, 'rb') as f:
                # FAST SCAN: Read first event to determine True CAEN Hardware Limits
                header_bytes = f.read(28)
                if not header_bytes or len(header_bytes) < 28: return 0.0, 0
                    
                ev_num, timestamp, n_samples, resolution, n_channels_recorded = struct.unpack("<IQIQi", header_bytes)
                
                # Sniff the true CAEN channels
                caen_ids = []
                for _ in range(n_channels_recorded):
                    ch_id = struct.unpack("<H", f.read(2))[0]
                    caen_ids.append(ch_id)
                    f.seek(n_samples * 4, 1) # skip waveform bytes
                
                # Array depth must accommodate the highest physical CAEN channel
                max_caen_ch = max(caen_ids) + 1 
                
                # Reset file pointer to beginning of data to actually read the events
                f.seek(0)
                
                chunk_shape = (1, max_caen_ch, n_samples)
                dset_waveforms = hf.create_dataset("waveforms_mV", 
                                                   shape=(0, max_caen_ch, n_samples), 
                                                   maxshape=(None, max_caen_ch, n_samples),
                                                   dtype='f4', 
                                                   chunks=chunk_shape)
                dset_ids = hf.create_dataset("event_ids", shape=(0,), maxshape=(None,), dtype='i4')
                dset_timestamps = hf.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype='u8')
                
                hf.attrs['resolution_ns'] = resolution
                det_types = [dc.get_channel_type(ch, meta['run_config']) for ch in range(max_caen_ch)]
                hf.attrs['detector_config'] = np.array([s.encode('utf-8') for s in det_types])

                while True:
                    if file_limit > 0 and events_processed >= file_limit: break

                    header_bytes = f.read(28)
                    if not header_bytes or len(header_bytes) < 28: break 
                        
                    ev_num, timestamp, n_samples, resolution, n_channels_recorded = struct.unpack("<IQIQi", header_bytes)
                    
                    # Fill array with NaNs to protect downstream scripts
                    event_waveforms = np.full((max_caen_ch, n_samples), np.nan, dtype=np.float32)
                    
                    for _ in range(n_channels_recorded):
                        caen_id = struct.unpack("<H", f.read(2))[0]
                        bytes_to_read = n_samples * 4
                        wave_bytes = f.read(bytes_to_read)
                        if len(wave_bytes) < bytes_to_read: break
                        
                        # PLACE WAVEFORM AT TRUE CAEN INDEX
                        event_waveforms[caen_id, :] = np.frombuffer(wave_bytes, dtype=np.float32)

                    new_size = dset_ids.shape[0] + 1
                    dset_waveforms.resize((new_size, max_caen_ch, n_samples))
                    dset_ids.resize((new_size,))
                    dset_timestamps.resize((new_size,))
                    
                    dset_waveforms[new_size-1] = event_waveforms
                    dset_ids[new_size-1] = ev_num
                    dset_timestamps[new_size-1] = timestamp
                    
                    events_processed += 1
            
            return file_size_mb, events_processed

    except Exception as e:
        with print_lock: print(f"[FAIL] {filename}: {e}")
        return 0.0, events_processed

def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Input file or directory")
    parser.add_argument('--output', '-o', default='./HDF5_Output', help="Destination Directory")
    parser.add_argument('--workers', '-w', type=int, default=cpu_count(), help="Cores")
    parser.add_argument('--limit', '-l', type=int, default=0, help="Max events per file")
    parser.add_argument('--events', '-e', type=int, default=0, help="HARD global limit")
    parser.add_argument('--pre_trigger', '-pt', type=float, default=0.0, help="Pre-trigger window in us")
    args = parser.parse_args()
    
    if not os.path.exists(args.output): os.makedirs(args.output)
    
    file_list = []
    if os.path.isfile(args.input):
         if args.input.endswith(('.bin', '.dat')): file_list.append(args.input)
    elif os.path.isdir(args.input):
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.endswith(('.bin', '.dat')): file_list.append(os.path.join(root, file))
    
    total_count = len(file_list)
    file_list.sort()
    
    print(f"=== Binary to HDF5 Converter (ABSOLUTE CAEN INDEXING) ===")
    print(f"Files Found: {total_count}")
    
    total_size_mb = 0.0
    total_events = 0
    
    if total_count > 0:
        first_file = file_list[0]
        meta = extract_metadata(first_file)
        
        with open(first_file, 'rb') as f:
            header_bytes = f.read(28)
            if header_bytes and len(header_bytes) == 28:
                _, _, n_samples, resolution, n_channels = struct.unpack("<IQIQi", header_bytes)
                caen_ids = []
                for _ in range(n_channels):
                    caen_ids.append(struct.unpack("<H", f.read(2))[0])
                    f.seek(n_samples * 4, 1)
                
                print(f"--- Fast Scan Geometry ---")
                print(f"Recorded Channels: {n_channels}")
                print(f"True CAEN IDs:     {caen_ids}")
                print(f"HDF5 Array Depth:  {max(caen_ids) + 1} (Missing channels padded with NaNs)")
                print("-" * 58)

        if args.events > 0:
            c_val = Value('i', 0); t_val = Value('i', total_count); l_val = Lock()
            init_globals(c_val, t_val, l_val)
            shared_stats = {'initialized': False}
            events_remaining = args.events
            
            with tqdm(total=total_count, desc="Converting", unit="file") as pbar:
                for f in file_list:
                    current_limit = args.limit if (0 < args.limit < events_remaining) else events_remaining
                    file_mb, file_evts = convert_single_file((f, args.output, current_limit, args.pre_trigger, shared_stats))
                    total_size_mb += file_mb; total_events += file_evts; events_remaining -= file_evts
                    pbar.update(1)
                    if events_remaining <= 0: break
                    
        else:
            manager = Manager(); shared_stats = manager.dict(); shared_stats['initialized'] = False
            tasks = [(f, args.output, args.limit, args.pre_trigger, shared_stats) for f in file_list]
            c_val = Value('i', 0); t_val = Value('i', total_count); l_val = Lock()
            
            with Pool(args.workers, initializer=init_globals, initargs=(c_val, t_val, l_val)) as p:
                results = []
                with tqdm(total=total_count, desc="Converting", unit="file") as pbar:
                    for r in p.imap_unordered(convert_single_file, tasks):
                        results.append(r); pbar.update(1)
                
            total_size_mb = sum(r[0] for r in results)
            total_events = sum(r[1] for r in results)
            
    total_time = time.time() - script_start_time
    effective_rate = total_size_mb / total_time if total_time > 0 else 0
    print("-" * 58)
    print(f"Total Time:      {total_time:.2f} s")
    print(f"Total Data:      {total_size_mb:.2f} MB")
    print(f"Effective Rate:  {effective_rate:.2f} MB/s")
    print("-" * 58)

if __name__ == "__main__":
    main()