"""
04_Bin2HDF_Timestamp_Parallel.py
Multi-Board Synchronized Binary to HDF5 Converter
- Strict Metadata (No phantom attributes)
- Timestamp-based Split Grouping
- ProcessPoolExecutor for concurrent processing
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import struct
import numpy as np
import h5py
import re
import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import detector_config as dc

def get_board_id(filename):
    match = re.search(r'192\.168\.0\.(\d+)', filename)
    return match.group(1) if match else "unknown"

def get_split_timestamp(filename):
    """Extracts the 14-digit timestamp to group the synchronized board files."""
    match = re.search(r'_(\d{14})-\d+\.bin$', filename)
    return match.group(1) if match else "unknown_time"

def get_global_channel(bid, local_ch):
    if bid == '72': return local_ch
    elif bid == '75': return local_ch + 32
    elif bid == '85': return local_ch + 96
    return local_ch

def extract_metadata(dir_name):
    dir_lower = dir_name.lower()
    meta = {}
    meta['run_config'] = dc.detect_run_mode(dir_name)
    
    acq_match = re.search(r'(?i)acq(\d+)', dir_name)
    meta['acq_tag'] = f"acq{acq_match.group(1)}" if acq_match else "acq_unknown"
    
    if 'pedestal' in dir_lower:
        meta['hv_tag'] = 'Pedestal'
    else:
        hv_match = re.search(r'TPCHV(\d+)', dir_name)
        meta['hv_tag'] = f"TPCHV{hv_match.group(1)}" if hv_match else 'Data'
    return meta

def convert_single_split(split_ts, file_list, dir_name, output_root, pt_dict):
    """Worker function: Processes 3 board files grouped by their start timestamp."""
    meta = extract_metadata(dir_name)
    
    save_dir = os.path.join(output_root, meta['run_config'], meta['hv_tag'])
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{meta['acq_tag']}_{split_ts}.h5")
    
    header_format = "<IQIQi"
    header_size = struct.calcsize(header_format)
    
    total_bytes_processed = 0
    events_processed = 0
    
    with h5py.File(output_file, 'w') as hf:
        # Write only genuine directory-derived metadata
        for k, v in meta.items():
            hf.attrs[k] = v
        hf.attrs['split_timestamp'] = split_ts
        
        for filepath in file_list:
            bid = get_board_id(filepath)
            file_size = os.path.getsize(filepath)
            total_bytes_processed += file_size
            
            grp = hf.create_group(f"Board_{bid}")
            dset_waveforms, dset_ids, dset_timestamps = None, None, None
            
            # ADVISOR RULE: Only write pre-trigger if explicitly passed
            if pt_dict and bid in pt_dict:
                grp.attrs['pre_trigger_us'] = pt_dict[bid]
            
            with open(filepath, 'rb') as f:
                while True:
                    header_bytes = f.read(header_size)
                    if not header_bytes or len(header_bytes) < header_size: break 
                        
                    ev_num, timestamp, n_samples, sampling_period, n_channels = struct.unpack(header_format, header_bytes)
                    
                    if dset_waveforms is None:
                        chunk_shape = (1, n_channels, n_samples)
                        dset_waveforms = grp.create_dataset("waveforms", shape=(0, n_channels, n_samples), maxshape=(None, n_channels, n_samples), dtype='f4', chunks=chunk_shape)
                        dset_ids = grp.create_dataset("event_ids", shape=(0,), maxshape=(None,), dtype='i4')
                        dset_timestamps = grp.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype='u8')
                        
                        grp.attrs['sampling_period_ns'] = sampling_period
                        grp.attrs['n_channels'] = n_channels
                        grp.attrs['n_samples'] = n_samples
                        
                        det_types = [dc.get_channel_type(get_global_channel(bid, ch), meta['run_config']) for ch in range(n_channels)]
                        grp.attrs['detector_config'] = np.array([s.encode('utf-8') for s in det_types])
                        
                    event_waveforms = np.zeros((n_channels, n_samples), dtype=np.float32)
                    for i in range(n_channels):
                        f.read(2) 
                        wave_bytes = f.read(n_samples * 4)
                        if len(wave_bytes) < n_samples * 4: break
                        event_waveforms[i, :] = np.frombuffer(wave_bytes, dtype=np.float32)
                        
                    new_size = dset_ids.shape[0] + 1
                    dset_waveforms.resize((new_size, n_channels, n_samples))
                    dset_ids.resize((new_size,))
                    dset_timestamps.resize((new_size,))
                    
                    dset_waveforms[new_size-1] = event_waveforms
                    dset_ids[new_size-1] = ev_num
                    dset_timestamps[new_size-1] = timestamp
                    
                    if bid == '72': # Track event count against the master
                        events_processed += 1

    mb_processed = total_bytes_processed / (1024 * 1024)
    return f"[SUCCESS] Split TS:{split_ts} -> {events_processed} evts | {mb_processed:.1f} MB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help="Target Acq Directory containing the .bin files")
    parser.add_argument('--output', '-o', default='./HDF5_Output', help="Destination Directory")
    parser.add_argument('--pre_trigger', '-pt', type=float, nargs='*', default=None, 
                        help="Pre-trigger window in us. Pass 1 value for all boards, or 3 values for [72, 75, 85]. Omit to ignore.")
    parser.add_argument('--limit', '-l', type=int, default=0, help="Max number of split groups to process (e.g., 2 = ~200 events)")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Number of concurrent splits to process")
    args = parser.parse_args()
    
    # Process Pre-Trigger Arguments
    pt_dict = {}
    if args.pre_trigger is not None:
        if len(args.pre_trigger) == 1:
            pt_dict = {'72': args.pre_trigger[0], '75': args.pre_trigger[0], '85': args.pre_trigger[0]}
        elif len(args.pre_trigger) == 3:
            pt_dict = {'72': args.pre_trigger[0], '75': args.pre_trigger[1], '85': args.pre_trigger[2]}
        else:
            print("[WARN] --pre_trigger must have exactly 1 or 3 values. Ignoring.")
    
    dir_name = os.path.basename(os.path.normpath(args.input_dir))
    all_files = glob.glob(os.path.join(args.input_dir, '*.bin'))
    
    # Group files by timestamp
    splits = {}
    for f in all_files:
        ts = get_split_timestamp(f)
        if ts not in splits: splits[ts] = []
        splits[ts].append(f)
        
    split_timestamps = sorted(list(splits.keys()))
    
    if args.limit > 0:
        split_timestamps = split_timestamps[:args.limit]
        print(f"--- LIMIT ENGAGED: Processing first {args.limit} split groups ---")
        
    print(f"=== Multi-Board Sync Converter (Timestamp-Grouped) ===")
    print(f"Target Acq:      {dir_name}")
    print(f"Splits Found:    {len(split_timestamps)}")
    print(f"Pre-Trigger:     {pt_dict if pt_dict else 'Omitted (Strict Mode)'}")
    print(f"Workers:         {args.workers}")
    print("-" * 60)
    
    script_start = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(convert_single_split, ts, splits[ts], dir_name, args.output, pt_dict) for ts in split_timestamps]
        for future in as_completed(futures):
            print(future.result())

    print("-" * 60)
    print(f"Total Time: {time.time() - script_start:.2f} s")

if __name__ == "__main__":
    main()