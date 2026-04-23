"""
15_calc_baselines_MultiBoard_Fast.py
Calculates Baseline Mean and RMS Noise for Multi-Board LArTPC HDF5 Data
- Traverses Board_72, Board_75, Board_85 groups.
- Gracefully handles NaN-padded dropped events.
- Full tqdm progress bar with effective MB/s readout.
"""

import h5py
import numpy as np
import os
import argparse
import time
import gc
from multiprocessing import Pool
from tqdm import tqdm
import grams_tools as gt 

def process_file(task):
    filepath, pre_trigger_us, overwrite = task
    filename = os.path.basename(filepath)
    events_processed = 0
    file_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    try:
        with h5py.File(filepath, 'r+') as h5_file:
            fname_lower = filename.lower()
            use_full_trace_rms = 'osaka' in fname_lower or 'ups' in fname_lower
            
            boards = ['Board_72', 'Board_75', 'Board_85']
            for board_name in boards:
                if board_name not in h5_file: continue
                grp = h5_file[board_name]
                
                if 'baseline_mean' in grp and not overwrite: continue

                dset_waveforms = grp['waveforms']
                n_events, n_ch, n_samples = dset_waveforms.shape
                
                res_ns = grp.attrs.get('sampling_period_ns', 2.0)
                n_pre_samples = gt.get_pre_trigger_samples(res_ns, safe_window_us=pre_trigger_us)
                if n_pre_samples > n_samples: n_pre_samples = n_samples

                bytes_per_event = n_ch * n_samples * 4
                chunk_size = max(1, int((200 * 1024 * 1024) / bytes_per_event))
                chunk_size = min(chunk_size, 1000)

                if 'detector_config' in grp.attrs:
                    det_config = [x.decode('utf-8') for x in grp.attrs['detector_config']]
                else: det_config = ["SiPM"] * n_ch

                if 'baseline_mean' in grp: del grp['baseline_mean']
                if 'baseline_std' in grp: del grp['baseline_std']
                
                dset_bl = grp.create_dataset('baseline_mean', shape=(n_events, n_ch), dtype=np.float32)
                dset_std = grp.create_dataset('baseline_std', shape=(n_events, n_ch), dtype=np.float32)
                
                for start_idx in range(0, n_events, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_events)
                    wave_chunk = dset_waveforms[start_idx:end_idx]
                    
                    bl_chunk = np.zeros((end_idx - start_idx, n_ch), dtype=np.float32)
                    std_chunk = np.zeros((end_idx - start_idx, n_ch), dtype=np.float32)
                    
                    for i in range(len(wave_chunk)): 
                        for j in range(n_ch):
                            if np.isnan(wave_chunk[i, j, 0]):
                                bl_chunk[i, j], std_chunk[i, j] = np.nan, np.nan
                                continue

                            segment_pt = wave_chunk[i, j, :n_pre_samples]
                            if "Charge" in det_config[j]:
                                bl = gt.baseline_charge(segment_pt, fine_tune_window=2.0)
                            else:
                                bl = gt.baseline_iterative_clipping_RT(segment_pt, sigma=2.5)
                            
                            rms_segment = wave_chunk[i, j, :] if use_full_trace_rms else segment_pt 
                            bl_chunk[i, j] = bl
                            std_chunk[i, j] = np.std(rms_segment)
                    
                    dset_bl[start_idx:end_idx] = bl_chunk
                    dset_std[start_idx:end_idx] = std_chunk
                    del wave_chunk, bl_chunk, std_chunk
                    
                grp.attrs['baseline_pre_trigger_us'] = pre_trigger_us
                if board_name == 'Board_72': events_processed = n_events
                gc.collect()

    except Exception as e:
        print(f"\n[FAIL] {filename}: {e}")
        return (filename, 0, 0)
        
    return (filename, events_processed, file_mb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--pre_trigger', '-pt', type=float, required=True, help="Pre-trigger in us")
    parser.add_argument('--overwrite', '-f', action='store_true')
    parser.add_argument('--workers', '-w', type=int, default=4)
    args = parser.parse_args()
    
    tasks = []
    if os.path.isfile(args.input_dir):
        tasks.append((args.input_dir, args.pre_trigger, args.overwrite))
    else:
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith('.h5'):
                    tasks.append((os.path.join(root, file), args.pre_trigger, args.overwrite))
    
    if not tasks:
        print("No HDF5 files found to process.")
        return
        
    total_mb_to_process = sum(os.path.getsize(t[0]) for t in tasks) / (1024 * 1024)
    print(f"Calculating Multi-Board Baselines: {len(tasks)} files | {total_mb_to_process:.1f} MB")
    
    start_time = time.time()
    total_events = 0
    
    active_workers = min(args.workers, 8)
    with Pool(active_workers) as p:
        # Use imap_unordered so we can update tqdm sequentially as chunks finish
        iterator = p.imap_unordered(process_file, tasks)
        with tqdm(total=total_mb_to_process, unit='MB', desc="Processing") as pbar:
            for fname, count, file_mb in iterator:
                total_events += count
                pbar.update(file_mb)
                
    total_time = time.time() - start_time
    rate_mb = total_mb_to_process / total_time if total_time > 0 else 0
    
    print("-" * 60)
    print(f"Complete:       {total_events} total events processed.")
    print(f"Total Time:     {total_time:.2f} s")
    print(f"Effective Rate: {rate_mb:.2f} MB/s")
    print("-" * 60)

if __name__ == "__main__":
    main()