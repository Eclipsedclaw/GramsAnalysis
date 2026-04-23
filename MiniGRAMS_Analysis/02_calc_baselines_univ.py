"""
02_calc_baselines.py
Calculates Baseline Mean and RMS Noise for Universal LArTPC HDF5 Data
- Supports Single-Board (MicroGRAMS) and Multi-Board (MiniGRAMS) formats.
- Dynamically routes Charge vs SiPM baseline algorithms.
- Gracefully handles NaN-padded dropped events.
- Full tqdm progress bar with effective MB/s readout.
- Updated: Configurable start/end points for baseline region.
- Updated: True RMS noise calculation relative to robust baseline.
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
    filepath, pt_start_us, pt_end_us, overwrite = task
    filename = os.path.basename(filepath)
    events_processed = 0
    file_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    try:
        with h5py.File(filepath, 'r+') as h5_file:
            fname_lower = filename.lower()
            # If the run type requires full trace RMS instead of just pre-trigger
            use_full_trace_rms = 'osaka' in fname_lower or 'ups' in fname_lower
            
            # --- DYNAMIC DAQ ROUTING ---
            targets = []
            if 'waveforms_mV' in h5_file:
                # Single-Board (MicroGRAMS)
                targets.append((h5_file, 'waveforms_mV', 'baseline_mean_mV', 'baseline_std_mV'))
            else:
                # Multi-Board (MiniGRAMS)
                for b in ['Board_72', 'Board_75', 'Board_85']:
                    if b in h5_file:
                        targets.append((h5_file[b], 'waveforms', 'baseline_mean', 'baseline_std'))
            
            # --- PROCESS EACH TARGET GROUP ---
            for grp, wave_key, bl_key, std_key in targets:
                # Skip if already processed and not overwriting
                if bl_key in grp and not overwrite: 
                    continue

                dset_waveforms = grp[wave_key]
                n_events, n_ch, n_samples = dset_waveforms.shape
                
                # Get resolution (handles both naming conventions)
                res_ns = grp.attrs.get('resolution_ns', grp.attrs.get('sampling_period_ns', 8.0))
                
                # Calculate window indices for the baseline region
                start_idx_pt = int((pt_start_us * 1000.0) / res_ns)
                end_idx_pt = int((pt_end_us * 1000.0) / res_ns)
                
                # Bounds checking
                start_idx_pt = max(0, min(start_idx_pt, n_samples - 1))
                end_idx_pt = max(start_idx_pt + 1, min(end_idx_pt, n_samples))

                # Dynamic memory chunking to avoid RAM blowouts
                bytes_per_event = n_ch * n_samples * 4
                chunk_size = max(1, int((200 * 1024 * 1024) / bytes_per_event))
                chunk_size = min(chunk_size, 1000)

                # Fetch detector configuration for algorithm routing
                if 'detector_config' in grp.attrs:
                    det_config = [x.decode('utf-8') if isinstance(x, bytes) else x for x in grp.attrs['detector_config']]
                else: 
                    det_config = ["SiPM"] * n_ch

                # Clean up old datasets if overwriting
                if bl_key in grp: del grp[bl_key]
                if std_key in grp: del grp[std_key]
                
                # Initialize new datasets
                dset_bl = grp.create_dataset(bl_key, shape=(n_events, n_ch), dtype=np.float32)
                dset_std = grp.create_dataset(std_key, shape=(n_events, n_ch), dtype=np.float32)
                
                for start_idx in range(0, n_events, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_events)
                    wave_chunk = dset_waveforms[start_idx:end_idx]
                    
                    bl_chunk = np.zeros((end_idx - start_idx, n_ch), dtype=np.float32)
                    std_chunk = np.zeros((end_idx - start_idx, n_ch), dtype=np.float32)
                    
                    for i in range(len(wave_chunk)): 
                        for j in range(n_ch):
                            # Handle Missing/Dropped events (NaN padded)
                            if np.isnan(wave_chunk[i, j, 0]):
                                bl_chunk[i, j], std_chunk[i, j] = np.nan, np.nan
                                continue

                            # Slice specific isolation window
                            segment_pt = wave_chunk[i, j, start_idx_pt:end_idx_pt]
                            
                            # Route algorithm based on hardware
                            if "Charge" in det_config[j]:
                                bl = gt.baseline_charge(segment_pt, fine_tune_window=2.0)
                            else:
                                bl = gt.baseline_iterative_clipping_RT(segment_pt, sigma=2.5)
                            
                            rms_segment = wave_chunk[i, j, :] if use_full_trace_rms else segment_pt 
                            bl_chunk[i, j] = bl
                            # True RMS calculation relative to robust baseline
                            std_chunk[i, j] = np.sqrt(np.mean((rms_segment - bl)**2))
                    
                    dset_bl[start_idx:end_idx] = bl_chunk
                    dset_std[start_idx:end_idx] = std_chunk
                    del wave_chunk, bl_chunk, std_chunk
                    
                grp.attrs['baseline_pre_trigger_start_us'] = pt_start_us
                grp.attrs['baseline_pre_trigger_end_us'] = pt_end_us
                
                if events_processed == 0:
                    events_processed = n_events
                    
                gc.collect()

    except Exception as e:
        print(f"\n[FAIL] {filename}: {e}")
        return (filename, 0, 0)
        
    return (filename, events_processed, file_mb)

def main():
    parser = argparse.ArgumentParser(description="Universal Baseline Calculator")
    parser.add_argument('input_dir', help="Target Directory or HDF5 File")
    parser.add_argument('--pt_start', '-pts', type=float, default=0.0, help="Pre-trigger window start in us")
    parser.add_argument('--pt_end', '-pte', type=float, required=True, help="Pre-trigger window end in us")
    parser.add_argument('--overwrite', '-f', action='store_true', help="Force overwrite existing baselines")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Number of CPU Cores to use")
    args = parser.parse_args()
    
    tasks = []
    if os.path.isfile(args.input_dir):
        tasks.append((args.input_dir, args.pt_start, args.pt_end, args.overwrite))
    else:
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith('.h5'):
                    tasks.append((os.path.join(root, file), args.pt_start, args.pt_end, args.overwrite))
    
    if not tasks:
        print("No HDF5 files found to process.")
        return
        
    total_mb_to_process = sum(os.path.getsize(t[0]) for t in tasks) / (1024 * 1024)
    print(f"Calculating Universal Baselines: {len(tasks)} files | {total_mb_to_process:.1f} MB")
    print(f"Pre-Trigger Window: {args.pt_start} us -> {args.pt_end} us")
    
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