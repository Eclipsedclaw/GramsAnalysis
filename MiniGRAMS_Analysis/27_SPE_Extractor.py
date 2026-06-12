"""
03_SPE_SafeHarbor_Extractor.py
Mass Time-Binned SPE Safe Harbor Extractor (MULTIPROCESSING ENABLED)
- Processes an entire directory of baseline-corrected HDF5 files using parallel workers.
- Explicitly decouples 500 MHz trace sampling from 125 MHz CAEN FPGA timestamps.
- Prints an exact, 1-second precision timing diagnostic report before processing.
- Integrates the 0.5 µs to 3.0 µs safe harbor for pure DCR extraction.
- Updated: Routes the final '_SPE_Master_Integrals.h5' file into a dedicated 'SPE_data' subdirectory.
"""

import h5py
import numpy as np
import os
import glob
import argparse
import re
import time
from multiprocessing import Pool
from tqdm import tqdm

def natural_sort_key(s):
    """Ensures part2 comes before part10, avoiding lexicographical garbling."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def worker_extract(task):
    """Parallel worker function to process a single HDF5 file."""
    filepath, t0, clock_hz, bin_width_secs, num_bins, idx_start, idx_end, dt_ns = task
    
    # Initialize a local dictionary for this specific file
    local_bin_integrals = {b: {ch: [] for ch in range(32)} for b in range(num_bins)}
    events_processed = 0
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'Board_72' not in f:
                return (False, local_bin_integrals, 0, f"No Board_72 in {os.path.basename(filepath)}")
                
            grp = f['Board_72']
            if 'baseline_mean' not in grp:
                return (False, local_bin_integrals, 0, f"No baselines in {os.path.basename(filepath)}")
            
            waveforms = grp['waveforms']
            baselines = grp['baseline_mean']
            timestamps = grp['timestamps'][:]
            
            n_events, n_channels, n_samples = waveforms.shape
            events_processed = n_events
            
            # Convert absolute timestamps to relative seconds
            relative_secs = (timestamps - t0) / clock_hz
            
            # Map each event to a chronological bin
            bin_indices = np.floor(relative_secs / bin_width_secs).astype(int)
            np.clip(bin_indices, 0, num_bins - 1, out=bin_indices)
            
            chunk_size = 5000 
            for start_idx in range(0, n_events, chunk_size):
                end_idx = min(start_idx + chunk_size, n_events)
                
                wave_chunk = waveforms[start_idx:end_idx]
                bl_chunk = baselines[start_idx:end_idx]
                bin_chunk = bin_indices[start_idx:end_idx]
                
                for i in range(len(wave_chunk)):
                    current_bin = bin_chunk[i]
                    for ch in range(n_channels):
                        if ch >= 32: continue # Safety clamp
                        if np.isnan(bl_chunk[i, ch]): continue
                        
                        trace_slice = wave_chunk[i, ch, idx_start:idx_end]
                        # Discrete Riemann Sum
                        integral_mv_ns = np.sum(trace_slice - bl_chunk[i, ch]) * dt_ns
                        local_bin_integrals[current_bin][ch].append(integral_mv_ns)
                        
        return (True, local_bin_integrals, events_processed, "")
        
    except Exception as e:
        return (False, local_bin_integrals, 0, str(e))

def main():
    parser = argparse.ArgumentParser(description="Mass Time-Binned SPE Extractor (Parallel)")
    parser.add_argument('input_dir', help="Directory containing baseline-corrected HDF5 files")
    parser.add_argument('--start_us', type=float, default=0.5, help="Safe harbor start (µs)")
    parser.add_argument('--end_us', type=float, default=3.0, help="Safe harbor end (µs)")
    parser.add_argument('--num_bins', type=int, default=6, help="Number of time bins to split the run into")
    parser.add_argument('--workers', '-w', type=int, default=6, help="Number of CPU workers")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")), key=natural_sort_key)
    if not h5_files:
        print(f"[FAIL] No HDF5 files found in {args.input_dir}")
        return

    # Don't grab the master file if it already exists from a previous run
    h5_files = [f for f in h5_files if "SPE_Master_Integrals" not in f]
    
    # Subdirectory Routing Logic
    spe_out_dir = os.path.join(args.input_dir, "SPE_data")
    os.makedirs(spe_out_dir, exist_ok=True)
    
    out_file = os.path.join(spe_out_dir, f"{os.path.basename(os.path.normpath(args.input_dir))}_SPE_Master_Integrals.h5")
    
    try:
        # --- STEP 1: Global Time Synchronization (T0 and T_final) ---
        print("\nScanning timestamps to establish global run boundaries...")
        
        with h5py.File(h5_files[0], 'r') as f:
            t0 = f['Board_72']['timestamps'][0]
            dt_ns = f['Board_72'].attrs.get('sampling_period_ns', 2.0)
            n_channels = f['Board_72']['waveforms'].shape[1]
            det_config = f['Board_72'].attrs.get('detector_config', [])

        with h5py.File(h5_files[-1], 'r') as f:
            t_final = f['Board_72']['timestamps'][-1]

        clock_hz = 125000000.0 
        total_seconds = (t_final - t0) / clock_hz
        bin_width_secs = total_seconds / args.num_bins

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        print("\n" + "="*50)
        print("          TIMING DIAGNOSTIC REPORT")
        print("="*50)
        print(f"Start Time (T0):      {t0} ticks")
        print(f"End Time (T_final):   {t_final} ticks")
        print(f"Total Ticks:          {t_final - t0}")
        print(f"FPGA Hardware Clock:  {clock_hz/1e6:.1f} MHz (8 ns ticks)")
        print(f"Trace Sampling Rate:  {1000.0/dt_ns:.1f} MHz ({dt_ns} ns steps)")
        print("-" * 50)
        print(f"Exact Total Time:     {hours} hours, {minutes} minutes, {seconds} seconds")
        print(f"                      ({total_seconds:.2f} total seconds)")
        print("="*50 + "\n")

        idx_start = int((args.start_us * 1000.0) / dt_ns)
        idx_end = int((args.end_us * 1000.0) / dt_ns)

        # Master RAM Dictionary
        master_bin_integrals = {b: {ch: [] for ch in range(32)} for b in range(args.num_bins)}

        # --- STEP 2: Parallel Extraction ---
        tasks = [(f, t0, clock_hz, bin_width_secs, args.num_bins, idx_start, idx_end, dt_ns) for f in h5_files]
        total_events_processed = 0
        
        print(f"Deploying {args.workers} workers to integrate Safe Harbor...")
        with Pool(args.workers) as pool:
            # Use imap_unordered for fastest execution, tqdm handles the progress bar per file
            for success, local_data, ev_count, err_msg in tqdm(pool.imap_unordered(worker_extract, tasks), total=len(tasks), desc="Processing Files"):
                if success:
                    total_events_processed += ev_count
                    # Merge local worker data into the master dictionary
                    for b in range(args.num_bins):
                        for ch in range(32):
                            if local_data[b][ch]:
                                master_bin_integrals[b][ch].extend(local_data[b][ch])
                else:
                    print(f"\n[Warning] Worker Failed: {err_msg}")

        # --- STEP 3: Write Master HDF5 ---
        print(f"\nWriting Master Integrals to: {out_file}")
        with h5py.File(out_file, 'w') as f_out:
            f_out.attrs['safe_harbor_start_us'] = args.start_us
            f_out.attrs['safe_harbor_end_us'] = args.end_us
            f_out.attrs['num_bins'] = args.num_bins
            f_out.attrs['bin_width_seconds'] = bin_width_secs
            f_out.attrs['detector_config'] = det_config
            
            for b in range(args.num_bins):
                bin_grp = f_out.create_group(f"TimeBin_{b}")
                bin_grp.attrs['start_sec'] = b * bin_width_secs
                bin_grp.attrs['end_sec'] = (b + 1) * bin_width_secs
                
                for ch in range(32):
                    data = np.array(master_bin_integrals[b][ch], dtype=np.float32)
                    if len(data) > 0:
                        bin_grp.create_dataset(f"Ch_{ch}", data=data, compression='lzf')

        # --- STEP 4: YIELD SUMMARY REPORT ---
        print("\n" + "="*50)
        print("          EXTRACTION SUMMARY REPORT")
        print("="*50)
        print(f"Target Acq:     {os.path.basename(os.path.normpath(args.input_dir))}")
        print(f"Bin Width:      {bin_width_secs / 60.0:.2f} minutes")
        print(f"Output Dir:     {spe_out_dir}")
        print("-" * 50)
        
        for b in range(args.num_bins):
            # Using Channel 0 to count events (all channels have the same event count per bin)
            n_events_in_bin = len(master_bin_integrals[b][0]) 
            print(f"Time Bin {b}:    {n_events_in_bin} events extracted")
            
        print("-" * 50)
        print(f"Total Processed: {total_events_processed} events")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Extraction error: {e}")

if __name__ == "__main__":
    main()