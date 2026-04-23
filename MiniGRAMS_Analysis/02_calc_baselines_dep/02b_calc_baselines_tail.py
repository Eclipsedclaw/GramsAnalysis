#!/usr/bin/env python3
"""
02b_calc_baselines_tail.py (QUARANTINED SCRIPT)
Calculates baselines by anchoring to the END of the waveform.
Specifically designed to salvage data taken with an incorrect Rising-Edge Trigger.
"""

import h5py
import numpy as np
import os
import argparse
import time
import warnings
from multiprocessing import Pool
from tqdm import tqdm

def process_tail_baseline(task):
    filepath, tail_window_us = task
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'waveforms_mV' not in f: return 0, 0 # Only applies to MicroG format
                
            waves = f['waveforms_mV']
            n_events, n_channels, n_samples = waves.shape
            res_ns = f.attrs.get('resolution_ns', 8.0)
            
            # Calculate how many samples make up the tail window
            tail_samples = int((tail_window_us * 1000.0) / res_ns)
            if tail_samples >= n_samples or tail_samples <= 0:
                tail_samples = int(n_samples * 0.2) # Fallback to last 20% if user asks for too much
                
            # Slice the end of the trace
            tail_slice = waves[:, :, -tail_samples:]
            
            # Safely calculate mean and std, ignoring the NaN channels
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                base_mean = np.nanmean(tail_slice, axis=2)
                base_std = np.nanstd(tail_slice, axis=2)
                
            # Overwrite existing baseline datasets if they exist
            if 'baseline_mean_mV' in f: del f['baseline_mean_mV']
            if 'baseline_std_mV' in f: del f['baseline_std_mV']
            
            f.create_dataset('baseline_mean_mV', data=base_mean)
            f.create_dataset('baseline_std_mV', data=base_std)
            f.attrs['baseline_tail_window_us'] = tail_window_us
            
            return n_events, os.path.getsize(filepath) / (1024*1024)
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description="Tail-Anchor Baseline Calculator")
    parser.add_argument('input_dir', help="Directory with HDF5 files")
    parser.add_argument('--tail_window', '-t', type=float, default=3.0, help="Microseconds at the END of the trace to average")
    parser.add_argument('--workers', '-w', type=int, default=4)
    args = parser.parse_args()
    
    tasks = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.h5'):
                tasks.append((os.path.join(root, file), args.tail_window))
                
    if not tasks:
        print("No files found.")
        return
        
    total_mb = sum(os.path.getsize(t[0]) for t in tasks) / (1024 * 1024)
    print(f"Calculating Tail Baselines: {len(tasks)} files | Window: Last {args.tail_window} us")
    
    start_time = time.time()
    total_events = 0
    
    with Pool(min(args.workers, 8)) as p:
        iterator = p.imap_unordered(process_tail_baseline, tasks)
        with tqdm(total=total_mb, unit='MB', desc="Processing") as pbar:
            for count, file_mb in iterator:
                total_events += count
                pbar.update(file_mb)
                
    print(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()