import h5py
import numpy as np
import os
import argparse
import time
import gc
from multiprocessing import Pool, cpu_count
import grams_tools as gt 

def process_file(task):
    filepath, pre_trigger_us, overwrite = task
    filename = os.path.basename(filepath)
    events_processed = 0
    
    try:
        with h5py.File(filepath, 'r+') as h5_file:
            # 1. Check if processing is needed
            if 'baseline_mean_mV' in h5_file and not overwrite:
                return (filename, 0)

            # 2. Get Dimensions & Sanity Check
            dset_waveforms = h5_file['waveforms_mV']
            n_events, n_ch, n_samples = dset_waveforms.shape
            
            # --- Dynamic Chunking ---
            # Target RAM per chunk: 200 MB
            # Formula: events = target_bytes / (n_ch * n_samples * 4 bytes)
            bytes_per_event = n_ch * n_samples * 4
            target_chunk_bytes = 200 * 1024 * 1024 # 200 MB
            
            if bytes_per_event > target_chunk_bytes:
                # If a SINGLE event is > 200MB, we have a "Poison File" or massive corruption.
                print(f"[WARN] {filename}: massive event size ({bytes_per_event/1024/1024:.1f} MB/evt). Processing 1 by 1.")
                chunk_size = 1
            else:
                chunk_size = int(target_chunk_bytes / bytes_per_event)
                chunk_size = min(chunk_size, 1000)
            
            # Sanity Check for zero chunk size
            if chunk_size < 1: chunk_size = 1

            # Setup Pre-trigger
            res_ns = h5_file.attrs.get('resolution_ns', 2)
            n_pre_samples = gt.get_pre_trigger_samples(res_ns, safe_window_us=pre_trigger_us)
            if n_pre_samples > n_samples: n_pre_samples = n_samples

            # --- READ CONFIG ---
            if 'detector_config' in h5_file.attrs:
                det_config = [x.decode('utf-8') for x in h5_file.attrs['detector_config']]
            else:
                det_config = ["SiPM"] * n_ch

            # 3. Initialize Output Datasets
            if 'baseline_mean_mV' in h5_file: del h5_file['baseline_mean_mV']
            if 'baseline_std_mV' in h5_file: del h5_file['baseline_std_mV']
            
            dset_bl = h5_file.create_dataset('baseline_mean_mV', shape=(n_events, n_ch), dtype=np.float32)
            dset_std = h5_file.create_dataset('baseline_std_mV', shape=(n_events, n_ch), dtype=np.float32)
            
            # 4. CHUNKED PROCESSING LOOP
            for start_idx in range(0, n_events, chunk_size):
                end_idx = min(start_idx + chunk_size, n_events)
                
                # Load Chunk
                wave_chunk = dset_waveforms[start_idx:end_idx]
                
                # Alloc Results
                bl_chunk = np.zeros((end_idx - start_idx, n_ch), dtype=np.float32)
                std_chunk = np.zeros((end_idx - start_idx, n_ch), dtype=np.float32)
                
                # Process
                for i in range(len(wave_chunk)): 
                    for j in range(n_ch):
                        segment = wave_chunk[i, j, :n_pre_samples]
                        ch_type = det_config[j]
                        
                        if ch_type == "Charge":
                            bl = gt.baseline_charge(segment, fine_tune_window=2.0)
                            std = np.std(segment)
                        else:
                            bl = gt.baseline_iterative_clipping_RT(segment, sigma=2.5)
                            std = np.std(segment)
                        
                        bl_chunk[i, j] = bl
                        std_chunk[i, j] = std
                
                # Write Back
                dset_bl[start_idx:end_idx] = bl_chunk
                dset_std[start_idx:end_idx] = std_chunk
                
                # CLEANUP: Force release memory
                del wave_chunk
                del bl_chunk
                del std_chunk
                
            h5_file.attrs['baseline_pre_trigger_us'] = pre_trigger_us
            events_processed = n_events
            
            # Manual garbage collection to be safe
            gc.collect()

    except Exception as e:
        print(f"[FAIL] {filename}: {e}")
        return (filename, 0)
        
    return (filename, events_processed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--pre_trigger', '-pt', type=float, required=True)
    parser.add_argument('--overwrite', '-f', action='store_true')
    parser.add_argument('--workers', '-w', type=int, default=4)
    args = parser.parse_args()
    
    tasks = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.h5'):
                tasks.append((os.path.join(root, file), args.pre_trigger, args.overwrite))
    
    print(f"Calculating Baselines for {len(tasks)} files...")
    
    # Print first file dimensions to verify
    if len(tasks) > 0:
        try:
            with h5py.File(tasks[0][0], 'r') as f:
                shape = f['waveforms_mV'].shape
                print(f"[INFO] Sample Check - {os.path.basename(tasks[0][0])} Shape: {shape}")
                size_mb = (shape[1] * shape[2] * 4) / 1024 / 1024
                print(f"[INFO] Size per event: {size_mb:.2f} MB")
        except:
            pass

    start_time = time.time()
    
    total_events = 0
    if tasks:
        # 8 core limit for server safety
        active_workers = min(args.workers, 8)
        print(f"Active Workers: {active_workers}")
        with Pool(active_workers) as p:
            results = p.map(process_file, tasks)
            for fname, count in results:
                total_events += count
                
    elapsed = time.time() - start_time
    avg_per_evt = elapsed / total_events if total_events > 0 else 0.0
    
    print("-" * 40)
    print(f"Baseline Calculation Complete.")
    print(f"Total Time:   {elapsed:.2f} s")
    print(f"Total Events: {total_events}")
    print(f"Performance:  {avg_per_evt*1000:.2f} ms/event")
    print("-" * 40)

if __name__ == "__main__":
    main()