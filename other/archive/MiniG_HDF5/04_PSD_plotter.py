"""
04_generate_psd.py
MiniGRAMS LArTPC Analysis Pipeline - Phase II
---------------------------------------------
Generates a single aggregate Power Spectral Density (PSD) plot per Acquisition.
Uses grams_tools.py for FFT/PSD math.
Supports Multiprocessing with RAM-safe chunking for long traces.

Version: 2.3 (RAM-Safe Combo Mode)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import os
import argparse
import time
from multiprocessing import Pool, cpu_count
import detector_config as dc
import grams_tools as gt

# --- CONFIGURATION ---
DEFAULT_EVENTS = 100
BATCH_SIZE = 10  # Safety Cap: Process only 10 events at a time to prevent RAM spikes on Combo data

def detect_config(h5_file, filename):
    """
    Determines plotting layout based on metadata or filename.
    """
    # 1. Attribute Check
    if 'run_config' in h5_file.attrs:
        cfg = h5_file.attrs['run_config']
        if 'combo' in cfg.lower(): return 'combo'
        if 'light' in cfg.lower(): return 'light_only'
    
    # 2. Filename Check
    fname_lower = filename.lower()
    if 'combo' in fname_lower: return 'combo'
    
    return 'light_only'

def process_acquisition_task(payload):
    """
    Worker wrapper to unpack arguments for Pool map.
    """
    acq_path, h5_files, target_events = payload
    
    acq_name = os.path.basename(acq_path)
    print(f"--> [START] {acq_name}", flush=True)
    
    # Sort files to ensure events are selected in chronological order
    h5_files.sort()
    
    # Structure of containers: { ch_id: [psd_trace_1, psd_trace_2, ...] }
    channel_data = {}
    events_collected = 0
    
    # Metadata for plotting
    config_mode = 'light_only' 
    freq_axis_mhz = None
    res_ns = None 
    representative_name = acq_name # Fallback
    
    start_t = time.time()

    # ---  Get PSD's ---
    for filepath in h5_files:
        if events_collected >= target_events:
            break
            
        try:
            with h5py.File(filepath, 'r') as f:
                filename = os.path.basename(filepath)
                
                # Capture metadata from the first valid file we touch
                if events_collected == 0:
                    config_mode = detect_config(f, filename)
                    res_ns = f.attrs.get('resolution_ns', 2.0)
                    # Use filename without extension as the "Long Label"
                    representative_name = os.path.splitext(filename)[0]
                
                # Load Waveforms handle
                dset = f['waveforms_mV']
                n_in_file = dset.shape[0]
                n_samples = dset.shape[2]
                
                # Determine how many to grab from this file
                needed = target_events - events_collected
                grab_total = min(needed, n_in_file)
                
                # Initialize Freq Axis if needed
                if freq_axis_mhz is None:
                    freq_axis = gt.get_freq_axis(n_samples, res_ns)
                    freq_axis_mhz = freq_axis / 1e6
                    
                # Get Map
                ch_map = dc.get_channel_map(config_mode)
                active_channels = []
                for group in ch_map.values():
                    active_channels.extend(group)
                active_channels = sorted(list(set(active_channels)))

                # --- SAFETY CHUNK LOOP ---
                # Process this file in small batches (BATCH_SIZE) to keep RAM flat
                for start_idx in range(0, grab_total, BATCH_SIZE):
                    end_idx = min(start_idx + BATCH_SIZE, grab_total)
                    current_batch_len = end_idx - start_idx
                    
                    # Load ONLY the mini-batch into RAM (Safe for 300us traces)
                    waveforms_batch = dset[start_idx:end_idx]
                    
                    for ch in active_channels:
                        if ch >= waveforms_batch.shape[1]: continue
                        if ch not in channel_data: channel_data[ch] = []
                        
                        # Extract channel traces
                        ch_waves = waveforms_batch[:, ch, :]
                        
                        for i in range(current_batch_len):
                            # Math via grams_tools
                            val = gt.compute_psd_trace(ch_waves[i], res_ns, window_type='hanning')
                            channel_data[ch].append(val)
                    
                    # Cleanup batch from memory immediately
                    del waveforms_batch
                    
                events_collected += grab_total
                
        except Exception as e:
            print(f"    [WARN] {acq_name}: Skipping file {os.path.basename(filepath)}: {e}")
            continue

    if events_collected == 0:
        print(f"    [FAIL] {acq_name}: No events collected.")
        return

    # --- AVERAGING ---
    avg_psd_data = {}
    for ch, traces in channel_data.items():
        if traces:
            stack = np.vstack(traces)
            avg_psd_data[ch] = np.mean(stack, axis=0)

    # --- PLOTTING ---
    save_dir = os.path.join(acq_path, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{acq_name}_PSD.png"
    save_path = os.path.join(save_dir, save_name)

    try:
        # Layout
        ch_map = dc.get_channel_map(config_mode)
        
        if config_mode == 'combo':
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
            layout = [
                (axes[0, 0], 'VUV', ch_map['VUV']),
                (axes[0, 1], 'VIS', ch_map['VIS']),
                (axes[1, 0], 'Charge X', [c for c in ch_map['Charge'] if 24 <= c <= 27]),
                (axes[1, 1], 'Charge Y', [c for c in ch_map['Charge'] if 28 <= c <= 31]),
            ]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
            layout = [
                (axes[0], 'VUV', ch_map['VUV']),
                (axes[1], 'VIS', ch_map['VIS'])
            ]

        cmap = matplotlib.colormaps['tab20']

        for ax, title, channels in layout:
            if not channels:
                ax.text(0.5, 0.5, "No Channels", ha='center', transform=ax.transAxes)
                continue
            
            colors = cmap(np.linspace(0, 1, len(channels))) if len(channels) > 0 else []
            
            for i, ch in enumerate(channels):
                if ch in avg_psd_data:
                    ax.semilogx(freq_axis_mhz, avg_psd_data[ch], 
                                color=colors[i % len(colors)], 
                                alpha=0.6, 
                                lw=1.0, 
                                label=f"Ch {ch}")
            
            ax.set_title(f"Group: {title}", fontweight='bold')
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("PSD (dBm/Hz)")
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(fontsize='small', ncol=2, loc='upper right')
            
            # Info Box (Updated Label)
            info_text = f"Events: {events_collected}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.05, info_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=props)

        # Updated Title Format
        fig.suptitle(f"Event Averaged PSD Analysis\nAcquisition: {representative_name}", fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        print(f"--> [DONE] {acq_name} | Saved: {save_path} ({time.time() - start_t:.2f}s)", flush=True)

    except Exception as e:
        print(f"    [FAIL] Plotting error {acq_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="LArTPC PSD Generator (Per Acquisition)")
    parser.add_argument('input_target', help="Input directory containing Acquisitions")
    parser.add_argument('--events', '-e', type=int, default=DEFAULT_EVENTS, help="Events to average per acq")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    print("=== PSD GENERATOR (ACQUISITION MODE) ===")
    print(f"Target: {args.input_target}")
    print(f"Workers: {args.workers}")
    
    # 1. Group files by Acquisition
    acq_groups = {}
    
    if os.path.isfile(args.input_target):
        # Fallback if user points to a single file
        d = os.path.dirname(args.input_target)
        acq_groups[d] = [args.input_target]
    else:
        for root, _, files in os.walk(args.input_target):
            h5s = [os.path.join(root, f) for f in files if f.endswith('.h5')]
            if h5s:
                acq_groups[root] = h5s

    print(f"Found {len(acq_groups)} Acquisitions.")
    
    # 2. Prepare Tasks
    # List of tuples: (acq_path, file_list, target_events)
    tasks = []
    for acq_path, files in acq_groups.items():
        tasks.append((acq_path, files, args.events))
    
    # 3. Execute with Pool
    if tasks:
        with Pool(args.workers) as p:
            p.map(process_acquisition_task, tasks)
    else:
        print("No HDF5 files found.")
        
    print("=== Complete ===")

if __name__ == "__main__":
    main()