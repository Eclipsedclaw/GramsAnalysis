"""
06_Anomaly_Sniper.py
Targeted Waveform Extractor for Anomaly Investigation
- Sweeps the run directory to find the exact HDF5 file containing a target minute.
- Extracts and plots raw waveforms (VIS Left, VUV Right) to visually identify noise vs physics.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import glob
import argparse
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_trace(grp, local_ch, event_idx):
    dt_ns = grp.attrs.get('sampling_period_ns', 2.0)
    trace = grp['waveforms'][event_idx, local_ch, :].copy()
    
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
        trace -= grp['baseline_mean'][event_idx, local_ch]
        
    t_us = np.arange(len(trace)) * dt_ns / 1000.0
    return t_us, trace

def main():
    parser = argparse.ArgumentParser(description="Anomaly Sniper Plotter")
    parser.add_argument('input_dir', help="Directory containing baseline-corrected HDF5 files")
    parser.add_argument('--target_minute', '-t', type=float, required=True, help="The exact minute to investigate")
    parser.add_argument('--events', '-e', type=int, default=50, help="Number of events to plot")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")), key=natural_sort_key)
    h5_files = [f for f in h5_files if "SPE_Master_Integrals" not in f]
    
    if not h5_files:
        print(f"[FAIL] No HDF5 files found.")
        return

    clock_hz = 125000000.0 
    
    print(f"Hunting for the file containing Minute {args.target_minute}...")
    
    # Establish T0
    with h5py.File(h5_files[0], 'r') as f:
        t0 = f['Board_72']['timestamps'][0]

    target_file = None
    
    # Locate the target file
    for filepath in h5_files:
        try:
            with h5py.File(filepath, 'r') as f:
                timestamps = f['Board_72']['timestamps']
                t_start_min = (timestamps[0] - t0) / clock_hz / 60.0
                t_end_min = (timestamps[-1] - t0) / clock_hz / 60.0
                
                if t_start_min <= args.target_minute <= t_end_min:
                    target_file = filepath
                    break
        except Exception:
            continue
            
    if not target_file:
        print(f"[FAIL] Could not find any file containing minute {args.target_minute}.")
        return
        
    print(f"[LOCKED] Target File: {os.path.basename(target_file)}")
    
    out_dir = os.path.join(args.input_dir, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"Anomaly_Investigation_Min{args.target_minute}.pdf")

    # Fast CAEN Even/Odd logic
    vuv_chans = [i for i in range(32) if i % 2 == 0]
    vis_chans = [i for i in range(32) if i % 2 != 0]

    print("Generating Meat Suit Inspection PDF...")
    try:
        with h5py.File(target_file, 'r') as f, PdfPages(out_pdf) as pdf:
            grp = f['Board_72']
            n_events = grp['waveforms'].shape[0]
            limit = min(n_events, args.events)
            
            for i in tqdm(range(limit), desc="Rendering Events"):
                fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
                y_min, y_max = np.inf, -np.inf
                
                # VIS on LEFT (Odds)
                for ch in vis_chans:
                    t, v = get_trace(grp, ch, i)
                    if not np.all(np.isnan(v)):
                        axes[0].plot(t, v, alpha=0.7, linewidth=1.0)
                        y_min, y_max = min(y_min, np.nanmin(v)), max(y_max, np.nanmax(v))
                        
                # VUV on RIGHT (Evens)
                for ch in vuv_chans:
                    t, v = get_trace(grp, ch, i)
                    if not np.all(np.isnan(v)):
                        axes[1].plot(t, v, alpha=0.7, linewidth=1.0)
                        y_min, y_max = min(y_min, np.nanmin(v)), max(y_max, np.nanmax(v))

                axes[0].set_title(f"VIS SiPMs (Odds)", fontweight='bold')
                axes[1].set_title(f"VUV SiPMs (Evens)", fontweight='bold')
                
                for ax in axes:
                    ax.set_xlabel("Time (µs)", fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    if len(t) > 0: ax.set_xlim(0, t[-1])
                
                buffer = (y_max - y_min) * 0.1 if y_max != -np.inf else 10.0
                if y_min != np.inf and y_max != -np.inf:
                    axes[0].set_ylim(y_min - buffer, y_max + buffer)
                    
                axes[0].set_ylabel("Amplitude (mV)", fontweight='bold')
                bl_status = "Baseline Corrected" if 'baseline_mean' in grp else "Raw Data"
                fig.suptitle(f"Target Minute: {args.target_minute} | Event: {i}\nFile: {os.path.basename(target_file)} ({bl_status})", fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
                
        print(f"\n[SUCCESS] Anomaly Waveforms saved to: {out_pdf}")

    except Exception as e:
        print(f"\n[FAIL] Plotting error: {e}")

if __name__ == "__main__":
    main()