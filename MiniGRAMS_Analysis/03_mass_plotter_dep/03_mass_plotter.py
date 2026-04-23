"""
18_Mass_Plotter_GAr_Rasterized.py
LArTPC Mass PDF Compiler (Golden Events Only)
- Rasterizes dense waveforms to drastically reduce PDF file size.
- Inherits strict formatting from Sanity Checker.
- Automatically applies Baseline Correction.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse
import time
import gc
from multiprocessing import Pool
from tqdm import tqdm
import detector_config as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_event_to_pdf(f, event_idx, mapping, filename, pdf):
    """Generates the 4-panel plot for a single event and saves it to the open PDF."""
    vuv_chans = mapping["VUV"]
    vis_chans = mapping["VIS"]
    x_chans = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"]
    y_chans = mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]

    master_dt = f['Board_72'].attrs['sampling_period_ns']
    master_samples = f['Board_72'].attrs['n_samples']
    master_max_t = (master_samples * master_dt) / 1000.0
    
    pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
        
    def get_trace(global_ch):
        if global_ch < 32:
            grp = f['Board_72']
            local_ch = global_ch
        elif global_ch < 96:
            grp = f['Board_85']   # <--- Board 85 is now Slave 1 (Channels 32-95)
            local_ch = global_ch - 32
        else:
            grp = f['Board_75']   # <--- Board 75 is now Slave 2 (Channels 96-159)
            local_ch = global_ch - 96
        
        dt_ns = grp.attrs['sampling_period_ns']
        trace = grp['waveforms'][event_idx, local_ch, :].copy() 
        
        if 'baseline_mean' in grp:
            bl_val = grp['baseline_mean'][event_idx, local_ch]
            if not np.isnan(bl_val): trace -= bl_val

        t_axis = np.arange(len(trace)) * dt_ns / 1000.0 
        return t_axis, trace

    # --- Top Row: SiPMs ---
    for ch in vuv_chans:
        t, v = get_trace(ch)
        # Added rasterized=True to flatten the lines
        axes[0,0].plot(t, v, alpha=0.5, linewidth=0.8, zorder=2, rasterized=True)
    axes[0,0].set_title(f"VUV SiPMs (N={len(vuv_chans)})", fontweight='bold')
    axes[0,0].set_xlim(0, master_max_t) 
    
    for ch in vis_chans:
        t, v = get_trace(ch)
        axes[0,1].plot(t, v, alpha=0.5, linewidth=0.8, zorder=2, rasterized=True)
    axes[0,1].set_title(f"VIS SiPMs (N={len(vis_chans)})", fontweight='bold')
    axes[0,1].set_xlim(0, master_max_t) 
    
    # Sync SiPM Y-Axes
    vuv_y = axes[0,0].get_ylim()
    vis_y = axes[0,1].get_ylim()
    sipm_min = min(vuv_y[0], vis_y[0])
    sipm_max = max(vuv_y[1], vis_y[1])
    axes[0,0].set_ylim(sipm_min, sipm_max)
    axes[0,1].set_ylim(sipm_min, sipm_max)
    
    # --- Bottom Row: Charge ---
    x_min, x_max = np.inf, -np.inf
    for ch in x_chans:
        t, v = get_trace(ch)
        axes[1,0].plot(t, v, alpha=0.4, linewidth=0.6, zorder=2, rasterized=True)
        if not np.all(np.isnan(v)):
            x_min = min(x_min, np.nanmin(v))
            x_max = max(x_max, np.nanmax(v))
    axes[1,0].set_title(f"Charge X-Direction (Banks A, B, C)", fontweight='bold')
    
    y_min, y_max = np.inf, -np.inf
    for ch in y_chans:
        t, v = get_trace(ch)
        axes[1,1].plot(t, v, alpha=0.4, linewidth=0.6, zorder=2, rasterized=True)
        if not np.all(np.isnan(v)):
            y_min = min(y_min, np.nanmin(v))
            y_max = max(y_max, np.nanmax(v))
    axes[1,1].set_title(f"Charge Y-Direction (Banks D, E, F)", fontweight='bold')

    # Sync and Smart-Limit Charge Y-Axes
    c_min = min(x_min, y_min)
    c_max = max(x_max, y_max)
    axes[1,0].set_ylim(min(-10, c_min - 2), max(20, c_max + 5))
    axes[1,1].set_ylim(min(-10, c_min - 2), max(20, c_max + 5))

    # --- Global Formatting & Trigger Line ---
    for ax in axes.flatten():
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        
        if pre_trig_us > 0:
            ax.axvline(pre_trig_us, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

    fig.suptitle(f"Event: {event_idx} | File: {filename}\n(Baseline Corrected - Golden Event)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save to PDF with DPI capped to keep file size small
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def process_acquisition(task):
    acq_dir, h5_files, limit = task
    acq_name = os.path.basename(acq_dir)
    h5_files.sort()
    
    save_dir = os.path.join(acq_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    pdf_path = os.path.join(save_dir, f"{acq_name}_Golden_MassPlot.pdf")
    mapping = dc.get_channel_map("MINIG")

    events_plotted = 0
    start_time = time.time()
    
    try:
        with PdfPages(pdf_path) as pdf, tqdm(total=limit, desc=acq_name, unit="evt") as pbar:
            for f_path in h5_files:
                if events_plotted >= limit: break
                    
                filename = os.path.basename(f_path)
                try:
                    with h5py.File(f_path, 'r') as f:
                        if 'baseline_mean' not in f['Board_72']:
                            continue
                        
                        n_in_file = f['Board_72']['waveforms'].shape[0]
                        
                        for i in range(n_in_file):
                            if events_plotted >= limit: break
                            
                            # Skip Orphaned or Partial drops entirely
                            if np.isnan(f['Board_75']['waveforms'][i, 0, 0]) or np.isnan(f['Board_85']['waveforms'][i, 0, 0]):
                                continue 
                            
                            plot_event_to_pdf(f, i, mapping, filename, pdf)
                            events_plotted += 1
                            pbar.update(1)
                            
                            # Force garbage collection every 10 events to keep RAM perfectly flat
                            if events_plotted % 10 == 0:
                                gc.collect()

                except Exception as e:
                    print(f"\n[ERR] Reading {filename}: {e}")
                    continue
            
            total_time = time.time() - start_time
            rate = total_time / events_plotted if events_plotted > 0 else 0
            
            print(f"\n[DONE] Saved: {pdf_path} ({events_plotted} Golden Events compiled in {total_time:.1f}s)")

    except Exception as e:
        print(f"\n[FAIL] Generating PDF for {acq_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="LArTPC Multi-Board Mass Plotter (Rasterized Golden Events)")
    parser.add_argument('input_target', help="Input file or directory containing HDF5 files")
    parser.add_argument('--events', '-e', type=int, default=100, help="Max number of GOLDEN events to plot")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Max concurrent processes")
    args = parser.parse_args()
    
    print(f"=== MASS PLOTTER (Golden Event Edition - Rasterized) ===")
    print(f"Target: {args.input_target}")
    print(f"Limit:  {args.events} Golden events max")
    print("-" * 55)
    
    acq_groups = {}
    
    if os.path.isfile(args.input_target):
        d = os.path.dirname(args.input_target)
        acq_groups[d] = [args.input_target]
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5s = [os.path.join(root, f) for f in files if f.endswith('.h5')]
            if h5s: acq_groups[root] = h5s
    
    tasks = [(acq_dir, files, args.events) for acq_dir, files in acq_groups.items()]
    
    if tasks:
        with Pool(min(args.workers, len(tasks))) as p:
            p.map(process_acquisition, tasks)
    else:
        print("No HDF5 files found.")

if __name__ == "__main__":
    main()