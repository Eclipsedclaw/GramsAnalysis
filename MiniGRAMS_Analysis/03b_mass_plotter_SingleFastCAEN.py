"""
07_Light_Only_Plotter_1x2.py
Quarantined 1x2 Light Mass Plotter (Master CAEN Only)
- Plots VIS (Odds) on LEFT, VUV (Evens) on RIGHT for Board 72.
- Handles both Raw and Baseline Corrected data dynamically.
- Generates a multi-page PDF for rapid visual inspection.
- Updated: Routes output PDF cleanly into a local 'Plots' subdirectory.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_trace(grp, local_ch, event_idx):
    dt_ns = grp.attrs.get('sampling_period_ns', 8.0)
    trace = grp['waveforms'][event_idx, local_ch, :].copy()
    
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
        trace -= grp['baseline_mean'][event_idx, local_ch]
        
    t_us = np.arange(len(trace)) * dt_ns / 1000.0
    return t_us, trace

def main():
    parser = argparse.ArgumentParser(description="1x2 Light Only Mass Plotter")
    parser.add_argument('input_file', help="HDF5 file to plot")
    parser.add_argument('--events', '-e', type=int, default=100, help="Number of events to plot into the PDF")
    args = parser.parse_args()

    # Dynamic path routing for the 'Plots' subdirectory
    input_abs_path = os.path.abspath(args.input_file)
    base_dir = os.path.dirname(input_abs_path)
    base_name = os.path.basename(input_abs_path)
    
    plot_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    out_pdf = os.path.join(plot_dir, base_name.replace('.h5', '_Light1x2_Inspection.pdf'))

    try:
        with h5py.File(args.input_file, 'r') as f:
            if 'Board_72' not in f:
                print("[FAIL] Board_72 not found in file. Is this single-board data?")
                return
                
            grp = f['Board_72']
            n_events = grp['waveforms'].shape[0]
            limit = min(n_events, args.events)
            
            # Fast CAEN Even/Odd logic
            vuv_chans = [i for i in range(32) if i % 2 == 0]
            vis_chans = [i for i in range(32) if i % 2 != 0]

            with PdfPages(out_pdf) as pdf:
                for i in tqdm(range(limit), desc="Rendering PDF Pages"):
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
                    
                    y_min, y_max = np.inf, -np.inf
                    
                    # Plot VIS (Odds) on LEFT
                    for ch in vis_chans:
                        t, v = get_trace(grp, ch, i)
                        if not np.all(np.isnan(v)):
                            axes[0].plot(t, v, alpha=0.7, linewidth=1.0, label=f"Ch {ch}")
                            y_min = min(y_min, np.nanmin(v))
                            y_max = max(y_max, np.nanmax(v))
                            
                    # Plot VUV (Evens) on RIGHT
                    for ch in vuv_chans:
                        t, v = get_trace(grp, ch, i)
                        if not np.all(np.isnan(v)):
                            axes[1].plot(t, v, alpha=0.7, linewidth=1.0, label=f"Ch {ch}")
                            y_min = min(y_min, np.nanmin(v))
                            y_max = max(y_max, np.nanmax(v))

                    axes[0].set_title(f"VIS SiPMs (Odds)", fontweight='bold')
                    axes[1].set_title(f"VUV SiPMs (Evens)", fontweight='bold')
                    
                    for ax in axes:
                        ax.set_xlabel("Time (µs)", fontweight='bold')
                        ax.grid(True, linestyle='--', alpha=0.5)
                        if len(t) > 0:
                            ax.set_xlim(0, t[-1])
                    
                    # Dynamic Y-Axis scaling
                    buffer = (y_max - y_min) * 0.1 if y_max != -np.inf else 10.0
                    if y_min != np.inf and y_max != -np.inf:
                        axes[0].set_ylim(y_min - buffer, y_max + buffer)
                        
                    axes[0].set_ylabel("Amplitude (mV)", fontweight='bold')
                    
                    bl_status = "Baseline Corrected" if 'baseline_mean' in grp else "Raw Data"
                    fig.suptitle(f"Event: {i} | File: {base_name}\n({bl_status})", fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=150)
                    plt.close(fig)
                    
            print(f"\n[SUCCESS] Inspection PDF saved to: {out_pdf}")

    except Exception as e:
        print(f"\n[FAIL] Error plotting: {e}")

if __name__ == "__main__":
    main()