"""
03_mass_plotter_univ.py
LArTPC Mass PDF Compiler (Universal Edition)
- Supports Single-Board (MicroGRAMS) and Multi-Board (MiniGRAMS) HDF5s.
- Preserves high-density 4-panel aesthetic (SiPMs top, Charge bottom).
- Auto-detects Raw vs Baselined data.
- Independent X-axis limit controls for Light and Charge panels.
- Updated: Substring config checking to support MINIG_RUN9_BANDAID DB50 swaps.
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

def get_label(ch_idx, run_config):
    """Bridge between array index, CAEN channel, and physical signal name."""
    if run_config == 'MICROG_PURITY_STUDY':
        bridge = {
            30: "CAEN 30 (VUV_1)", 31: "CAEN 31 (VUV_2)",
            32: "CAEN 32 (VIS_1)", 33: "CAEN 33 (VIS_2)"
        }
        return bridge.get(ch_idx, f"Ch {ch_idx}")
        
    elif 'MINIG' in run_config:
        return f"Ch {ch_idx}"
        
    return f"Ch {ch_idx}"

def get_trace_universal(f, global_ch, event_idx, is_single_board):
    """Fetches a trace dynamically. ONLY applies baseline if explicitly found in HDF5."""
    if is_single_board:
        trace = f['waveforms_mV'][event_idx, global_ch, :].copy()
        dt_ns = f.attrs.get('resolution_ns', 8.0)
        
        if 'baseline_mean_mV' in f:
            bl_val = f['baseline_mean_mV'][event_idx, global_ch]
            if not np.isnan(bl_val): trace -= bl_val
            
    else:
        if global_ch < 32:
            grp = f['Board_72']; local_ch = global_ch
        elif global_ch < 96:
            grp = f['Board_85']; local_ch = global_ch - 32
        else:
            grp = f['Board_75']; local_ch = global_ch - 96
            
        dt_ns = grp.attrs['sampling_period_ns']
        
        if local_ch >= grp['waveforms'].shape[1]:
            return np.array([]), np.full(grp['waveforms'].shape[2], np.nan)
            
        trace = grp['waveforms'][event_idx, local_ch, :].copy()
        
        if 'baseline_mean' in grp and local_ch < grp['baseline_mean'].shape[1]:
            bl_val = grp['baseline_mean'][event_idx, local_ch]
            if not np.isnan(bl_val): trace -= bl_val

    t_axis = np.arange(len(trace)) * dt_ns / 1000.0 
    return t_axis, trace

def plot_event_to_pdf(f, event_idx, mapping, filename, pdf, is_single_board, run_config, xlim_light=None, xlim_charge=None):
    """Generates the 4-panel plot in the desired physical layout."""
    
    vuv_chans = mapping.get("VUV", [])
    vis_chans = mapping.get("VIS", [])
    
    # Substring check ensures MINIG_RUN9_BANDAID falls into this correct geometry
    if 'MINIG' in run_config:
        x_chans = mapping.get("Charge_A", []) + mapping.get("Charge_B", []) + mapping.get("Charge_C", [])
        y_chans = mapping.get("Charge_D", []) + mapping.get("Charge_E", []) + mapping.get("Charge_F", [])
        x_title = f"Charge X-Direction (Banks A, B, C)"
        y_title = f"Charge Y-Direction (Banks D, E, F)"
    else:
        x_chans = mapping.get("Charge_C", []) or mapping.get("Charge_A", []) or mapping.get("Charge", [])[:3]
        y_chans = mapping.get("Charge_D", []) or mapping.get("Charge_B", []) or mapping.get("Charge", [])[3:]
        x_title = f"Charge X-Direction (N={len(x_chans)})"
        y_title = f"Charge Y-Direction (N={len(y_chans)})"

    if is_single_board:
        master_dt = f.attrs.get('resolution_ns', 8.0)
        master_samples = f['waveforms_mV'].shape[2]
        pre_trig_us = f.attrs.get('baseline_pre_trigger_start_us', 0.0) # Fallback if end isn't available
        # Prefer plotting the end of the window if available
        trigger_line_us = f.attrs.get('baseline_pre_trigger_end_us', f.attrs.get('baseline_pre_trigger_us', 4.0)) 
    else:
        master_dt = f['Board_72'].attrs['sampling_period_ns']
        master_samples = f['Board_72'].attrs['n_samples']
        trigger_line_us = f['Board_72'].attrs.get('baseline_pre_trigger_end_us', f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0))

    master_max_t = (master_samples * master_dt) / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    def plot_group(ax, chans, title, is_sipm=False):
        c_min, c_max = np.inf, -np.inf
        plotted_any = False
        valid_chans_plotted = 0
        
        for ch in chans:
            t, v = get_trace_universal(f, ch, event_idx, is_single_board)
            if len(t) == 0 or np.all(np.isnan(v)): continue 
            
            plotted_any = True
            valid_chans_plotted += 1
            label_str = get_label(ch, run_config)
            
            plot_alpha = 0.5 if not is_sipm else 0.7
            
            ax.plot(t, v, alpha=plot_alpha, linewidth=1.0, label=label_str, rasterized=True)
            c_min = min(c_min, np.nanmin(v))
            c_max = max(c_max, np.nanmax(v))
        
        final_title = f"{title}" if not is_sipm else f"{title} (N={valid_chans_plotted})"
        ax.set_title(final_title, fontweight='bold')
            
        return c_min, c_max

    # Top Row: SiPMs
    plot_group(axes[0,0], vuv_chans, "VUV SiPMs", is_sipm=True)
    plot_group(axes[0,1], vis_chans, "VIS SiPMs", is_sipm=True)
    
    v_min = min(axes[0,0].get_ylim()[0], axes[0,1].get_ylim()[0])
    v_max = max(axes[0,0].get_ylim()[1], axes[0,1].get_ylim()[1])
    axes[0,0].set_ylim(v_min - 5, v_max + 10)
    axes[0,1].set_ylim(v_min - 5, v_max + 10)

    # Bottom Row: Charge
    x_min, x_max = plot_group(axes[1,0], x_chans, x_title)
    y_min, y_max = plot_group(axes[1,1], y_chans, y_title)
    
    valid_x_min = x_min if x_min != np.inf else -30
    valid_y_min = y_min if y_min != np.inf else -30
    valid_x_max = x_max if x_max != -np.inf else 30
    valid_y_max = y_max if y_max != -np.inf else 30
    
    valid_c_min = min(valid_x_min, valid_y_min)
    valid_c_max = max(valid_x_max, valid_y_max)
    axes[1,0].set_ylim(valid_c_min - 5, valid_c_max + 5)
    axes[1,1].set_ylim(valid_c_min - 5, valid_c_max + 5)

    # Global Axis Formatting
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlabel(r"Time ($\mu$s)")
        if ax in [axes[0,0], axes[1,0]]: ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Plot the pre-trigger window end-line
        if trigger_line_us > 0:
            ax.axvline(trigger_line_us, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            
        is_light_panel = idx < 2
        if is_light_panel and xlim_light is not None:
            ax.set_xlim(0, xlim_light)
        elif not is_light_panel and xlim_charge is not None:
            ax.set_xlim(0, xlim_charge)
        else:
            ax.set_xlim(0, master_max_t)

    bl_status = "Raw Data" if 'baseline_mean_mV' not in f and 'Board_72' not in f or ('Board_72' in f and 'baseline_mean' not in f['Board_72']) else "Baseline Corrected"
    fig.suptitle(f"Event: {event_idx} | File: {filename}\n({bl_status})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def process_acquisition(task):
    acq_dir, h5_files, limit, xl, xc = task
    acq_name = os.path.basename(acq_dir)
    h5_files.sort()
    
    save_dir = os.path.join(acq_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"{acq_name}_MassPlot.pdf")

    events_plotted = 0
    start_time = time.time()
    
    try:
        with PdfPages(pdf_path) as pdf, tqdm(total=limit, desc=acq_name, unit="evt") as pbar:
            for f_path in h5_files:
                if events_plotted >= limit: break
                filename = os.path.basename(f_path)
                
                try:
                    with h5py.File(f_path, 'r') as f:
                        run_config = f.attrs.get('run_config', 'MINIG')
                        mapping = dc.get_channel_map(run_config)
                        is_single_board = 'waveforms_mV' in f
                        
                        n_in_file = f['waveforms_mV'].shape[0] if is_single_board else f['Board_72']['waveforms'].shape[0]
                        
                        for i in range(n_in_file):
                            if events_plotted >= limit: break
                            if not is_single_board:
                                if np.isnan(f['Board_75']['waveforms'][i, 0, 0]) or np.isnan(f['Board_85']['waveforms'][i, 0, 0]):
                                    continue 
                            
                            plot_event_to_pdf(f, i, mapping, filename, pdf, is_single_board, run_config, xl, xc)
                            events_plotted += 1
                            pbar.update(1)
                            if events_plotted % 10 == 0: gc.collect()

                except OSError as oe: continue
                except Exception as e: 
                    continue
            
            total_time = time.time() - start_time
            print(f"\n[DONE] Saved: {pdf_path} ({events_plotted} Events in {total_time:.1f}s)")

    except Exception as e:
        print(f"\n[FAIL] Generating PDF for {acq_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="LArTPC Universal Mass Plotter")
    parser.add_argument('input_target', help="Input file or directory containing HDF5 files")
    parser.add_argument('--events', '-e', type=int, default=100, help="Max number of events to plot")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Max concurrent processes")
    parser.add_argument('--xlim_light', '-xl', type=float, default=None, help="Custom X-axis limit for SiPMs (us)")
    parser.add_argument('--xlim_charge', '-xc', type=float, default=None, help="Custom X-axis limit for Charge (us)")
    args = parser.parse_args()
    
    acq_groups = {}
    if os.path.isfile(args.input_target):
        d = os.path.dirname(args.input_target)
        acq_groups[d] = [args.input_target]
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5s = [os.path.join(root, f) for f in files if f.endswith('.h5')]
            if h5s: acq_groups[root] = h5s
    
    tasks = [(acq_dir, files, args.events, args.xlim_light, args.xlim_charge) for acq_dir, files in acq_groups.items()]
    
    if tasks:
        with Pool(min(args.workers, len(tasks))) as p:
            p.map(process_acquisition, tasks)

if __name__ == "__main__":
    main()
    
    