"""
05_Charge_Averaged_Plotter_6x15.py
LArTPC 6-Bank Averaged Waveform Plotter (Pure Voltage)
- Automatically sniffs MINIG_RUN9_BANDAID configuration.
- Averages N events per channel to produce a single, ultra-clean trace.
- Explicit 6x15 physical geometry mapping with N/C handling.
- Upgraded: Arrays reversed (SIG14 -> SIG0) to perfectly map the physical L-shape of the TPC.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse
from tqdm import tqdm
import detector_config as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def detect_config(h5_file, filepath):
    """Path sniffing logic for BANDAID config override."""
    path_config = dc.detect_run_mode(filepath)
    if path_config != 'LIGHT_ONLY' and path_config != 'MINIG':
        return path_config
    if 'run_config' in h5_file.attrs:
        return h5_file.attrs['run_config']
    return 'MINIG' 

def get_trace(f, event_idx, global_ch):
    """Fetches and baseline-corrects a single trace."""
    if global_ch < 32: 
        grp = f['Board_72']; local_ch = global_ch
    elif global_ch < 96: 
        grp = f['Board_85']; local_ch = global_ch - 32
    else: 
        grp = f['Board_75']; local_ch = global_ch - 96
        
    dt_ns = grp.attrs['sampling_period_ns']
    trace = grp['waveforms'][event_idx, local_ch, :].copy() 
    
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
        trace -= grp['baseline_mean'][event_idx, local_ch]
        
    t_us = np.arange(len(trace)) * dt_ns / 1000.0
    return t_us, trace

def main():
    parser = argparse.ArgumentParser(description="6x15 Averaged Waveform Plotter")
    parser.add_argument('input_target', help="HDF5 file or directory")
    parser.add_argument('--events', '-e', type=int, default=100, help="Number of events to average")
    parser.add_argument('--x_max', '-x', type=float, default=None, help="Max limit for the Time (us) x-axis")
    args = parser.parse_args()

    # File locating logic
    h5_files = []
    if os.path.isfile(args.input_target): 
        h5_files.append(args.input_target)
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5_files.extend([os.path.join(root, f) for f in files if f.endswith('.h5')])
    h5_files.sort()

    if not h5_files:
        print("[FAIL] No HDF5 files found in target.")
        return

    # Extract mapping from the first file to establish the geometry
    with h5py.File(h5_files[0], 'r') as f:
        config_mode = detect_config(f, h5_files[0])
        mapping = dc.get_channel_map(config_mode)

    # Reverse arrays (SIG14 -> SIG0) to match the physical L-shape trace
    banks = [
        ("A", mapping.get("Charge_A", [])[::-1]), 
        ("B", mapping.get("Charge_B", [])[::-1]), 
        ("C", mapping.get("Charge_C", [])[::-1]), 
        ("D", mapping.get("Charge_D", [])[::-1]), 
        ("E", mapping.get("Charge_E", [])[::-1]), 
        ("F", mapping.get("Charge_F", [])[::-1])  
    ]

    all_charge_chans = []
    for _, chans in banks: 
        all_charge_chans.extend(chans)

    # Accumulation dictionaries
    accumulated_traces = {ch: [] for ch in all_charge_chans}
    time_axis = None
    events_collected = 0

    print(f"\nTargeting {args.events} events for averaging...")
    
    with tqdm(total=args.events, desc="Accumulating Traces") as pbar:
        for f_path in h5_files:
            if events_collected >= args.events: break
            try:
                with h5py.File(f_path, 'r') as f:
                    # Skip files that haven't been baseline corrected
                    if 'Board_85' in f and 'baseline_mean' not in f['Board_85']: continue
                    
                    is_single_board = 'waveforms_mV' in f
                    n_in_file = f['waveforms_mV'].shape[0] if is_single_board else f['Board_72']['waveforms'].shape[0]

                    # Cap the loop so we don't over-collect
                    grab_amount = min(n_in_file, args.events - events_collected)

                    for i in range(grab_amount):
                        for ch in all_charge_chans:
                            t, v = get_trace(f, i, ch)
                            if not np.all(np.isnan(v)):
                                accumulated_traces[ch].append(v)
                                if time_axis is None: time_axis = t

                        events_collected += 1
                        pbar.update(1)
            except Exception as e:
                print(f"Error reading {f_path}: {e}")
                continue

    if events_collected == 0:
        print("\n[FAIL] No valid events found to average.")
        return

    print(f"\nSuccessfully collected {events_collected} events. Plotting 6x15 array...")

    # Setup 6x15 Plot
    fig, axes = plt.subplots(6, 15, figsize=(32, 20), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.3)

    y_min_global, y_max_global = np.inf, -np.inf

    for row_idx, (bank_name, chans) in enumerate(banks):
        is_x_anode = row_idx < 3
        trace_color = 'navy' if is_x_anode else 'darkred'
        
        for col_idx in range(15):
            ax = axes[row_idx, col_idx]
            
            # If the bank has a channel for this column index, plot it
            if col_idx < len(chans):
                ch = chans[col_idx]
                ax.set_title(f"Bank {bank_name} | CH {ch}", fontsize=9, fontweight='bold')
                
                # Darker, more visible gridding
                ax.grid(True, which='both', linestyle=':', linewidth=1.2, alpha=0.7, color='gray')
                
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                
                if accumulated_traces[ch]:
                    mean_trace = np.mean(accumulated_traces[ch], axis=0)
                    ax.plot(time_axis, mean_trace, color=trace_color, alpha=1.0, linewidth=1.5)
                    
                    y_min_global = min(y_min_global, np.nanmin(mean_trace))
                    y_max_global = max(y_max_global, np.nanmax(mean_trace))
                else:
                    ax.set_facecolor('#eaeaea')
                    ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center', transform=ax.transAxes, color='grey', fontweight='bold')
            # If there is no channel (N/C panels)
            else:
                ax.set_facecolor('#eaeaea')
                ax.text(0.5, 0.5, 'N/C', ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=14, fontweight='bold')
                # Safely turn off spines and tick parameters without destroying the shared axes
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(which='both', bottom=False, left=False)

    # Force tick labels to only appear on the outer perimeter of the 6x15 array
    for ax in axes.flat:
        ax.label_outer()

    # Format the global axes limits
    buffer = (y_max_global - y_min_global) * 0.1 if y_max_global != -np.inf else 1.0
    axes[0, 0].set_ylim(y_min_global - buffer, y_max_global + buffer)
    if args.x_max is not None: axes[0, 0].set_xlim(0, args.x_max)
    
    for row in range(6): 
        axes[row, 0].set_ylabel("Amplitude (mV)", fontsize=11, fontweight='bold')
        axes[row, 0].tick_params(axis='y', labelsize=10)
    for col in range(15): 
        axes[5, col].set_xlabel("Time (us)", fontsize=11, fontweight='bold')
        axes[5, col].tick_params(axis='x', labelsize=10, rotation=45)
        
    fig.suptitle(f"MiniGRAMS Charge Readout (Banks A-F)\nAveraged Raw Voltage ({events_collected} Events) | Config: {config_mode}", 
                 fontsize=20, fontweight='bold', y=0.94)

    # Output Routing
    base_dir = os.path.dirname(h5_files[0]) if os.path.isfile(args.input_target) else args.input_target
    save_dir = os.path.join(base_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    acq_name = os.path.basename(os.path.normpath(args.input_target))
    if not acq_name.startswith("acq"): acq_name = "Acq"
    
    out_file = os.path.join(save_dir, f"{acq_name}_All_Banks_6x15_Averaged.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[SUCCESS] Saved array plot to: {out_file}\n")

if __name__ == "__main__":
    main()