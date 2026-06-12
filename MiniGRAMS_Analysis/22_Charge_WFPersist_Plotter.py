"""
persist_banks_b_e.py
LArTPC Quick-Check: Oscilloscope Persist Plots for Banks B and E.
Focuses on deep-TPC channels near the Thorium rod.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import detector_config_beta as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_trace(f, event_idx, global_ch):
    """Fetches and baseline-corrects a trace based on global channel index."""
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

def check_coincidence(f, event_idx, light_chans, charge_chans, light_thresh, charge_thresh):
    """Strictly requires at least one light AND one charge channel to fire."""
    has_light = False
    for ch in light_chans:
        _, v = get_trace(f, event_idx, ch)
        if not np.all(np.isnan(v)) and np.nanmax(np.abs(v)) > light_thresh:
            has_light = True
            break
            
    if not has_light: return False

    has_charge = False
    for ch in charge_chans:
        _, v = get_trace(f, event_idx, ch)
        if not np.all(np.isnan(v)) and np.nanmax(np.abs(v)) > charge_thresh:
            has_charge = True
            break
            
    return has_charge

def main():
    parser = argparse.ArgumentParser(description="Generate Persist Plots for Banks B and E")
    parser.add_argument('input_target', help="HDF5 file or directory containing HDF5 files")
    parser.add_argument('--events', '-e', type=int, default=150, help="Number of coincidence events to stack")
    parser.add_argument('--charge_thresh', '-c', type=float, default=10.0, help="Charge trigger threshold (mV)")
    parser.add_argument('--light_thresh', '-l', type=float, default=5.0, help="Light trigger threshold (mV)")
    args = parser.parse_args()

    # Locate files
    h5_files = []
    if os.path.isfile(args.input_target): 
        h5_files.append(args.input_target)
    else:
        for root, dirs, files in os.walk(args.input_target):
            for f in files:
                if f.endswith('.h5'): h5_files.append(os.path.join(root, f))
    h5_files.sort()

    if not h5_files:
        print("[FAIL] No HDF5 files found.")
        return

    # Load Hardware Mapping
    mapping = dc.get_channel_map("MINIG")
    bank_b = mapping["Charge_B"] # Length: 13
    bank_e = mapping["Charge_E"] # Length: 15
    light_chans = mapping["VUV"] + mapping["VIS"]
    target_charge_chans = bank_b + bank_e

    # Setup the massive 2 x 15 plot array
    # Using sharey=True so we can easily compare relative pulse heights across all channels
    fig, axes = plt.subplots(2, 15, figsize=(32, 8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.2)
    
    # Configure axes titles and hide unused Bank B subplots
    for i, ch in enumerate(bank_b):
        axes[0, i].set_title(f"Bank B | CH {ch}", fontsize=10, fontweight='bold')
    for i in range(len(bank_b), 15):
        axes[0, i].axis('off') # Hide unused slots

    for i, ch in enumerate(bank_e):
        axes[1, i].set_title(f"Bank E | CH {ch}", fontsize=10, fontweight='bold')

    # Formatting aesthetics
    for ax in axes.flatten():
        if ax.axis() != (0.0, 1.0, 0.0, 1.0): # Only format active axes
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_facecolor('#f7f7f7') # Slight grey background for contrast with traces

    events_plotted = 0
    y_min_global, y_max_global = np.inf, -np.inf

    print(f"Hunting for {args.events} coincidence events...")
    
    with tqdm(total=args.events, desc="Stacking Traces") as pbar:
        for f_path in h5_files:
            if events_plotted >= args.events: break
            
            try:
                with h5py.File(f_path, 'r') as f:
                    if 'baseline_mean' not in f['Board_72']: continue
                    n_in_file = f['Board_72']['waveforms'].shape[0]
                    
                    for i in range(n_in_file):
                        if events_plotted >= args.events: break
                        
                        # Apply Coincidence Filter
                        if not check_coincidence(f, i, light_chans, target_charge_chans, args.light_thresh, args.charge_thresh):
                            continue
                        
                        # Valid Event: Plot Bank B
                        for idx, ch in enumerate(bank_b):
                            t, v = get_trace(f, i, ch)
                            if not np.all(np.isnan(v)):
                                axes[0, idx].plot(t, v, color='navy', alpha=0.15, linewidth=0.8)
                                y_min_global = min(y_min_global, np.nanmin(v))
                                y_max_global = max(y_max_global, np.nanmax(v))

                        # Valid Event: Plot Bank E
                        for idx, ch in enumerate(bank_e):
                            t, v = get_trace(f, i, ch)
                            if not np.all(np.isnan(v)):
                                axes[1, idx].plot(t, v, color='darkred', alpha=0.15, linewidth=0.8)
                                y_min_global = min(y_min_global, np.nanmin(v))
                                y_max_global = max(y_max_global, np.nanmax(v))

                        events_plotted += 1
                        pbar.update(1)
                        
            except Exception as e:
                print(f"\n[ERR] Skipping file {os.path.basename(f_path)}: {e}")
                continue

    if events_plotted == 0:
        print("\n[WARNING] No events passed coincidence threshold! Lower your limits.")
        return

    # Standardize Y-limits across the board for fair comparison
    buffer = (y_max_global - y_min_global) * 0.1
    axes[0, 0].set_ylim(y_min_global - buffer, y_max_global + buffer)
    axes[0, 0].set_xlim(0, 100)
    axes[1, 0].set_xlim(0, 100)
    
    # Label outer axes
    axes[0, 0].set_ylabel("Amplitude (mV)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Amplitude (mV)", fontsize=12, fontweight='bold')
    for i in range(15):
        axes[1, i].set_xlabel("Time (us)", fontsize=10)

    fig.suptitle(f"Biased Th232 Rod TPC Charge Channels (Banks B & E) - Persist Plot\n{events_plotted} Coincidence Events | Charge Threshold: {args.charge_thresh}mV | Light Threshold: {args.light_thresh}mV | TPC HV: -777V", 
                 fontsize=16, fontweight='bold', y=0.98)

    # Save output
    save_dir = os.path.dirname(h5_files[0]) if os.path.isfile(args.input_target) else args.input_target
    out_file = os.path.join(save_dir, "Bank_B_E_Persist_Plot.png")
    
    print("Rendering final high-res image... (this might take a few seconds)")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SUCCESS] Saved array plot to: {out_file}")

if __name__ == "__main__":
    main()