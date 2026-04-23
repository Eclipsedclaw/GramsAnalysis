"""
14_Sanity_Check_Plot_SharedY.py
4-Panel Sanity Check Plot for LArTPC HDF5 Data
- Automatically hunts for a Golden Event.
- Shared Y-axes across SiPM panels and Charge panels.
- Smart dynamic Y-axis bounds for Charge channels.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import detector_config as dc
import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def find_golden_event(f):
    n_events = f['Board_72']['waveforms'].shape[0]
    for i in range(n_events):
        if not np.isnan(f['Board_75']['waveforms'][i, 0, 0]) and \
           not np.isnan(f['Board_85']['waveforms'][i, 0, 0]):
            return i
    return -1

def sanity_check_plot(hdf5_file, target_event=None, apply_baseline=False):
    mapping = dc.get_channel_map("MINIG") 
    
    vuv_chans = mapping["VUV"]
    vis_chans = mapping["VIS"]
    x_chans = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"]
    y_chans = mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]
    
    with h5py.File(hdf5_file, 'r') as f:
        if target_event is None:
            event_idx = find_golden_event(f)
            if event_idx == -1:
                print("[WARN] No Golden Events found! Plotting Event 0.")
                event_idx = 0
            else:
                print(f"[SUCCESS] Found Golden Event at Index: {event_idx}")
        else:
            event_idx = target_event
            print(f"--- Plotting User-Specified Event: {event_idx} ---")

        master_dt = f['Board_72'].attrs['sampling_period_ns']
        master_samples = f['Board_72'].attrs['n_samples']
        master_max_t = (master_samples * master_dt) / 1000.0

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
        
        def get_trace(global_ch):
            if global_ch < 32:
                grp = f['Board_72']
                local_ch = global_ch
            elif global_ch < 96:
                grp = f['Board_75']
                local_ch = global_ch - 32
            else:
                grp = f['Board_85']
                local_ch = global_ch - 96
            
            dt_ns = grp.attrs['sampling_period_ns']
            trace = grp['waveforms'][event_idx, local_ch, :].copy() 
            
            if apply_baseline:
                if 'baseline_mean' in grp:
                    bl_val = grp['baseline_mean'][event_idx, local_ch]
                    if not np.isnan(bl_val): trace -= bl_val
                else: print(f"[WARN] Baseline requested but not found in {grp.name}.")

            t_axis = np.arange(len(trace)) * dt_ns / 1000.0 
            return t_axis, trace

        # --- Top Row: SiPMs ---
        for ch in vuv_chans:
            t, v = get_trace(ch)
            axes[0,0].plot(t, v, alpha=0.5, linewidth=0.8)
        axes[0,0].set_title(f"VUV SiPMs (N={len(vuv_chans)})", fontweight='bold')
        axes[0,0].set_xlim(0, master_max_t) 
        
        for ch in vis_chans:
            t, v = get_trace(ch)
            axes[0,1].plot(t, v, alpha=0.5, linewidth=0.8)
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
        for ch in x_chans:
            t, v = get_trace(ch)
            axes[1,0].plot(t, v, alpha=0.4, linewidth=0.6)
        axes[1,0].set_title(f"Charge X-Direction (Banks A, B, C)", fontweight='bold')
        
        for ch in y_chans:
            t, v = get_trace(ch)
            axes[1,1].plot(t, v, alpha=0.4, linewidth=0.6)
        axes[1,1].set_title(f"Charge Y-Direction (Banks D, E, F)", fontweight='bold')

        # Sync and Smart-Limit Charge Y-Axes
        x_y = axes[1,0].get_ylim()
        y_y = axes[1,1].get_ylim()
        c_min = min(x_y[0], y_y[0])
        c_max = max(x_y[1], y_y[1])
        
        if apply_baseline:
            axes[1,0].set_ylim(min(-10, c_min - 2), max(20, c_max + 5))
            axes[1,1].set_ylim(min(-10, c_min - 2), max(20, c_max + 5))
        else:
            axes[1,0].set_ylim(min(-40, c_min - 5), max(75, c_max + 5))
            axes[1,1].set_ylim(min(-40, c_min - 5), max(75, c_max + 5))

    # Formatting
    for ax in axes.flatten():
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
    title_str = f"LArTPC Sanity Check - Event {event_idx}"
    if apply_baseline: title_str += " (Baseline Corrected)"
        
    fig.suptitle(title_str, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_name = f"SanityCheck_Evt{event_idx}_BLCorrected.png" if apply_baseline else f"SanityCheck_Evt{event_idx}.png"
    plt.savefig(save_name, dpi=200)
    print(f"--> Plot saved as {save_name}. Awaiting visual inspection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to the synced HDF5 file")
    parser.add_argument('--event', '-e', type=int, default=None, help="Specific event index to plot (optional)")
    parser.add_argument('--baseline', '-b', action='store_true', help="Apply baseline subtraction")
    args = parser.parse_args()
    sanity_check_plot(args.file, args.event, args.baseline)