"""
35_Single_Event_Clean.py
LArTPC Single Event Filter Visualizer (Clean 2x2 Grid)
- Top Row: SG-Smoothed Voltage
- Bottom Row: Gaussian-Smoothed Current (Derivative)
- Outputs to high-res PNG.
- UPDATED FOR RUN 7: Board 85 is Slave 1, Board 75 is Slave 2.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import detector_config as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- STYLE ---
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '--'

def get_trace(f, event_idx, global_ch):
    # RUN 7 Mapping: Board 85 is Slave 1, Board 75 is Slave 2
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
    return np.arange(len(trace)) * dt_ns / 1000.0, trace, dt_ns

def apply_filters(trace, res_ns, sg_win_ns, gauss_sigma_ns):
    win_samples = int(sg_win_ns / res_ns)
    if win_samples % 2 == 0: win_samples += 1
    if win_samples < 3: win_samples = 3
    
    try: 
        sg_voltage = savgol_filter(trace, window_length=win_samples, polyorder=2)
    except ValueError: 
        sg_voltage = np.zeros_like(trace)
        
    deriv = np.gradient(sg_voltage)
    gauss_current = gaussian_filter1d(deriv, sigma=(gauss_sigma_ns / res_ns))
    return sg_voltage, gauss_current

def main():
    parser = argparse.ArgumentParser(description="Clean 2x2 Charge Filter Visualizer")
    parser.add_argument('filename', help="Target HDF5 File")
    parser.add_argument('--event', '-e', type=int, required=True, help="Target Event ID")
    parser.add_argument('--sg_x', type=float, required=True, help="SG Window for X (ns)")
    parser.add_argument('--gauss_x', type=float, required=True, help="Gaussian Sigma for X (ns)")
    parser.add_argument('--sg_y', type=float, required=True, help="SG Window for Y (ns)")
    parser.add_argument('--gauss_y', type=float, required=True, help="Gaussian Sigma for Y (ns)")
    args = parser.parse_args()
    
    mapping = dc.get_channel_map("MINIG")
    x_chans = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"]
    y_chans = mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]

    with h5py.File(args.filename, 'r') as f:
        pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)
        max_t = (f['Board_85'].attrs['n_samples'] * f['Board_85'].attrs['sampling_period_ns']) / 1000.0
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        
        # [0,0]: X-Voltage | [0,1]: Y-Voltage
        # [1,0]: X-Current | [1,1]: Y-Current
        
        planes = [
            (x_chans, axes[0,0], axes[1,0], args.sg_x, args.gauss_x, "X-Anode"),
            (y_chans, axes[0,1], axes[1,1], args.sg_y, args.gauss_y, "Y-Anode")
        ]
        
        for chans, ax_v, ax_i, sg_win, gauss_sig, title in planes:
            for ch in chans:
                t, raw_v, res_ns = get_trace(f, args.event, ch)
                if np.all(np.isnan(raw_v)): continue
                    
                sg_v, gauss_i = apply_filters(raw_v, res_ns, sg_win, gauss_sig)
                
                # Plot Voltage
                ax_v.plot(t, sg_v, alpha=0.7, lw=1.2, rasterized=True)
                # Plot Current
                ax_i.plot(t, gauss_i, alpha=0.7, lw=1.2, rasterized=True)

            ax_v.set_title(f"{title} Voltage\n(SG: {sg_win}ns)", fontweight='bold')
            ax_v.set_ylabel("Amplitude (mV)")
            
            ax_i.set_title(f"{title} Induced Current\n(Gauss: {gauss_sig}ns)", fontweight='bold')
            ax_i.set_ylabel("Current (arb)")
            ax_i.set_xlabel("Time (us)")

        for ax in axes.flatten():
            ax.set_xlim(0, max_t)
            if pre_trig_us > 0: ax.axvline(pre_trig_us, color='black', ls='--', lw=1.5, alpha=0.6)

        fig.suptitle(f"Filtered Charge Responses | Event {args.event}", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        out_dir = os.path.join(os.path.dirname(args.filename), "Plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"Clean_Filtered_Evt{args.event}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved PNG to: {out_path}")

if __name__ == "__main__": 
    main()