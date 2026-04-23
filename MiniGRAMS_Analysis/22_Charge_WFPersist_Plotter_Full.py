"""
persist_6x15_all_banks.py
LArTPC 6-Bank Persist Plotter (Raw Voltage & Ramo Current DSP)
- Updated: Dynamic Motherboard Light Trigger Mask (-m MB1, MB2, MB3, ALL).
- Updated: Separate DSP tuning arguments for X-Anode and Y-Anode.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import detector_config_beta as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_trace(f, event_idx, global_ch):
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

def apply_ramo_dsp(trace, sg_win, sg_poly, g_sig):
    """Applies Savitzky-Golay, Differentiation, and Gaussian smoothing."""
    if sg_win % 2 == 0: sg_win += 1 
    if len(trace) < sg_win: return np.zeros_like(trace)
    
    sg_trace = savgol_filter(trace, window_length=sg_win, polyorder=sg_poly)
    deriv = np.gradient(sg_trace)
    ramo_current = gaussian_filter1d(deriv, sigma=g_sig)
    
    return ramo_current

def check_coincidence(f, event_idx, light_chans, charge_chans, light_thresh, charge_thresh):
    has_light = False
    for ch in light_chans:
        _, v = get_trace(f, event_idx, ch)
        # Using the negative-going check from your working script
        if not np.all(np.isnan(v)) and np.nanmin(v) < -abs(light_thresh): 
            has_light = True
            break
    if not has_light: return False

    has_charge = False
    for ch in charge_chans:
        _, v = get_trace(f, event_idx, ch)
        # Using the absolute max check from your working script
        if not np.all(np.isnan(v)) and np.nanmax(np.abs(v)) > charge_thresh:
            has_charge = True
            break
    return has_charge

def main():
    parser = argparse.ArgumentParser(description="6x15 Persist Plotter (Raw & DSP)")
    parser.add_argument('input_target', help="HDF5 file or directory")
    parser.add_argument('--events', '-e', type=int, default=100)
    parser.add_argument('--charge_thresh', '-c', type=float, default=40.0)
    parser.add_argument('--light_thresh', '-l', type=float, default=30.0)
    parser.add_argument('--x_max', '-x', type=float, default=None, help="Max limit for the Time (us) x-axis")
    
    # Motherboard Trigger Mask
    parser.add_argument('-m', '--motherboards', nargs='+', default=['ALL'], 
                        choices=['MB1', 'MB2', 'MB3', 'ALL'], 
                        help="Motherboards to include in light trigger mask (e.g., -m MB1)")
    
    # DSP Arguments
    parser.add_argument('--apply_dsp', action='store_true', help="Toggle on Ramo Current extraction")
    
    parser.add_argument('--x_sg', type=int, default=15, help="X-Anode SG window length (samples)")
    parser.add_argument('--x_sg_p', type=int, default=2, help="X-Anode SG polynomial order")
    parser.add_argument('--x_g', type=float, default=2.0, help="X-Anode Gaussian sigma (samples)")
    
    parser.add_argument('--y_sg', type=int, default=15, help="Y-Anode SG window length (samples)")
    parser.add_argument('--y_sg_p', type=int, default=2, help="Y-Anode SG polynomial order")
    parser.add_argument('--y_g', type=float, default=2.0, help="Y-Anode Gaussian sigma (samples)")
    
    args = parser.parse_args()

    # File locating logic
    h5_files = []
    if os.path.isfile(args.input_target): h5_files.append(args.input_target)
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5_files.extend([os.path.join(root, f) for f in files if f.endswith('.h5')])
    h5_files.sort()

    if not h5_files:
        print("[FAIL] No HDF5 files found.")
        return

    mapping = dc.get_channel_map("MINIG")
    
    # Dynamic Trigger Mask Logic (Strictly tied to the hardware index)
    all_light = mapping["VUV"] + mapping["VIS"]
    trigger_light_chans = []

    if 'ALL' in args.motherboards:
        trigger_light_chans = all_light
    else:
        if 'MB1' in args.motherboards:
            trigger_light_chans.extend([ch for ch in all_light if ch < 12]) # Channels 0-11
        if 'MB2' in args.motherboards:
            trigger_light_chans.extend([ch for ch in all_light if 12 <= ch < 24]) # Channels 12-23
        if 'MB3' in args.motherboards:
            trigger_light_chans.extend([ch for ch in all_light if ch >= 24]) # Channels 24+
            
    print(f"Trigger Mask Active: {args.motherboards} ({len(trigger_light_chans)} channels)")
    
    banks = [
        ("A", mapping["Charge_A"]), # Row 0
        ("B", mapping["Charge_B"]), # Row 1
        ("C", mapping["Charge_C"]), # Row 2
        ("D", mapping["Charge_D"]), # Row 3
        ("E", mapping["Charge_E"]), # Row 4
        ("F", mapping["Charge_F"])  # Row 5
    ]
    
    all_charge_chans = []
    for _, chans in banks: all_charge_chans.extend(chans)

    # Setup 6x15 Plot
    fig, axes = plt.subplots(6, 15, figsize=(32, 20), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.3)
    
    # Configure Titles and hide unused axes
    for row_idx, (bank_name, chans) in enumerate(banks):
        for col_idx in range(15):
            if col_idx < len(chans):
                axes[row_idx, col_idx].set_title(f"Bank {bank_name} | CH {chans[col_idx]}", fontsize=8, fontweight='bold')
                axes[row_idx, col_idx].grid(True, linestyle='--', alpha=0.3)
                axes[row_idx, col_idx].set_facecolor('#f7f7f7')
            else:
                axes[row_idx, col_idx].axis('off') 

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
                        
                        # Use dynamic trigger mask
                        if not check_coincidence(f, i, trigger_light_chans, all_charge_chans, args.light_thresh, args.charge_thresh):
                            continue
                        
                        # Plot all banks
                        for row_idx, (bank_name, chans) in enumerate(banks):
                            is_x_anode = row_idx < 3
                            trace_color = 'navy' if is_x_anode else 'darkred'
                            
                            for col_idx, ch in enumerate(chans):
                                t, v = get_trace(f, i, ch)
                                if not np.all(np.isnan(v)):
                                    
                                    # APPLY DSP IF REQUESTED
                                    if args.apply_dsp:
                                        if is_x_anode:
                                            v = apply_ramo_dsp(v, args.x_sg, args.x_sg_p, args.x_g)
                                        else:
                                            v = apply_ramo_dsp(v, args.y_sg, args.y_sg_p, args.y_g)
                                        
                                    axes[row_idx, col_idx].plot(t, v, color=trace_color, alpha=0.15, linewidth=0.8)
                                    y_min_global = min(y_min_global, np.nanmin(v))
                                    y_max_global = max(y_max_global, np.nanmax(v))

                        events_plotted += 1
                        pbar.update(1)
            except Exception as e:
                continue

    if events_plotted == 0:
        print("\n[WARNING] No events passed coincidence threshold.")
        return

    buffer = (y_max_global - y_min_global) * 0.1
    axes[0, 0].set_ylim(y_min_global - buffer, y_max_global + buffer)
    
    # Apply optional x-axis limit
    if args.x_max is not None:
        axes[0, 0].set_xlim(0, args.x_max)
    
    y_label = "Derivative / Ramo Current (arb)" if args.apply_dsp else "Amplitude (mV)"
    for row in range(6): axes[row, 0].set_ylabel(y_label, fontsize=10, fontweight='bold')
    for col in range(15): axes[5, col].set_xlabel("Time (us)", fontsize=10)

    mode_title = "DSP RAMO CURRENT" if args.apply_dsp else "RAW VOLTAGE"
    
    param_text = ""
    if args.apply_dsp:
        param_text = f"\nX-Anode (SG:{args.x_sg}, Poly:{args.x_sg_p}, G:{args.x_g}) | Y-Anode (SG:{args.y_sg}, Poly:{args.y_sg_p}, G:{args.y_g})"
        
    fig.suptitle(f"Whole TPC (Banks A-F) - {mode_title} Persist Plot\n{events_plotted} Events | Charge Thresh: {args.charge_thresh}mV | Light Thresh: {args.light_thresh}mV | Trigger Mask: {args.motherboards}{param_text}", 
                 fontsize=18, fontweight='bold', y=0.96)

    out_suffix = "RamoDSP" if args.apply_dsp else "Raw"
    save_dir = os.path.dirname(h5_files[0]) if os.path.isfile(args.input_target) else args.input_target
    out_file = os.path.join(save_dir, f"All_Banks_6x15_{out_suffix}.png")
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SUCCESS] Saved array plot to: {out_file}")

if __name__ == "__main__":
    main()