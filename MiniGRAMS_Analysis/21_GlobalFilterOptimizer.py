"""
39_Global_Parameter_Optimizer_V3.py
LArTPC Global Parameter Grid Scanner (Polished)
- STRICT COINCIDENCE: Harvests traces passing Light & Charge thresholds.
- Generates 2D Grid Search (SG Window vs. Gaussian Sigma) on the ensemble.
- White background, global titles, parameter text boxes.
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
import glob
from tqdm import tqdm
import detector_config_beta as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- STYLE: White Background ---
plt.style.use('default')
cmap_choice = 'viridis'

def get_charge_trace(f, event_idx, global_ch):
    if global_ch < 32: return None, None
    elif global_ch < 96: grp, local_ch = f['Board_85'], global_ch - 32
    else: grp, local_ch = f['Board_75'], global_ch - 96
        
    dt_ns = grp.attrs['sampling_period_ns']
    trace = grp['waveforms'][event_idx, local_ch, :].copy() 
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
        trace -= grp['baseline_mean'][event_idx, local_ch]
    return trace, dt_ns

def check_light_trigger(f, event_idx, light_thresh):
    """Checks ONLY SiPM Motherboard 1 (Assuming CAEN Channels 0-11)"""
    b72 = f['Board_72']
    # Restrict to the first 12 channels (MB1) - adjust range if MB1 is mapped differently
    for ch in range(12): 
        trace = b72['waveforms'][event_idx, ch, :].copy()
        if 'baseline_mean' in b72 and not np.isnan(b72['baseline_mean'][event_idx, ch]):
            trace -= b72['baseline_mean'][event_idx, ch]
        
        if np.nanmin(trace) < -abs(light_thresh):
            return True
    return False

def calc_fom(trace, res_ns, sg_win, gauss_sig, trigger_idx):
    win = int(sg_win / res_ns)
    if win % 2 == 0: win += 1
    if win < 3: win = 3
    
    try: 
        sg_v = savgol_filter(trace, window_length=win, polyorder=2)
    except ValueError: 
        return 0.0
        
    deriv = np.gradient(sg_v)
    current = gaussian_filter1d(deriv, sigma=(gauss_sig / res_ns))
    
    safe_noise_end = max(1, trigger_idx - int(2000 / res_ns))
    noise_slice = current[:safe_noise_end]
    signal_slice = current[trigger_idx:]
    
    noise_rms = np.std(noise_slice)
    if noise_rms == 0 or np.isnan(noise_rms): return 0.0
    
    peak_val = np.max(np.abs(signal_slice))
    return peak_val / noise_rms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_target')
    parser.add_argument('--events', '-e', type=int, default=50)
    parser.add_argument('--grid_size', '-g', type=int, default=25)
    parser.add_argument('--win_min', type=float, default=2000.0)
    parser.add_argument('--win_max', type=float, default=10000.0)
    parser.add_argument('--charge_thresh', '-c', type=float, default=10.0)
    parser.add_argument('--light_thresh', '-l', type=float, default=75.0)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_target, "*.h5"))) if os.path.isdir(args.input_target) else [args.input_target]
    if not files: return

    mapping = dc.get_channel_map("MINIG")
    x_chans = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"]
    y_chans = mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]

    x_trace_pool, y_trace_pool = [], []
    events_harvested = 0
    res_ns, trigger_idx = 8.0, 2000 
    
    print("Harvesting strictly coincident traces...")
    for filepath in files:
        if events_harvested >= args.events: break
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean' not in f['Board_72']: continue
            n_events = f['Board_72']['waveforms'].shape[0]
            res_ns = f['Board_85'].attrs['sampling_period_ns']
            pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)
            trigger_idx = int((pre_trig_us * 1000) / res_ns)
            
            for i in range(n_events):
                if events_harvested >= args.events: break
                if np.isnan(f['Board_75']['waveforms'][i, 0, 0]): continue
                if not check_light_trigger(f, i, args.light_thresh): continue
                
                max_x_val, best_x_trace = 0, None
                for ch in x_chans:
                    tr, _ = get_charge_trace(f, i, ch)
                    if tr is not None:
                        peak = np.nanmax(np.abs(tr))
                        if peak > max_x_val:
                            max_x_val = peak; best_x_trace = tr
                
                max_y_val, best_y_trace = 0, None
                for ch in y_chans:
                    tr, _ = get_charge_trace(f, i, ch)
                    if tr is not None:
                        peak = np.nanmax(np.abs(tr))
                        if peak > max_y_val:
                            max_y_val = peak; best_y_trace = tr
                
                if max_x_val > args.charge_thresh and max_y_val > args.charge_thresh:
                    x_trace_pool.append(best_x_trace)
                    y_trace_pool.append(best_y_trace)
                    events_harvested += 1

    if len(x_trace_pool) == 0:
        print("[ERR] Not enough traces passing cuts.")
        return

    sg_vals = np.linspace(args.win_min, args.win_max, args.grid_size)
    gauss_vals = np.linspace(args.win_min, args.win_max, args.grid_size)
    x_fom_matrix = np.zeros((args.grid_size, args.grid_size))
    y_fom_matrix = np.zeros((args.grid_size, args.grid_size))

    print("Executing Grid Search...")
    with tqdm(total=args.grid_size*args.grid_size) as pbar:
        for i, sg in enumerate(sg_vals):
            for j, gauss in enumerate(gauss_vals):
                x_foms = [calc_fom(tr, res_ns, sg, gauss, trigger_idx) for tr in x_trace_pool]
                x_fom_matrix[i, j] = np.mean(x_foms) if x_foms else 0
                y_foms = [calc_fom(tr, res_ns, sg, gauss, trigger_idx) for tr in y_trace_pool]
                y_fom_matrix[i, j] = np.mean(y_foms) if y_foms else 0
                pbar.update(1)

    x_best_idx = np.unravel_index(np.argmax(x_fom_matrix), x_fom_matrix.shape)
    y_best_idx = np.unravel_index(np.argmax(y_fom_matrix), y_fom_matrix.shape)
    best_x_sg, best_x_gauss = sg_vals[x_best_idx[0]], gauss_vals[x_best_idx[1]]
    best_y_sg, best_y_gauss = sg_vals[y_best_idx[0]], gauss_vals[y_best_idx[1]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    fig.suptitle(f"Global DSP Parameter Optimization Matrix\nEnsemble Size: {events_harvested} Coincident Events", fontsize=16, fontweight='bold')

    # Text Box Properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

    # X-Anode Plot
    im_x = axes[0].pcolormesh(gauss_vals, sg_vals, x_fom_matrix, cmap=cmap_choice, shading='auto')
    axes[0].plot(best_x_gauss, best_x_sg, marker='*', color='red', markersize=18, markeredgecolor='white', markeredgewidth=1.5)
    axes[0].set_title(f"X-Anode (Banks A,B,C)", fontweight='bold')
    axes[0].set_xlabel("Gaussian Sigma (ns)"); axes[0].set_ylabel("SG Window (ns)")
    text_x = f"OPTIMAL PARAMETERS:\nSG Window: {best_x_sg:.0f} ns\nSG Poly Order: 2\nGauss Sigma: {best_x_gauss:.0f} ns\nMax FoM: {np.max(x_fom_matrix):.2f}"
    axes[0].text(0.05, 0.95, text_x, transform=axes[0].transAxes, va='top', bbox=props, fontsize=10, fontweight='bold')
    fig.colorbar(im_x, ax=axes[0], label='Mean Signal-to-Noise (FoM)')

    # Y-Anode Plot
    im_y = axes[1].pcolormesh(gauss_vals, sg_vals, y_fom_matrix, cmap=cmap_choice, shading='auto')
    axes[1].plot(best_y_gauss, best_y_sg, marker='*', color='red', markersize=18, markeredgecolor='white', markeredgewidth=1.5)
    axes[1].set_title(f"Y-Anode (Banks D,E,F)", fontweight='bold')
    axes[1].set_xlabel("Gaussian Sigma (ns)"); axes[1].set_ylabel("SG Window (ns)")
    text_y = f"OPTIMAL PARAMETERS:\nSG Window: {best_y_sg:.0f} ns\nSG Poly Order: 2\nGauss Sigma: {best_y_gauss:.0f} ns\nMax FoM: {np.max(y_fom_matrix):.2f}"
    axes[1].text(0.05, 0.95, text_y, transform=axes[1].transAxes, va='top', bbox=props, fontsize=10, fontweight='bold')
    fig.colorbar(im_y, ax=axes[1], label='Mean Signal-to-Noise (FoM)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_dir = os.path.join(os.path.dirname(args.input_target) if os.path.isfile(args.input_target) else args.input_target, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Global_Optimization_Map.png")
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved Global Heatmap to: {out_path}")

if __name__ == "__main__": 
    main()