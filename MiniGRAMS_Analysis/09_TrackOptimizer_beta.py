"""
Digital Filter Parameter Optimizer for Multi-Board LArTPC Data
- Scans a 2D grid of Savitzky-Golay windows and Gaussian Sigmas.
- Evaluates X-Anode and Y-Anode independently.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import re
import detector_config as dc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- STYLE ---
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'

def extract_voltage(filename):
    match = re.search(r'(?i)HV(\d+)', filename)
    if match: return f"{match.group(1)} V"
    return "Unknown Voltage"

def get_charge_trace(f, event_id, global_ch):
    if global_ch < 32: return None, None 
    elif global_ch < 96: grp, local_ch = f['Board_75'], global_ch - 32
    else: grp, local_ch = f['Board_85'], global_ch - 96

    raw = grp['waveforms'][event_id, local_ch]
    if np.isnan(raw[0]): return None, None
    
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_id, local_ch]):
        bl = grp['baseline_mean'][event_id, local_ch]
    else:
        bl = np.nanmedian(raw[:1000])
        
    return raw - bl, grp.attrs['sampling_period_ns']

def calculate_fom(trace, res_ns, pre_trig_us=16.0):
    if trace is None or np.all(np.isnan(trace)): return 0.0
    t_axis = np.arange(len(trace)) * (res_ns / 1000.0)
    
    noise_mask = (t_axis < (pre_trig_us - 1.0))
    signal_mask = (t_axis > pre_trig_us) & (t_axis < (pre_trig_us + 150.0))
    
    if np.sum(noise_mask) == 0 or np.sum(signal_mask) == 0: return 0.0
        
    noise_seg, signal_seg = trace[noise_mask], trace[signal_mask]
    noise_rms = np.std(noise_seg)
    if noise_rms == 0: noise_rms = 1e-6
    
    return np.max(signal_seg) / noise_rms

def process_waveform_test(trace, res_ns, sg_win_ns, gauss_sigma_ns, poly_order=2):
    win_samples = int(sg_win_ns / res_ns)
    if win_samples % 2 == 0: win_samples += 1
    if win_samples < 3: win_samples = 3
    
    try: smoothed = savgol_filter(trace, window_length=win_samples, polyorder=poly_order)
    except ValueError: return np.zeros_like(trace)
    
    deriv = np.gradient(smoothed)
    
    if gauss_sigma_ns > 0:
        return gaussian_filter1d(deriv, sigma=(gauss_sigma_ns / res_ns))
    return deriv

def run_grid_scan(f, event_id, target_channels, sg_windows, gauss_sigmas, pre_trig_us, plane_name):
    """Executes the 2D grid scan with a tqdm progress bar."""
    fom_grid = np.zeros((len(gauss_sigmas), len(sg_windows)))
    
    # Wrap the outer loop with tqdm for progress tracking
    for i, sigma in enumerate(tqdm(gauss_sigmas, desc=f"Scanning {plane_name}", unit="row")):
        for j, win in enumerate(sg_windows):
            total_contrast = 0
            valid_chs = 0
            for ch in target_channels:
                trace, res_ns = get_charge_trace(f, event_id, ch)
                if trace is not None:
                    proc = process_waveform_test(trace, res_ns, win, sigma)
                    total_contrast += calculate_fom(proc, res_ns, pre_trig_us)
                    valid_chs += 1
            fom_grid[i, j] = total_contrast / valid_chs if valid_chs > 0 else 0
            
    return fom_grid

def run_optimizer(filename, event_id, max_sg, max_gauss):
    print(f"\n--- DUAL-PLANE OPTIMIZER: Event {event_id} ---")
    drift_voltage = extract_voltage(filename)
    
    with h5py.File(filename, 'r') as f:
        ch_map = dc.get_channel_map("MINIG")
        x_chans = ch_map["Charge_A"] + ch_map["Charge_B"] + ch_map["Charge_C"]
        y_chans = ch_map["Charge_D"] + ch_map["Charge_E"] + ch_map["Charge_F"]
        pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)
        
        # Rank channels by activity
        x_integrals, y_integrals = [], []
        for ch in x_chans:
            trace, _ = get_charge_trace(f, event_id, ch)
            if trace is not None: x_integrals.append((ch, np.sum(np.abs(trace))))
        for ch in y_chans:
            trace, _ = get_charge_trace(f, event_id, ch)
            if trace is not None: y_integrals.append((ch, np.sum(np.abs(trace))))
            
        x_integrals.sort(key=lambda item: item[1], reverse=True)
        y_integrals.sort(key=lambda item: item[1], reverse=True)
        
        target_x = [item[0] for item in x_integrals[:5]]
        target_y = [item[0] for item in y_integrals[:5]]
        
        sg_windows = np.linspace(500, max_sg, 50) 
        gauss_sigmas = np.linspace(100, max_gauss, 50)
        
        print(f"\nTargeting X-Anode Channels: {target_x}")
        fom_grid_x = run_grid_scan(f, event_id, target_x, sg_windows, gauss_sigmas, pre_trig_us, "X-Anode")
        
        print(f"\nTargeting Y-Anode Channels: {target_y}")
        fom_grid_y = run_grid_scan(f, event_id, target_y, sg_windows, gauss_sigmas, pre_trig_us, "Y-Anode")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    extent = [sg_windows[0], sg_windows[-1], gauss_sigmas[0], gauss_sigmas[-1]]
    
    grids = [("X-Anode", fom_grid_x, axes[0]), ("Y-Anode", fom_grid_y, axes[1])]
    
    for name, grid, ax in grids:
        im = ax.imshow(grid, origin='lower', aspect='auto', extent=extent, cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Ridge Contrast (FoM)")
        
        max_idx = np.unravel_index(np.argmax(grid), grid.shape)
        best_sigma = gauss_sigmas[max_idx[0]]
        best_win = sg_windows[max_idx[1]]
        best_score = grid[max_idx]
        
        ax.plot(best_win, best_sigma, 'r*', markersize=16, markeredgecolor='white')
        ax.set_xlabel("Savitzky-Golay Window (ns)")
        if name == "X-Anode": ax.set_ylabel("Post-Derivative Gaussian Sigma (ns)")
        ax.set_title(f"{name} Optimization", fontweight='bold')
        
        textstr = '\n'.join((f'SG Window: {best_win:.0f} ns', f'Gauss Sigma: {best_sigma:.0f} ns', f'Max FoM: {best_score:.2f}'))
        box_x = 0.05 if best_win > (max_sg / 2) else 0.55
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(box_x, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)

    fig.suptitle(f"Filter Optimization | Event {event_id} | Drift Field: {drift_voltage}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(filename), "Plots")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"Optimizer_Dual_Evt{event_id}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="HDF5 File")
    parser.add_argument('--event', '-e', type=int, required=True, help="Event ID")
    parser.add_argument('--max_sg', type=float, default=8000.0, help="Max SG window search limit (ns)")
    parser.add_argument('--max_gauss', type=float, default=6000.0, help="Max Gaussian Sigma search limit (ns)")
    args = parser.parse_args()
    
    run_optimizer(args.filename, args.event, args.max_sg, args.max_gauss)