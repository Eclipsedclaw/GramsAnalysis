import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import re
import detector_config as dc

# --- STYLE: PUBLICATION (White Background) ---
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = False # Heatmaps look better without grids
plt.rcParams['font.family'] = 'sans-serif'

def extract_voltage(filename):
    match = re.search(r'TPCHV(\d+)', filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)} V"
    return "Unknown Voltage"

def calculate_fom(trace, res_ns, pre_trig_us=16.0):
    """
    Calculates Ridge Contrast: Peak Signal / Noise RMS
    """
    t_axis = np.arange(len(trace)) * (res_ns / 1000.0)
    
    # Noise Window: Start to (Trigger - 1us)
    noise_mask = (t_axis < (pre_trig_us - 1.0))
    # Signal Window: Trigger to (Trigger + 80us)
    signal_mask = (t_axis > pre_trig_us) & (t_axis < (pre_trig_us + 80.0))
    
    if np.sum(noise_mask) == 0 or np.sum(signal_mask) == 0:
        return 0.0
        
    noise_seg = trace[noise_mask]
    signal_seg = trace[signal_mask]
    
    noise_rms = np.std(noise_seg)
    if noise_rms == 0: noise_rms = 1e-6
    
    signal_peak = np.max(signal_seg)
    
    return signal_peak / noise_rms

def process_waveform_test(trace, res_ns, sg_win_ns, gauss_sigma_ns, poly_order=2):
    win_samples = int(sg_win_ns / res_ns)
    if win_samples % 2 == 0: win_samples += 1
    if win_samples < 3: win_samples = 3
    
    try:
        smoothed = savgol_filter(trace, window_length=win_samples, polyorder=poly_order)
    except ValueError:
        return np.zeros_like(trace)
    
    deriv = np.gradient(smoothed)
    
    if gauss_sigma_ns > 0:
        sigma_samples = gauss_sigma_ns / res_ns
        final = gaussian_filter1d(deriv, sigma=sigma_samples)
    else:
        final = deriv
        
    return final

def run_optimizer(filename, event_id):
    print(f"--- OPTIMIZER v2: Event {event_id} ---")
    
    drift_voltage = extract_voltage(os.path.basename(filename))
    
    with h5py.File(filename, 'r') as f:
        run_config = f.attrs.get('run_config', 'UPS')
        res_ns = f.attrs.get('resolution_ns', 8.0)
        ch_map = dc.get_channel_map(run_config)
        
        waves = f['waveforms_mV'][event_id]
        if 'baseline_mean_mV' in f:
            baselines = f['baseline_mean_mV'][event_id]
        else:
            baselines = np.mean(waves[:, :20], axis=1)
            
    # Auto-detect Active Plane
    int_x = sum([np.sum(np.abs(waves[c]-baselines[c])) for c in ch_map['Charge_X']])
    int_y = sum([np.sum(np.abs(waves[c]-baselines[c])) for c in ch_map['Charge_Y']])
    
    target_channels = ch_map['Charge_X'] if int_x > int_y else ch_map['Charge_Y']
    plane_name = "X-Anode" if int_x > int_y else "Y-Anode"
    print(f"Optimizing on {plane_name} (Drift: {drift_voltage})...")

    # --- HIGH RES GRID ---
    # Increased to 50x50 for smoother gradients
    sg_windows = np.linspace(500, 5000, 50) 
    gauss_sigmas = np.linspace(100, 3000, 50)
    
    fom_grid = np.zeros((len(gauss_sigmas), len(sg_windows)))
    
    print(f"Grid Scan: {len(sg_windows)}x{len(gauss_sigmas)} points...")

    for i, sigma in enumerate(gauss_sigmas):
        print(f"  > Progress: {int((i/len(gauss_sigmas))*100)}%", end='\r')
        for j, win in enumerate(sg_windows):
            total_contrast = 0
            for ch in target_channels:
                raw = waves[ch] - baselines[ch]
                proc = process_waveform_test(raw, res_ns, win, sigma)
                score = calculate_fom(proc, res_ns)
                total_contrast += score
            fom_grid[i, j] = total_contrast / len(target_channels)
            
    print("\nScan Complete.")

    # Find Optimal Point
    max_idx = np.unravel_index(np.argmax(fom_grid), fom_grid.shape)
    best_sigma = gauss_sigmas[max_idx[0]]
    best_win = sg_windows[max_idx[1]]
    best_score = fom_grid[max_idx]
    
    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [sg_windows[0], sg_windows[-1], gauss_sigmas[0], gauss_sigmas[-1]]
    
    im = ax.imshow(fom_grid, origin='lower', aspect='auto', extent=extent, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Figure of Merit (Ridge Contrast)")
    
    # Mark Best Point
    ax.plot(best_win, best_sigma, 'r*', markersize=18, markeredgecolor='white', label='Global Max')
    
    # Labels & Titles
    ax.set_xlabel("Savitzky-Golay Window (ns)", fontsize=14)
    ax.set_ylabel("Post-Derivative Gaussian Sigma (ns)", fontsize=14)
    
    # Dynamic Title
    title_main = "Digital Filter Parameter Optimization"
    title_sub = f"Event {event_id} | {plane_name} | Drift Field: {drift_voltage}"
    ax.set_title(f"{title_main}\n{title_sub}", fontsize=14, pad=15)

    # Info Box (The "Readout")
    textstr = '\n'.join((
        r'$\bf{Optimal Parameters}$',
        f'SG Window: {best_win:.0f} ns',
        f'Gauss Sigma: {best_sigma:.0f} ns',
        f'Max FoM: {best_score:.2f}'))

    # Place text box in upper right (or left if peak is there)
    # Heuristic: if peak is on the right, move box to left
    box_x = 0.05 if best_win > 2500 else 0.65
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(box_x, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Save
    input_dir = os.path.dirname(filename)
    save_dir = os.path.join(input_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    out_name = f"Optimizer_V2_Evt{event_id}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Publication Plot to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="HDF5 File")
    parser.add_argument('--event', '-e', type=int, required=True)
    args = parser.parse_args()
    
    run_optimizer(args.filename, args.event)