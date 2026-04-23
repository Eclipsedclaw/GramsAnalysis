"""
05_pedestal_quality.py
MicroGRAMS Analysis Pipeline
----------------------------
Generates "True RMS" Dashboard using Gaussian Fits.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import argparse
import time
import glob
import detector_config as dc

# --- CONFIGURATION ---
COLORS = {
    'Charge_A': '#E57373', # Red
    'Charge_B': '#64B5F6', # Blue
    'VUV':      '#546E7A', # Dark Slate Grey
    'VIS':      '#B0BEC5', # Light Grey
    'Charge_C': '#81C784', # Green
    'Charge_D': '#BA68C8', # Purple
}

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def get_gaussian_stats(data_array):
    """
    Fits a Gaussian to the distribution of RMS values.
    Returns: (mu, sigma)
    """
    # Filter out dead zeros
    data = data_array[data_array > 0.001]
    if len(data) < 10: return 0.0, 0.0

    # 1. Histogram
    med = np.median(data)
    std = np.std(data)
    if std == 0: std = 0.01
    
    # Window +/- 4 sigmas around median
    bins = np.linspace(max(0, med - 4*std), med + 4*std, 40)
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Fit
    try:
        p0 = [max(counts), med, std]
        popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=p0, maxfev=2000)
        mu, sigma = popt[1], abs(popt[2])
        return mu, sigma
    except:
        return np.mean(data), np.std(data)

def process_aggregation(input_path, gain_factor):
    start_t = time.time()
    
    # Gather Files
    if os.path.isfile(input_path):
        files = [input_path]
        root_dir = os.path.dirname(input_path)
    else:
        files = glob.glob(os.path.join(input_path, "*.h5"))
        root_dir = input_path
        
    if not files:
        print("[ERR] No HDF5 files found.")
        return

    print(f"--> Aggregating {len(files)} files...")
    
    all_std_data = []
    config_mode = 'OSAKA' 
    
    for f_path in files:
        try:
            with h5py.File(f_path, 'r') as f:
                if 'baseline_std_mV' in f:
                    data = f['baseline_std_mV'][:] * gain_factor
                    all_std_data.append(data)
                    if 'run_config' in f.attrs:
                        config_mode = f.attrs['run_config']
        except Exception as e:
            print(f"    [WARN] Skipping {os.path.basename(f_path)}: {e}")

    if not all_std_data:
        print("[ERR] No baseline data found.")
        return

    combined_std = np.vstack(all_std_data)
    total_events, n_channels = combined_std.shape
    
    print(f"    [INFO] Total Events: {total_events}")
    print(f"    [INFO] Calculating Gaussian Fits...")

    # Calculate True RMS
    true_rms_means = np.zeros(n_channels)
    true_rms_errs = np.zeros(n_channels)
    
    for ch in range(n_channels):
        mu, sigma = get_gaussian_stats(combined_std[:, ch])
        true_rms_means[ch] = mu
        true_rms_errs[ch] = sigma

    # Plotting
    ch_map = dc.get_channel_map(config_mode)
    save_dir = os.path.join(root_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    
    # --- BAR CHART ---
    ax_main = fig.add_subplot(gs[0, :])
    groups_ordered = ['Charge_A', 'Charge_B', 'VUV', 'VIS', 'Charge_C', 'Charge_D']
    all_charge_rms = []
    
    for group in groups_ordered:
        if group not in ch_map: continue
        
        indices_0 = ch_map[group]
        if not indices_0: continue
        
        valid_idx_0 = [i for i in indices_0 if i < n_channels]
        if not valid_idx_0: continue
        
        vals = true_rms_means[valid_idx_0]
        errs = true_rms_errs[valid_idx_0]
        
        # Display Mapping (Shift +2 if OSAKA)
        if "OSAKA" in config_mode:
            indices_display = [i + 2 for i in valid_idx_0]
            xlabel_text = "Physical Channel ID (Caen 2-62)"
            xlims = (0, 64)
            xticks = range(2, 63)
        else:
            indices_display = valid_idx_0
            xlabel_text = "Channel Index"
            xlims = (-1, n_channels)
            xticks = range(n_channels)

        c = COLORS.get(group, '#333333')
        label = group.replace("Charge_", "Bank ")
        
        ax_main.bar(indices_display, vals, yerr=errs, color=c, label=label, 
                    capsize=2, alpha=0.9, width=0.8)
        
        if 'Charge' in group:
            all_charge_rms.extend(vals)

    ax_main.set_title(f"True RMS Noise (Gaussian Fit)\nEvents Included: {total_events}", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("RMS Noise (mV)")
    ax_main.set_xlabel(xlabel_text)
    ax_main.set_xlim(xlims)
    
    if len(xticks) <= 64:
        ax_main.set_xticks(xticks)
        ax_main.set_xticklabels(xticks, rotation=90, fontsize=8)
    ax_main.grid(axis='y', linestyle='--', alpha=0.5)
    ax_main.legend(loc='upper left', ncol=6)

    # --- HISTOGRAM ---
    ax_hist = fig.add_subplot(gs[1, 0])
    if all_charge_rms:
        clean_rms = [x for x in all_charge_rms if x > 0.01]
        ax_hist.hist(clean_rms, bins=20, color='teal', alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel("RMS Noise (mV)")
        ax_hist.set_ylabel("Count (Charge Channels)")
        ax_hist.set_title("Charge Array Uniformity")
        ax_hist.axvline(np.median(clean_rms), color='red', linestyle='--', label='Median')
        ax_hist.legend()

    # --- STATS ---
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')
    elapsed = time.time() - start_t
    
    if all_charge_rms:
        clean_rms = np.array([x for x in all_charge_rms if x > 0.01])
        txt = "--- V5 Unmodified CSP STATS ---\n\n"
        txt += f"Method:       Gaussian Fit Mode\n"
        txt += f"Events:       {total_events}\n"
        txt += f"Channels:     {len(clean_rms)}\n\n"
        txt += f"Median Noise: {np.median(clean_rms):.3f} mV\n"
        txt += f"Mean Noise:   {np.mean(clean_rms):.3f} mV\n"
        txt += f"Uniformity:   {np.std(clean_rms):.3f} mV\n\n"
        txt += f"Total Time:   {elapsed:.2f} s"
        
        ax_text.text(0.1, 0.5, txt, fontsize=11, fontfamily='monospace', va='center')

    plt.tight_layout()
    acq_name = os.path.basename(os.path.normpath(input_path))
    save_path = os.path.join(save_dir, f"TrueRMS_{acq_name}_debug.png")
    
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"--> [DONE] Saved: {save_path}")
    print(f"    [TIME] {elapsed:.2f} s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('--gain_correct', type=float, default=1.0)
    args = parser.parse_args()
    process_aggregation(args.input_path, args.gain_correct)

if __name__ == "__main__":
    main()