#!/usr/bin/env python3
"""
compare_gmm_fixed.py
Generates a 2x6 grid of Rate-Normalized 1D GMM fits.
Top Row: acq0 | Bottom Row: acq1 
Now includes FoM and Gamma/Beta Peak Ratio in the legend.
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

MB2_VUV_CHS = [12, 14, 16, 18, 20, 22]

def get_live_time(file_list):
    if len(file_list) == 0: return 1.0
    try:
        with h5py.File(file_list[0], 'r') as f_start:
            start_tick = f_start['timestamps'][0]
        with h5py.File(file_list[-1], 'r') as f_end:
            end_tick = f_end['timestamps'][-1]
            
        clock_freq_hz = 125_000_000.0
        duration_sec = (end_tick - start_tick) / clock_freq_hz
        if duration_sec <= 0: return 1.0
        return duration_sec
    except Exception as e:
        t_start = os.path.getmtime(file_list[0])
        t_end = os.path.getmtime(file_list[-1])
        duration_sec = abs(t_end - t_start)
        if duration_sec < 1.0: duration_sec = len(file_list) * 50.0 
        return duration_sec

def extract_channel_fprompt(filepath, p_ns, t_us, e_min, e_max):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean_mV' not in f: return None
            
            dset_waves = f['waveforms_mV']
            dset_base = f['baseline_mean_mV']
            res_ns = f.attrs.get('resolution_ns', 2.0)
            pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 2.0)
            
            n_events = dset_waves.shape[0]
            trig_idx = int((pre_trig_us * 1000) / res_ns)
            p_end = trig_idx + int(p_ns / res_ns)
            t_end = trig_idx + int((t_us * 1000) / res_ns)
            
            ch_fp_data = {ch: [] for ch in MB2_VUV_CHS}
            chunk_size = 200
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                corrected = np.zeros((end_i - i, len(MB2_VUV_CHS), wave_chunk.shape[2]))
                for idx, ch in enumerate(MB2_VUV_CHS):
                    corrected[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                ch_q_p = np.sum(corrected[:, :, trig_idx:p_end], axis=2) * res_ns
                ch_q_t = np.sum(corrected[:, :, trig_idx:t_end], axis=2) * res_ns
                
                for idx, ch in enumerate(MB2_VUV_CHS):
                    e_vals = ch_q_t[:, idx]
                    p_vals = ch_q_p[:, idx]
                    
                    mask = (e_vals >= e_min) & (e_vals < e_max) & (e_vals > 0)
                    if np.sum(mask) > 0:
                        fp_vals = p_vals[mask] / e_vals[mask]
                        valid_fp = fp_vals[(fp_vals >= 0) & (fp_vals <= 1.2)]
                        ch_fp_data[ch].extend(valid_fp)
            return ch_fp_data
    except:
        return None

def fit_and_plot_gmm(ax, data, title, duration_sec, n_comp=2):
    if len(data) < 50:
        ax.text(0.5, 0.5, 'Low Stats', ha='center', va='center')
        ax.set_title(title, fontsize=10)
        return
        
    f_vals = np.array(data).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=200)
    try:
        gmm.fit(f_vals)
    except:
        return
        
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    gmm_weights = gmm.weights_.flatten()
    
    # Calculate FoM
    idx_ER = np.argmin(means) # Lower u (Gamma/Triplet)
    idx_NR = np.argmax(means) # Higher u (Beta/Alpha/Singlet)
    fwhm_ER = 2.355 * stds[idx_ER]
    fwhm_NR = 2.355 * stds[idx_NR]
    fom = np.abs(means[idx_NR] - means[idx_ER]) / (fwhm_ER + fwhm_NR) if (fwhm_ER + fwhm_NR) > 0 else 0
    
    # Calculate Rates
    rate_ER = (gmm_weights[idx_ER] * len(data)) / duration_sec
    rate_NR = (gmm_weights[idx_NR] * len(data)) / duration_sec
    
    # Calculate True Visual Peak Heights (Amplitude)
    amp_lower = (gmm_weights[idx_ER] / (stds[idx_ER] * np.sqrt(2*np.pi))) 
    amp_upper = (gmm_weights[idx_NR] / (stds[idx_NR] * np.sqrt(2*np.pi))) 
    lu_height_ratio = amp_lower / amp_upper if amp_upper > 0 else 0.0

    # Plot Rate Histogram
    n_bins = 80
    bin_edges = np.linspace(0, 1.2, n_bins + 1)
    bin_width = 1.2 / n_bins
    weights_hist = np.ones_like(f_vals) / duration_sec
    ax.hist(f_vals, bins=bin_edges, weights=weights_hist, alpha=0.5, color='gray')
    
    # Plot Scaled GMM Fit
    x = np.linspace(0, 1.2, 500)
    total_pdf = np.zeros_like(x)
    colors = ['blue', 'red']
    scale_factor = (len(data) / duration_sec) * bin_width
    
    for i in range(len(means)):
        mu, sig, w = means[i], stds[i], gmm_weights[i]
        pdf = w * (1/(sig * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sig)**2)
        scaled_pdf = pdf * scale_factor
        total_pdf += scaled_pdf
        
        rate_hz = (w * len(data)) / duration_sec
        # Force blue for Gamma band, red for Beta band
        c = 'blue' if i == idx_ER else 'red'
        ax.plot(x, scaled_pdf, '--', color=c, lw=1.5, label=f"u={mu:.2f} ({rate_hz:.2f} Hz)")
        
    # Legend with FoM and Ratio
    ax.plot(x, total_pdf, 'k-', lw=1.5, label=f"Fit (FoM={fom:.2f})\nL/U Height={lu_height_ratio:.2f}")
    ax.set_title(f"{title}\nN={len(data)}", fontsize=10)
    ax.legend(fontsize=7, loc='upper right')

def process_directory(directory, p_ns, t_us, e_min, e_max, workers, max_files):
    files = sorted(glob.glob(os.path.join(directory, "*.h5")))[:max_files]
    dur = get_live_time(files)
    
    agg_data = {ch: [] for ch in MB2_VUV_CHS}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_channel_fprompt, f, p_ns, t_us, e_min, e_max): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc=f"Scanning {os.path.basename(directory)}"):
            res = future.result()
            if res:
                for ch in MB2_VUV_CHS:
                    agg_data[ch].extend(res[ch])
    return agg_data, dur

if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Generate Rate-Normalized 2x6 GMM comparison grid.")
    parser.add_argument("-d0", "--dir_acq0", required=True)
    parser.add_argument("-d1", "--dir_acq1", required=True)
    parser.add_argument("-p", "--prompt_ns", type=float, default=64.0, help="Prompt window (ns)")
    parser.add_argument("-t", "--total_us", type=float, default=2.9, help="Total window (us)")
    parser.add_argument("--e_min", type=float, default=100000, help="Min Energy (mV*ns)")
    parser.add_argument("--e_max", type=float, default=200000, help="Max Energy (mV*ns)")
    parser.add_argument("-mf", "--max_files", type=int, default=480)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    
    print("--- GRAMS RATE-NORMALIZED GMM COMPARATOR ---")
    data_acq0, dur0 = process_directory(args.dir_acq0, args.prompt_ns, args.total_us, args.e_min, args.e_max, args.workers, args.max_files)
    data_acq1, dur1 = process_directory(args.dir_acq1, args.prompt_ns, args.total_us, args.e_min, args.e_max, args.workers, args.max_files)

    print(f"\n[TELEMETRY] ACQ0 Live-Time: {dur0:.2f}s | ACQ1 Live-Time: {dur1:.2f}s")

    fig, axes = plt.subplots(2, 6, figsize=(26, 8), sharex=True, sharey=True)
    fig.suptitle(f"MB2 VUV GMM Fits | Energy: {int(args.e_min/1000)}k - {int(args.e_max/1000)}k\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)", fontsize=16)

    for i, ch in enumerate(MB2_VUV_CHS):
        fit_and_plot_gmm(axes[0, i], data_acq0[ch], f"ACQ0 (Source) | CH {ch}", dur0)
        fit_and_plot_gmm(axes[1, i], data_acq1[ch], f"ACQ1 (Bkg) | CH {ch}", dur1)
        
        if i == 0:
            axes[0, i].set_ylabel("Rate (Hz / bin)")
            axes[1, i].set_ylabel("Rate (Hz / bin)")
        axes[1, i].set_xlabel("F_prompt")

    # Relax spacing so title doesn't collide with subplot headers
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, wspace=0.1)
    
    plt.savefig("GMM_Comparison_2x6.png", dpi=300, bbox_inches='tight')
    
    t_total = time.time() - t_start
    print(f"[TELEMETRY] Mission Complete. Grid saved as GMM_Comparison_2x6.png in {t_total:.1f}s")