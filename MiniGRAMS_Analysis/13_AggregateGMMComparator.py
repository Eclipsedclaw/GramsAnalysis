#!/usr/bin/env python3
"""
plot_aggregate_comparison.py
Generates a side-by-side 1D GMM fit of the TRUE SMART-SUMMED MB2 VUV data.
Y-Axis is normalized to True Rate (Hz / bin) using CAEN hardware timestamps.
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
    """
    Dynamically calculates total acquisition time using CAEN 125 MHz TTT timestamps.
    """
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
        print(f"Warning: Failed to extract CAEN timestamps. Falling back to OS time.")
        t_start = os.path.getmtime(file_list[0])
        t_end = os.path.getmtime(file_list[-1])
        duration_sec = abs(t_end - t_start)
        if duration_sec < 1.0: duration_sec = len(file_list) * 50.0 
        return duration_sec

def extract_valid_fprompt(filepath, p_ns, t_us, e_min, e_max, smart_sum_thresh):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean_mV' not in f: return []
            
            dset_waves = f['waveforms_mV']
            dset_base = f['baseline_mean_mV']
            res_ns = f.attrs.get('resolution_ns', 2.0)
            pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 2.0)
            
            n_events = dset_waves.shape[0]
            trig_idx = int((pre_trig_us * 1000) / res_ns)
            p_end = trig_idx + int(p_ns / res_ns)
            t_end = trig_idx + int((t_us * 1000) / res_ns)
            
            valid_fp_list = []
            
            chunk_size = 200
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                corrected = np.zeros((end_i - i, len(MB2_VUV_CHS), wave_chunk.shape[2]))
                for idx, ch in enumerate(MB2_VUV_CHS):
                    corrected[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                # --- PROPER SMART SUM LOGIC ---
                peaks = np.max(corrected, axis=2)
                mask = (peaks > smart_sum_thresh).astype(np.float32)
                masked_chunk = corrected * mask[:, :, np.newaxis]
                
                # Sum the waveforms across the channels for each event
                summed_waves = np.sum(masked_chunk, axis=1)
                
                # Integrate the summed waveform
                q_p = np.sum(summed_waves[:, trig_idx:p_end], axis=1) * res_ns
                q_t = np.sum(summed_waves[:, trig_idx:t_end], axis=1) * res_ns
                
                # Filter by total event energy
                valid_mask = (q_t >= e_min) & (q_t < e_max) & (q_t > 0)
                
                if np.sum(valid_mask) > 0:
                    fp_vals = q_p[valid_mask] / q_t[valid_mask]
                    valid_fp = fp_vals[(fp_vals >= 0) & (fp_vals <= 1.2)]
                    valid_fp_list.extend(valid_fp)
                        
            return valid_fp_list
    except:
        return []

def fit_and_plot(ax, data, title, duration_sec):
    if len(data) < 50: 
        ax.set_title(f"{title}\nLow Stats (N={len(data)})")
        return
        
    f_vals = np.array(data).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(f_vals)
        
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    gmm_weights = gmm.weights_.flatten()
    
    # Rate Histogram setup
    n_bins = 100
    bin_edges = np.linspace(0, 1.2, n_bins + 1)
    bin_width = 1.2 / n_bins
    weights_hist = np.ones_like(f_vals) / duration_sec
    
    ax.hist(f_vals, bins=bin_edges, weights=weights_hist, alpha=0.5, color='gray', label=f'Data (N={len(data)})')
    
    # GMM fit scaled to rate
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
        ax.plot(x, scaled_pdf, '--', color=colors[i%2], lw=1.5, label=f"u={mu:.2f} ({rate_hz:.2f} Hz)")
        
    ax.plot(x, total_pdf, 'k-', lw=1.5, label="Total Fit")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("F_prompt")
    ax.set_ylabel("Rate (Hz / bin)")
    ax.legend(fontsize=10)

def process_directory(directory, p_ns, t_us, e_min, e_max, smart_sum_thresh, workers, max_files):
    files = sorted(glob.glob(os.path.join(directory, "*.h5")))[:max_files]
    dur = get_live_time(files)
    
    agg_data = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_valid_fprompt, f, p_ns, t_us, e_min, e_max, smart_sum_thresh): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc=f"Scanning {os.path.basename(directory)}"):
            res = future.result()
            if res:
                agg_data.extend(res)
    return np.array(agg_data), dur

if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Compare Smart-Summed Aggregate GMMs (Rate Normalized).")
    parser.add_argument("--dir_acq0", required=True)
    parser.add_argument("--dir_acq1", required=True)
    parser.add_argument("-p", "--prompt_ns", type=float, default=64.0)
    parser.add_argument("-t", "--total_us", type=float, default=2.9)
    parser.add_argument("--e_min", type=float, default=100000)
    parser.add_argument("--e_max", type=float, default=200000)
    parser.add_argument("--smart_sum_thresh", type=float, default=6.0, help="Threshold in mV for a channel to be included.")
    parser.add_argument("--max_files", type=int, default=480)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    
    print("--- GRAMS RATE-NORMALIZED SMART-SUM GMM ---")
    data_acq0, dur0 = process_directory(args.dir_acq0, args.prompt_ns, args.total_us, args.e_min, args.e_max, args.smart_sum_thresh, args.workers, args.max_files)
    data_acq1, dur1 = process_directory(args.dir_acq1, args.prompt_ns, args.total_us, args.e_min, args.e_max, args.smart_sum_thresh, args.workers, args.max_files)

    print(f"\n[TELEMETRY] ACQ0 Live-Time: {dur0:.2f}s | ACQ1 Live-Time: {dur1:.2f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
    fig.suptitle(f"Smart-Summed MB2 VUV F_prompt | Energy: {int(args.e_min/1000)}k - {int(args.e_max/1000)}k\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)", fontsize=16)

    fit_and_plot(ax1, data_acq0, "ACQ0 (Cesium + Bkg)", dur0)
    fit_and_plot(ax2, data_acq1, "ACQ1 (Bkg Only)", dur1)

    plt.tight_layout()
    plt.savefig("Aggregate_GMM_Comparison.png", dpi=300)
    
    t_total = time.time() - t_start
    print(f"[TELEMETRY] Mission Complete. Saved Aggregate_GMM_Comparison.png in {t_total:.1f}s")