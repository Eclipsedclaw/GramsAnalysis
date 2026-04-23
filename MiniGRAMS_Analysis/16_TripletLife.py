#!/usr/bin/env python3
"""
fit_triplet_lifetime.py
Extracts events from the lower F_prompt band (Gamma), averages their raw waveforms,
and fits an exponential decay to extract the liquid argon triplet lifetime.
Versatile design for any input dataset.
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
from scipy.optimize import curve_fit

# --- Physical Channel Mapping ---
MB_MAP = {
    'MB1': {'VUV': [0, 2, 4, 6, 8, 10]},
    'MB2': {'VUV': [12, 14, 16, 18, 20, 22]},
    'MB3': {'VUV': [24, 26, 28, 30]}
}

def exp_decay(t, A, tau, C):
    """Exponential decay function for triplet light."""
    return A * np.exp(-t / tau) + C

def extract_waveforms_and_fp(filepath, p_ns, t_us, e_min, e_max, vuv_chs):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean_mV' not in f: return [], [], 2.0
            
            dset_waves = f['waveforms_mV']
            dset_base = f['baseline_mean_mV']
            res_ns = f.attrs.get('resolution_ns', 2.0)
            pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 2.0)
            
            n_events = dset_waves.shape[0]
            trig_idx = int((pre_trig_us * 1000) / res_ns)
            p_end = trig_idx + int(p_ns / res_ns)
            t_end = trig_idx + int((t_us * 1000) / res_ns)
            
            valid_fp = []
            valid_waves = []
            
            chunk_size = 100
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                # Baseline correction
                vuv_corr = np.zeros((end_i - i, len(vuv_chs), wave_chunk.shape[2]))
                for idx, ch in enumerate(vuv_chs):
                    vuv_corr[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                # DUMB SUMMING to preserve single-PE triplet tail
                vuv_summed = np.sum(vuv_corr, axis=1)
                
                vuv_q_p = np.sum(vuv_summed[:, trig_idx:p_end], axis=1) * res_ns
                vuv_q_t = np.sum(vuv_summed[:, trig_idx:t_end], axis=1) * res_ns
                
                mask = (vuv_q_t >= e_min) & (vuv_q_t < e_max) & (vuv_q_t > 0)
                
                for ev_idx in np.where(mask)[0]:
                    fp = vuv_q_p[ev_idx] / vuv_q_t[ev_idx]
                    if 0 <= fp <= 1.2:
                        valid_fp.append(fp)
                        valid_waves.append(vuv_summed[ev_idx, :])
                        
            return valid_fp, valid_waves, res_ns
    except:
        return [], [], 2.0

def process_and_fit(directory, p_ns, t_us, e_min, e_max, vuv_chs, sig_lower, n_comp, fit_start_ns, workers, max_files):
    files = sorted(glob.glob(os.path.join(directory, "*.h5")))[:max_files]
    
    all_fp = []
    all_waves = []
    global_res = 2.0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_waveforms_and_fp, f, p_ns, t_us, e_min, e_max, vuv_chs): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting Waveforms"):
            fp, waves, res = future.result()
            if fp:
                all_fp.extend(fp)
                all_waves.extend(waves)
                global_res = res
                
    if len(all_fp) < 50:
        print("Insufficient data to perform fit. Check energy bounds or file limits.")
        return

    # 1. GMM Isolation of the Lower Band
    f_vals = np.array(all_fp).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=200)
    gmm.fit(f_vals)
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    idx_low = np.argmin(means)
    
    mu_low = means[idx_low]
    sig_cut = stds[idx_low] * sig_lower
    
    # 2. Filter Waveforms
    waves_np = np.array(all_waves)
    mask_low = (f_vals.flatten() >= (mu_low - sig_cut)) & (f_vals.flatten() <= (mu_low + sig_cut))
    gamma_waves = waves_np[mask_low]
    
    n_gamma = len(gamma_waves)
    print(f"\n[INFO] Isolated {n_gamma} events in the Lower Band (u={mu_low:.3f}). Averaging...")
    
    # 3. Create Super-Pulse
    super_pulse = np.mean(gamma_waves, axis=0)
    
    # 4. Triplet Exponential Fit
    peak_idx = np.argmax(super_pulse)
    n_samples = len(super_pulse)
    time_us = (np.arange(n_samples) - peak_idx) * (global_res / 1000.0)
    
    fit_start_idx = peak_idx + int(fit_start_ns / global_res)
    fit_end_idx = peak_idx + int((t_us * 1000) / global_res)
    
    t_fit = time_us[fit_start_idx:fit_end_idx]
    y_fit = super_pulse[fit_start_idx:fit_end_idx]
    
    p0 = [y_fit[0], 1.3, 0.0]
    
    try:
        popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=p0, bounds=([0, 0.1, -50], [np.inf, 5.0, 50]))
        tau_3 = popt[1]
        tau_err = np.sqrt(np.diag(pcov))[1]
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(rf"Liquid Argon Triplet Lifetime ($\tau_3$) | MB Target: {args.mb}" + "\n" + rf"Events Averaged: {n_gamma} (Energy: {int(e_min/1000)}k - {int(e_max/1000)}k)", fontsize=16)
    
    # Linear Plot
    ax1.plot(time_us, super_pulse, 'k-', lw=1.5, label='Averaged Waveform', alpha=0.8)
    ax1.plot(t_fit, exp_decay(t_fit, *popt), 'r--', lw=2.5, label=rf'Fit Curve')
    ax1.axvline(time_us[fit_start_idx], color='blue', linestyle='--', lw=2.0, alpha=0.7, label=rf'Fit Start (+{fit_start_ns} ns)')
    
    # Display the LaTeX formatted function and results on the plot (Moved to center-right to avoid legend)
    eq_text = (r"$y(t) = A e^{-t/\tau_3} + C$" + "\n" +
               rf"$\tau_3 = {tau_3:.3f} \pm {tau_err:.3f}\ \mu$s" + "\n" +
               rf"$A = {popt[0]:.2f}$ mV" + "\n" +
               rf"$C = {popt[2]:.3f}$ mV")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.95, 0.65, eq_text, transform=ax1.transAxes, fontsize=12, 
             verticalalignment='center', horizontalalignment='right', bbox=props)

    ax1.set_xlim(-0.5, t_us + 0.5)
    ax1.set_xlabel(r"Time relative to Trigger ($\mu$s)", fontsize=12)
    ax1.set_ylabel(r"Amplitude (mV)", fontsize=12)
    ax1.set_title("Linear Scale", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    
    # Log Y Plot
    safe_y = np.clip(super_pulse, 0.01, None)
    ax2.plot(time_us, safe_y, 'k-', lw=1.5, alpha=0.8)
    ax2.plot(t_fit, exp_decay(t_fit, *popt), 'r--', lw=2.5)
    ax2.axvline(time_us[fit_start_idx], color='blue', linestyle='--', lw=2.0, alpha=0.7)
    
    ax2.set_yscale('log')
    ax2.set_xlim(-0.5, t_us + 0.5)
    ax2.set_xlabel(r"Time relative to Trigger ($\mu$s)", fontsize=12)
    ax2.set_ylabel(r"Log Amplitude (mV)", fontsize=12)
    ax2.set_title("Logarithmic Scale", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("Triplet_Lifetime_Fit.png", dpi=300)
    
    print(f"\n[SUCCESS] Triplet Lifetime Fitted: {tau_3:.3f} +/- {tau_err:.3f} us")
    print("Plot saved as Triplet_Lifetime_Fit.png")

if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Extract LAr Triplet Lifetime from Super-Pulse.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("-p", "--prompt_ns", type=float, default=64.0)
    parser.add_argument("-t", "--total_us", type=float, default=2.9)
    parser.add_argument("--e_min", type=float, default=100000)
    parser.add_argument("--e_max", type=float, default=200000)
    
    parser.add_argument("--mb", type=str, default='MB2', choices=['MB1', 'MB2', 'MB3'], help="Motherboard to target")
    parser.add_argument("-n", "--n_comp", type=int, default=2)
    parser.add_argument("-s0l", "--sig0_lower", type=float, default=1.0, help="Sigma cut multiplier for isolation")
    
    parser.add_argument("--fit_start_ns", type=float, default=200.0, help="How many ns past the trigger peak to begin the exponential fit.")
    
    parser.add_argument("-mf", "--max_files", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    target_vuv_chs = MB_MAP[args.mb]['VUV']
    
    print("--- GRAMS TRIPLET LIFETIME FITTER ---")
    process_and_fit(args.input_dir, args.prompt_ns, args.total_us, args.e_min, args.e_max, target_vuv_chs, args.sig0_lower, args.n_comp, args.fit_start_ns, args.workers, args.max_files)
    
    t_total = time.time() - t_start
    print(f"[TELEMETRY] Mission Complete in {t_total:.1f}s")