#!/usr/bin/env python3
"""
17b_Sliding_Fit_Analysis_DynamicT0.py (QUARANTINED)
Extracts pure ER events using the Dynamic T_0 selection box.
Actively ALIGNS (np.roll) every waveform to a fixed target index before averaging
to prevent prompt-peak smearing caused by hardware trigger errors.
Performs the clamped sliding exponential fit on the aligned super-pulse.
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
from scipy.optimize import curve_fit
import detector_config as dc

def exp_decay(t, A, tau, C):
    """Exponential decay function for triplet light."""
    return A * np.exp(-t / tau) + C

def extract_and_align_waveforms(filepath, p_ns, t_us, e_min, e_max, fp_min, fp_max, target_channels):
    """Extracts, filters via Dynamic T_0, and ALIGNS traces before returning."""
    try:
        with h5py.File(filepath, 'r') as f:
            is_single = 'waveforms_mV' in f
            
            if is_single:
                if 'baseline_mean_mV' not in f: return [], 2.0
                n_events = f['waveforms_mV'].shape[0]
                n_samples = f['waveforms_mV'].shape[2]
                res_ns = f.attrs.get('resolution_ns', 8.0)
            else:
                if 'Board_72' not in f or 'baseline_mean' not in f['Board_72']: return [], 2.0
                n_events = f['Board_72']['waveforms'].shape[0]
                n_samples = f['Board_72'].attrs['n_samples']
                res_ns = f['Board_72'].attrs.get('sampling_period_ns', 2.0)

            if not target_channels: return [], 2.0
            
            valid_event_mask = np.ones(n_events, dtype=bool)
            corrected_chunk = np.zeros((n_events, len(target_channels), n_samples), dtype=np.float32)

            for idx, global_ch in enumerate(target_channels):
                if is_single:
                    wave = f['waveforms_mV'][:, global_ch, :]
                    base = f['baseline_mean_mV'][:, global_ch]
                else:
                    if global_ch < 32: grp = f['Board_72']; local_ch = global_ch
                    elif global_ch < 96: grp = f['Board_85']; local_ch = global_ch - 32
                    else: grp = f['Board_75']; local_ch = global_ch - 96
                    wave = grp['waveforms'][:, local_ch, :]
                    base = grp['baseline_mean'][:, local_ch]

                drop_mask = np.isnan(wave[:, 0])
                valid_event_mask[drop_mask] = False
                corrected_chunk[:, idx, :] = -1 * (wave - base[:, np.newaxis])

            # Filter out dropped/corrupt events first
            valid_chunk = corrected_chunk[valid_event_mask]
            summed_waves = np.sum(valid_chunk, axis=1)
            
            # --- DYNAMIC T_0 CALCULATION ---
            peak_indices = np.argmax(summed_waves, axis=1)
            p_samples = int(p_ns / res_ns)
            t_samples = int((t_us * 1000) / res_ns)
            
            aligned_waves = []
            target_peak_idx = int((4.0 * 1000) / res_ns) # Force all peaks to 4.0 us
            
            for i in range(len(valid_chunk)):
                p_idx = peak_indices[i]
                p_end = min(p_idx + p_samples, n_samples)
                t_end = min(p_idx + t_samples, n_samples)
                
                q_p = np.sum(summed_waves[i, p_idx:p_end]) * res_ns
                q_t = np.sum(summed_waves[i, p_idx:t_end]) * res_ns
                
                f_p = q_p / q_t if q_t > 0 else 0
                
                # Check if event falls inside the Golden Cut Box
                if (e_min <= q_t <= e_max) and (fp_min <= f_p <= fp_max):
                    # WAVEFORM ALIGNMENT: Shift the array so the peak sits at target_peak_idx
                    shift_amount = target_peak_idx - p_idx
                    aligned_wave = np.roll(summed_waves[i], shift_amount)
                    
                    # Zero out the wrap-around artifact to keep the tail pristine
                    if shift_amount > 0: aligned_wave[:shift_amount] = 0.0
                    elif shift_amount < 0: aligned_wave[shift_amount:] = 0.0
                        
                    aligned_waves.append(aligned_wave)

            return aligned_waves, res_ns
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return [], 2.0

def process_and_sweep(args, target_channels):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    all_waves = []
    global_res = 2.0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_and_align_waveforms, f, args.prompt_ns, args.total_us, 
                                   args.e_min, args.e_max, args.fp_min, args.fp_max, target_channels): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting & Aligning ER Waves"):
            res = future.result()
            if res:
                waves, res_ns = res
                if waves:
                    all_waves.extend(waves)
                    global_res = res_ns
                
    n_gamma = len(all_waves)
    if n_gamma < 50:
        print(f"Insufficient data (N={n_gamma}) to perform fit.")
        return

    print(f"\n[INFO] Isolated & Aligned {n_gamma} pure ER events. Running Sliding Fit Analysis...")
    
    waves_np = np.array(all_waves)
    super_pulse = np.mean(waves_np, axis=0)
    
    # Because we aligned them, the peak is guaranteed to be here:
    peak_idx = int((4.0 * 1000) / global_res)
    n_samples = len(super_pulse)
    time_us = (np.arange(n_samples) - peak_idx) * (global_res / 1000.0)
    
    fit_end_idx = min(peak_idx + int((args.fit_end_us * 1000.0) / global_res), n_samples - 1)
    
    start_val, stop_val, step_val = args.fit_start_range
    start_times_ns = np.arange(start_val, stop_val + step_val, step_val)
    
    tau_results, err_results, valid_starts = [], [], []
    
    for start_ns in start_times_ns:
        fit_start_idx = peak_idx + int(start_ns / global_res)
        
        if fit_start_idx >= fit_end_idx - 10: break 
            
        t_fit = time_us[fit_start_idx:fit_end_idx]
        y_fit = super_pulse[fit_start_idx:fit_end_idx]
        
        p0 = [y_fit[0], 1.3, 0.0]
        try:
            # BOUNDS CLAMPED: Break degeneracy
            popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=p0, bounds=([0, 0.1, -2.0], [np.inf, 5.0, 2.0]), maxfev=2000)
            tau_3 = popt[1]
            tau_err = np.sqrt(np.diag(pcov))[1]
            
            if tau_err < 1.0:
                tau_results.append(tau_3)
                err_results.append(tau_err)
                valid_starts.append(start_ns)
        except:
            pass

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    tau_arr = np.array(tau_results)
    err_arr = np.array(err_results)
    starts_arr = np.array(valid_starts)
    
    plt.plot(starts_arr, tau_arr, 'k-', marker='o', markersize=4, lw=1.5, label=r'Fitted $\tau_3$')
    plt.fill_between(starts_arr, tau_arr - err_arr, tau_arr + err_arr, color='red', alpha=0.3, label=r'$\pm 1\sigma$ Error')
    
    plt.title(f"Sliding Triplet Fit (Aligned Dynamic $T_0$) | N = {n_gamma} events\nFixed Fit End = {args.fit_end_us} $\\mu$s", fontsize=14)
    plt.xlabel("Fit Start Time (ns past aligned peak)", fontsize=12)
    plt.ylabel(r"Measured Triplet Lifetime ($\tau_3$) [$\mu$s]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    
    if len(tau_arr) > 0:
        y_median = np.median(tau_arr)
        plt.ylim(y_median * 0.7, y_median * 1.3)
    
    save_path = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, "Sliding_Fit_Analysis_DynamicT0.png")
    plt.savefig(out_file, dpi=300)
    
    print(f"[SUCCESS] Sliding fit complete. Plot saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default="PSD_Validation_DynamicT0")
    
    # Optimal prompt/total integration from your heatmap
    parser.add_argument("-p", "--prompt_ns", type=float, default=200.0)
    parser.add_argument("-t", "--total_us", type=float, default=10.0)
    
    # YOUR GOLDEN CUT BOX
    parser.add_argument("--e_min", type=float, default=120000)
    parser.add_argument("--e_max", type=float, default=1000000)
    parser.add_argument("--fp_min", type=float, default=0.10)
    parser.add_argument("--fp_max", type=float, default=0.40)
    
    parser.add_argument("--fit_start_range", nargs=3, type=float, default=[100.0, 1500.0, 25.0])
    parser.add_argument("--fit_end_us", type=float, default=5.45) 
    
    parser.add_argument("--run_mode", type=str, default=None)
    parser.add_argument("-mf", "--max_files", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    args.run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(args.run_mode)
    
    print("--- QUARANTINED SLIDING FIT ANALYSIS (ALIGN & AVERAGE) ---")
    process_and_sweep(args, sorted(list(ch_map['VUV'])))