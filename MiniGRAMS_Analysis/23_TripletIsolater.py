#!/usr/bin/env python3
"""
17_Sliding_Fit_Analysis.py
Performs a highly granular sliding window analysis on the Triplet Lifetime fit start time.
Visualizes the plateau to avoid contamination from LAr intermediate components.
Now includes user-defined Fit End times and clamped baseline (C) bounds to break degeneracy.
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

def extract_waveforms(filepath, p_ns, t_us, e_min, e_max, fp_min, fp_max, target_channels):
    """Extracts and sums VUV traces for events inside the 2D selection box."""
    try:
        with h5py.File(filepath, 'r') as f:
            is_single = 'waveforms_mV' in f
            
            if is_single:
                if 'baseline_mean_mV' not in f: return []
                n_events = f['waveforms_mV'].shape[0]
                n_samples = f['waveforms_mV'].shape[2]
                res_ns = f.attrs.get('resolution_ns', 8.0)
                pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 4.0)
            else:
                if 'Board_72' not in f or 'baseline_mean' not in f['Board_72']: return []
                n_events = f['Board_72']['waveforms'].shape[0]
                n_samples = f['Board_72'].attrs['n_samples']
                res_ns = f['Board_72'].attrs.get('sampling_period_ns', 2.0)
                pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)

            if not target_channels: return []
            
            trig_idx = int((pre_trig_us * 1000) / res_ns)
            p_end = trig_idx + int(p_ns / res_ns)
            t_end = trig_idx + int((t_us * 1000) / res_ns)
            
            valid_summed_waves = []
            
            chunk_size = 200 
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                current_chunk_size = end_i - i
                
                corrected_chunk = np.zeros((current_chunk_size, len(target_channels), n_samples), dtype=np.float32)
                valid_event_mask = np.ones(current_chunk_size, dtype=bool)
                
                for idx, global_ch in enumerate(target_channels):
                    if is_single:
                        wave = f['waveforms_mV'][i:end_i, global_ch, :]
                        base = f['baseline_mean_mV'][i:end_i, global_ch]
                    else:
                        if global_ch < 32:
                            grp = f['Board_72']; local_ch = global_ch
                        elif global_ch < 96:
                            grp = f['Board_85']; local_ch = global_ch - 32
                        else:
                            grp = f['Board_75']; local_ch = global_ch - 96
                            
                        wave = grp['waveforms'][i:end_i, local_ch, :]
                        base = grp['baseline_mean'][i:end_i, local_ch]
                        
                        drop_mask = np.isnan(wave[:, 0])
                        valid_event_mask[drop_mask] = False

                    corrected_chunk[:, idx, :] = -1 * (wave - base[:, np.newaxis])

                summed_chunk = np.sum(corrected_chunk, axis=1)
                
                q_p = np.sum(summed_chunk[:, trig_idx:p_end], axis=1) * res_ns
                q_t = np.sum(summed_chunk[:, trig_idx:t_end], axis=1) * res_ns
                
                safe_q_t = np.where(q_t > 0, q_t, 1)
                f_p = np.where(q_t > 0, q_p / safe_q_t, 0)
                
                box_mask = (q_t >= e_min) & (q_t <= e_max) & (f_p >= fp_min) & (f_p <= fp_max)
                final_mask = valid_event_mask & box_mask
                
                for ev_idx in np.where(final_mask)[0]:
                    valid_summed_waves.append(summed_chunk[ev_idx, :])
                        
            return valid_summed_waves, res_ns
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return [], 2.0

def process_and_sweep(args, target_channels):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    all_waves = []
    global_res = 2.0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_waveforms, f, args.prompt_ns, args.total_us, args.e_min, args.e_max, args.fp_min, args.fp_max, target_channels): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting ER Waveforms"):
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

    print(f"\n[INFO] Isolated {n_gamma} pure ER events. Running Sliding Fit Analysis...")
    
    waves_np = np.array(all_waves)
    super_pulse = np.mean(waves_np, axis=0)
    
    peak_idx = np.argmax(super_pulse)
    n_samples = len(super_pulse)
    time_us = (np.arange(n_samples) - peak_idx) * (global_res / 1000.0)
    
    # Fix the fit end using the user-defined argument
    fit_end_idx = min(peak_idx + int((args.fit_end_us * 1000.0) / global_res), n_samples - 1)
    
    # Granular sweep from user arguments
    start_val, stop_val, step_val = args.fit_start_range
    start_times_ns = np.arange(start_val, stop_val + step_val, step_val)
    
    tau_results = []
    err_results = []
    valid_starts = []
    
    for start_ns in start_times_ns:
        fit_start_idx = peak_idx + int(start_ns / global_res)
        
        if fit_start_idx >= fit_end_idx - 10: 
            break # Not enough points left to fit
            
        t_fit = time_us[fit_start_idx:fit_end_idx]
        y_fit = super_pulse[fit_start_idx:fit_end_idx]
        
        p0 = [y_fit[0], 1.3, 0.0]
        try:
            # BOUNDS CLAMPED: C is restricted to [-2.0, 2.0] to break parameter degeneracy
            popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=p0, bounds=([0, 0.1, -2.0], [np.inf, 5.0, 2.0]), maxfev=2000)
            tau_3 = popt[1]
            tau_err = np.sqrt(np.diag(pcov))[1]
            
            # Reject wildly failed fits
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
    
    plt.title(f"Sliding Window Triplet Fit | N = {n_gamma} events\nFixed Fit End = {args.fit_end_us} $\\mu$s", fontsize=14)
    plt.xlabel("Fit Start Time (ns past peak)", fontsize=12)
    plt.ylabel(r"Measured Triplet Lifetime ($\tau_3$) [$\mu$s]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    
    if len(tau_arr) > 0:
        y_median = np.median(tau_arr)
        plt.ylim(y_median * 0.7, y_median * 1.3)
    
    save_path = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, "Sliding_Fit_Analysis_Constrained.png")
    plt.savefig(out_file, dpi=300)
    
    print(f"[SUCCESS] Sliding fit complete. Plot saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default="PSD_Validation")
    
    parser.add_argument("-p", "--prompt_ns", type=float, default=200.0)
    parser.add_argument("-t", "--total_us", type=float, default=4.0)
    parser.add_argument("--e_min", type=float, default=500000)
    parser.add_argument("--e_max", type=float, default=2000000)
    parser.add_argument("--fp_min", type=float, default=0.10)
    parser.add_argument("--fp_max", type=float, default=0.45)
    
    # User-defined sweep array
    parser.add_argument("--fit_start_range", nargs=3, type=float, default=[100.0, 1500.0, 20.0], 
                        help="Start, Stop, Step for the fit start time sweep (ns past peak)")
    # NEW: User-defined fit end
    parser.add_argument("--fit_end_us", type=float, default=5.4, 
                        help="Fixed end time for the exponential fit (us past peak)")
    
    parser.add_argument("--run_mode", type=str, default=None)
    parser.add_argument("-mf", "--max_files", type=int, default=2600)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    args.run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(args.run_mode)
    
    print("--- SLIDING FIT ANALYSIS ---")
    process_and_sweep(args, sorted(list(ch_map['VUV'])))