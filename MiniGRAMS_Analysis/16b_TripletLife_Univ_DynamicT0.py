#!/usr/bin/env python3
"""
23b_TripletIsolater_DynamicT0.py (QUARANTINED)
Extracts pure ER events using the Golden Cut Box and Dynamic T_0.
Actively ALIGNS (np.roll) every waveform to prevent prompt-peak smearing.
Averages them into a master Super Pulse and fits the Triplet Lifetime.
Outputs Linear and Log plots for visual artifact inspection.
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
                
                if (e_min <= q_t <= e_max) and (fp_min <= f_p <= fp_max):
                    # ALIGNMENT
                    shift_amount = target_peak_idx - p_idx
                    aligned_wave = np.roll(summed_waves[i], shift_amount)
                    if shift_amount > 0: aligned_wave[:shift_amount] = 0.0
                    elif shift_amount < 0: aligned_wave[shift_amount:] = 0.0
                    aligned_waves.append(aligned_wave)

            return aligned_waves, res_ns
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return [], 2.0

def process_and_fit(args, target_channels):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    all_waves = []
    global_res = 2.0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_and_align_waveforms, f, args.prompt_ns, args.total_us, args.e_min, args.e_max, args.fp_min, args.fp_max, target_channels): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting & Aligning ER Waveforms"):
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

    print(f"\n[INFO] Isolated & Aligned {n_gamma} pure ER events. Averaging...")
    
    waves_np = np.array(all_waves)
    super_pulse = np.mean(waves_np, axis=0)
    
    peak_idx = int((4.0 * 1000) / global_res)
    n_samples = len(super_pulse)
    time_us = (np.arange(n_samples) - peak_idx) * (global_res / 1000.0)
    
    fit_start_idx = peak_idx + int(args.fit_start_ns / global_res)
    fit_end_idx = min(peak_idx + int((args.fit_end_us * 1000.0) / global_res), n_samples - 1)
    
    t_fit = time_us[fit_start_idx:fit_end_idx]
    y_fit = super_pulse[fit_start_idx:fit_end_idx]
    
    p0 = [y_fit[0], 1.3, 0.0]
    
    try:
        # Clamped Bounds to prevent degeneracy
        popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=p0, bounds=([0, 0.1, -2.0], [np.inf, 5.0, 2.0]))
        tau_3 = popt[1]
        tau_err = np.sqrt(np.diag(pcov))[1]
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    box_str = rf"E $\in$ [{int(args.e_min/1000)}k, {int(args.e_max/1000)}k], $F_{{prompt}}$ $\in$ [{args.fp_min:.2f}, {args.fp_max:.2f}]"
    fig.suptitle(rf"Aligned Liquid Argon Triplet Lifetime ($\tau_3$) | Mode: {args.run_mode}" + "\n" + rf"Events Averaged: {n_gamma} | Cut: {box_str}", fontsize=15)
    
    # Linear Plot
    ax1.plot(time_us, super_pulse, 'k-', lw=1.5, label='Averaged VUV Waveform', alpha=0.8)
    ax1.plot(t_fit, exp_decay(t_fit, *popt), 'r--', lw=2.5, label=rf'Fit Curve')
    ax1.axvline(time_us[fit_start_idx], color='blue', linestyle='--', lw=2.0, alpha=0.7, label=rf'Fit Start (+{args.fit_start_ns} ns)')
    ax1.axvline(time_us[fit_end_idx], color='magenta', linestyle=':', lw=2.0, alpha=0.5, label=rf'Fit End ({args.fit_end_us} $\mu$s)')
    
    eq_text = (r"$y(t) = A e^{-t/\tau_3} + C$" + "\n" +
               rf"$\tau_3 = {tau_3:.3f} \pm {tau_err:.3f}\ \mu$s" + "\n" +
               rf"$A = {popt[0]:.2f}$ mV" + "\n" +
               rf"$C = {popt[2]:.3f}$ mV")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.95, 0.65, eq_text, transform=ax1.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='right', bbox=props)

    ax1.set_xlim(-0.5, max(8.0, args.fit_end_us + 1.0))
    ax1.set_ylim(-1.0, max(super_pulse) * 1.1)
    ax1.set_xlabel(r"Time relative to Peak ($\mu$s)", fontsize=12)
    ax1.set_ylabel(r"Amplitude (mV)", fontsize=12)
    ax1.set_title("Linear Scale", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    
    # Log Y Plot 
    safe_y = np.clip(super_pulse, 0.01, None)
    ax2.plot(time_us, safe_y, 'k-', lw=1.5, alpha=0.8)
    ax2.plot(t_fit, exp_decay(t_fit, *popt), 'r--', lw=2.5)
    ax2.axvline(time_us[fit_start_idx], color='blue', linestyle='--', lw=2.0, alpha=0.7)
    ax2.axvline(time_us[fit_end_idx], color='magenta', linestyle=':', lw=2.0, alpha=0.5)
    
    ax2.set_yscale('log')
    ax2.set_xlim(-0.5, max(8.0, args.fit_end_us + 1.0))
    ax2.set_xlabel(r"Time relative to Peak ($\mu$s)", fontsize=12)
    ax2.set_ylabel(r"Log Amplitude (mV)", fontsize=12)
    ax2.set_title("Logarithmic Scale", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    save_path = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, "Triplet_Lifetime_Fit_DynamicT0.png")
    plt.savefig(out_file, dpi=300)
    
    print(f"\n[SUCCESS] Triplet Lifetime Fitted: {tau_3:.3f} +/- {tau_err:.3f} us")
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quarantined Aligned LAr Triplet Lifetime Fitter.")
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default="PSD_Validation_DynamicT0")
    
    parser.add_argument("-p", "--prompt_ns", type=float, default=200.0)
    parser.add_argument("-t", "--total_us", type=float, default=10.0)
    
    # Pre-loaded Golden Box
    parser.add_argument("--e_min", type=float, default=120000)
    parser.add_argument("--e_max", type=float, default=1000000)
    parser.add_argument("--fp_min", type=float, default=0.10)
    parser.add_argument("--fp_max", type=float, default=0.40)
    
    parser.add_argument("--run_mode", type=str, default=None)
    
    # Fit window controls
    parser.add_argument("--fit_start_ns", type=float, default=1100.0, help="ns past the peak to start exp fit")
    parser.add_argument("--fit_end_us", type=float, default=5.00, help="Fixed end time (us past peak)")
    
    parser.add_argument("-mf", "--max_files", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    args.run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(args.run_mode)
    target_vuv_chs = sorted(list(ch_map['VUV']))
    
    print("--- QUARANTINED ALIGNED TRIPLET LIFETIME FITTER ---")
    process_and_fit(args, target_vuv_chs)