#!/usr/bin/env python3
"""
16_TripletLife.py (Universal Edition - Clamped Fit)
Extracts pure ER events using a strict 2D F_prompt vs Energy selection box, 
averages their raw waveforms, and fits the liquid argon triplet lifetime.
Includes user-defined fit boundaries and clamped baseline to break degeneracy.
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
    """Extracts and sums VUV traces only if the event falls inside the 2D selection box."""
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

def process_and_fit(args, target_channels):
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
        print(f"Insufficient data (N={n_gamma}) to perform fit. Check your 2D box limits or file count.")
        return

    print(f"\n[INFO] Isolated {n_gamma} pure ER events inside selection box. Averaging...")
    
    waves_np = np.array(all_waves)
    super_pulse = np.mean(waves_np, axis=0)
    
    peak_idx = np.argmax(super_pulse)
    n_samples = len(super_pulse)
    time_us = (np.arange(n_samples) - peak_idx) * (global_res / 1000.0)
    
    fit_start_idx = peak_idx + int(args.fit_start_ns / global_res)
    # Using the new user-defined fit_end_us
    fit_end_idx = min(peak_idx + int((args.fit_end_us * 1000.0) / global_res), n_samples - 1)
    
    t_fit = time_us[fit_start_idx:fit_end_idx]
    y_fit = super_pulse[fit_start_idx:fit_end_idx]
    
    p0 = [y_fit[0], 1.3, 0.0]
    
    try:
        # BOUNDS CLAMPED: C is restricted to [-2.0, 2.0]
        popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=p0, bounds=([0, 0.1, -2.0], [np.inf, 5.0, 2.0]))
        tau_3 = popt[1]
        tau_err = np.sqrt(np.diag(pcov))[1]
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    box_str = rf"E $\in$ [{int(args.e_min/1000)}k, {int(args.e_max/1000)}k], $F_{{prompt}}$ $\in$ [{args.fp_min:.2f}, {args.fp_max:.2f}]"
    fig.suptitle(rf"Liquid Argon Triplet Lifetime ($\tau_3$) | Mode: {args.run_mode}" + "\n" + rf"Events Averaged: {n_gamma} | Cut: {box_str}", fontsize=15)
    
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

    # Clean axes limits
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
    out_file = os.path.join(save_path, "Triplet_Lifetime_Fit_Universal.png")
    plt.savefig(out_file, dpi=300)
    
    print(f"\n[SUCCESS] Triplet Lifetime Fitted: {tau_3:.3f} +/- {tau_err:.3f} us")
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Extract LAr Triplet Lifetime using strict 2D F_prompt cuts.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("-o", "--output_dir", default="PSD_Validation", help="Output directory")
    
    parser.add_argument("-p", "--prompt_ns", type=float, default=200.0)
    parser.add_argument("-t", "--total_us", type=float, default=4.0)
    
    parser.add_argument("--e_min", type=float, default=500000)
    parser.add_argument("--e_max", type=float, default=2000000)
    parser.add_argument("--fp_min", type=float, default=0.10)
    parser.add_argument("--fp_max", type=float, default=0.45)
    
    parser.add_argument("--run_mode", type=str, default=None, help="Force run mode (e.g. MICROG_PURITY_STUDY)")
    parser.add_argument("--fit_start_ns", type=float, default=1000.0, help="ns past the trigger peak to start exp fit")
    
    parser.add_argument("--fit_end_us", type=float, default=5.45, help="Fixed end time for the exponential fit (us past peak)")
    
    parser.add_argument("-mf", "--max_files", type=int, default=2600)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    
    args.run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(args.run_mode)
    target_vuv_chs = sorted(list(ch_map['VUV']))
    
    print("--- UNIVERSAL TRIPLET LIFETIME FITTER ---")
    print(f"Targeting VUV Channels: {target_vuv_chs}")
    print(f"Applying Box Cut: Energy {args.e_min/1000}k-{args.e_max/1000}k | F_prompt {args.fp_min}-{args.fp_max}")
    print(f"Fitting Window: +{args.fit_start_ns} ns to {args.fit_end_us} us")
    
    process_and_fit(args, target_vuv_chs)
    
    t_total = time.time() - t_start
    print(f"[TELEMETRY] Mission Complete in {t_total:.1f}s")