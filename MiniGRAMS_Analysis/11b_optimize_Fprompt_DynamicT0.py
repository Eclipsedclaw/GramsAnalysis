#!/usr/bin/env python3
"""
11b_Optimize_Fprompt_DynamicT0.py (QUARANTINED)
PSD Grid Search specifically designed to salvage wandering pulses.
Uses a dynamic Software Peak-Finder (np.argmax) to redefine T_0 per event,
bypassing hardware trigger misalignments (e.g., Rising-Edge mistakes).
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
import detector_config as dc 

def calculate_fom_and_params(f_prompt_values, n_components):
    data = f_prompt_values.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
        gmm.fit(data)
    except:
        return 0.0, None, None, None

    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    stds = np.sqrt(covariances)
    weights = gmm.weights_.flatten()
    
    sorted_indices = np.argsort(means)
    idx_ER = sorted_indices[0] 
    idx_NR = sorted_indices[-1] 
    
    mu_ER, sig_ER = means[idx_ER], stds[idx_ER]
    mu_NR, sig_NR = means[idx_NR], stds[idx_NR]
    
    fom = abs(mu_ER - mu_NR) / (sig_ER + sig_NR)
    return fom, means, stds, weights

def process_file_fast(filepath, target_channels, ref_window_us=10.0):
    try:
        with h5py.File(filepath, 'r') as f:
            is_single = 'waveforms_mV' in f
            if is_single:
                if 'baseline_mean_mV' not in f: return None, None, None, None
                n_events = f['waveforms_mV'].shape[0]
                n_samples = f['waveforms_mV'].shape[2]
                res_ns = f.attrs.get('resolution_ns', 8.0)
            else:
                if 'Board_72' not in f or 'baseline_mean' not in f['Board_72']: return None, None, None, None
                n_events = f['Board_72']['waveforms'].shape[0]
                n_samples = f['Board_72'].attrs['n_samples']
                res_ns = f['Board_72'].attrs.get('sampling_period_ns', 2.0)

            if not target_channels: return None, None, None, None
            
            valid_event_mask = np.ones(n_events, dtype=bool)
            corrected_sum = np.zeros((n_events, n_samples), dtype=np.float32)

            for global_ch in target_channels:
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
                corrected_sum += -1 * (wave - base[:, np.newaxis])

            valid_waves = corrected_sum[valid_event_mask]
            
            # --- DYNAMIC T_0 PEAK FINDER ---
            peak_indices = np.argmax(valid_waves, axis=1)
            q_totals = np.zeros(len(valid_waves))
            
            for i in range(len(valid_waves)):
                p_idx = peak_indices[i]
                t_end = min(p_idx + int((ref_window_us * 1000) / res_ns), n_samples)
                q_totals[i] = np.sum(valid_waves[i, p_idx:t_end]) * res_ns
            
            return valid_waves, q_totals, res_ns, peak_indices
            
    except Exception as e:
        return None, None, None, None

def plot_global_spectrum(all_ens, save_dir, ref_window_us, bin_edges):
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(all_ens, bins=150, color='royalblue', alpha=0.7, log=True)
    for edge in bin_edges:
        plt.axvline(edge, color='red', linestyle='--', alpha=0.8, lw=1.5)
    plt.title(f"Global Energy Spectrum (Dynamic $T_0$ | $t_{{tot}}$={ref_window_us} $\\mu$s) | N={len(all_ens)}")
    plt.xlabel("Integrated Charge (mV*ns)")
    plt.ylabel("Counts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Global_Energy_Spectrum_DynamicT0.png"), dpi=300)
    plt.close()

def run_grid_search(valid_data, peak_indices, res_ns, e_min, e_max, args, slice_name, ens):
    mask = (ens >= e_min) & (ens <= e_max)
    sub_data = valid_data[mask]
    sub_peaks = peak_indices[mask]
    
    if len(sub_data) < 50:
        print(f"  -> Skipping {slice_name}: Insufficient stats ({len(sub_data)} events).")
        return None, None, None, None
        
    p_starts = np.arange(args.prompt_range[0], args.prompt_range[1] + args.prompt_range[2], args.prompt_range[2])
    t_starts = np.arange(args.total_range[0], args.total_range[1] + args.total_range[2], args.total_range[2])
    matrix = np.zeros((len(p_starts), len(t_starts)))
    
    n_samples = sub_data.shape[1]
    
    for p_idx, p_ns in enumerate(p_starts):
        p_samples = int(p_ns / res_ns)
        q_p = np.array([np.sum(w[p : min(p + p_samples, n_samples)]) for w, p in zip(sub_data, sub_peaks)]) * res_ns
        
        for t_idx, t_us in enumerate(t_starts):
            t_samples = int((t_us * 1000) / res_ns)
            q_t = np.array([np.sum(w[p : min(p + t_samples, n_samples)]) for w, p in zip(sub_data, sub_peaks)]) * res_ns
            
            safe_q_t = np.where(q_t > 0, q_t, 1)
            f_p = np.where(q_t > 0, q_p / safe_q_t, 0)
            f_p_valid = f_p[(f_p >= 0) & (f_p <= 1.2)]
            
            if len(f_p_valid) > 50:
                fom, _, _, _ = calculate_fom_and_params(f_p_valid, args.n_components)
                matrix[p_idx, t_idx] = fom
                
    return p_starts, t_starts, matrix, (sub_data, sub_peaks)

def generate_plots(p_ax, t_ax, matrix, data_tuple, res_ns, save_dir, tag, n_components):
    sub_data, sub_peaks = data_tuple
    
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(t_ax, p_ax)
    c = plt.pcolormesh(X, Y, matrix, cmap='viridis', shading='auto')
    plt.colorbar(c, label='GMM Figure of Merit (FoM)')
    
    max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    best_p = p_ax[max_idx[0]]
    best_t = t_ax[max_idx[1]]
    best_fom = matrix[max_idx]
    
    plt.plot(best_t, best_p, 'r*', markersize=15, markeredgecolor='black', 
             label=f'Optimal: $t_p$={best_p}ns, $t_{{tot}}$={best_t:.1f}$\\mu$s (FoM={best_fom:.2f})')
    plt.title(f"PSD Landscape (Dynamic $T_0$) | {tag}")
    plt.xlabel(r"Total Window $t_{tot}$ ($\mu$s)")
    plt.ylabel(r"Prompt Window $t_p$ (ns)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Heatmap_{tag}.png"), dpi=300)
    plt.close()
    
    n_samples = sub_data.shape[1]
    p_samples = int(best_p / res_ns)
    t_samples = int((best_t * 1000) / res_ns)
    
    q_p = np.array([np.sum(w[p : min(p + p_samples, n_samples)]) for w, p in zip(sub_data, sub_peaks)]) * res_ns
    q_t = np.array([np.sum(w[p : min(p + t_samples, n_samples)]) for w, p in zip(sub_data, sub_peaks)]) * res_ns
    
    safe_q_t = np.where(q_t > 0, q_t, 1)
    f_vals = np.where(q_t > 0, q_p / safe_q_t, 0)
    f_vals = f_vals[(f_vals >= 0) & (f_vals <= 1.2)]
    
    fom, means, stds, weights = calculate_fom_and_params(f_vals, n_components)
    
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(f_vals, bins=100, alpha=0.5, color='gray', label=f'Data (N={len(f_vals)})')
    bin_width = bins[1] - bins[0]
    
    x = np.linspace(0, 1.2, 1000)
    total_pdf = np.zeros_like(x)
    colors = ['blue', 'red', 'green', 'purple']
    
    if means is not None:
        for i in range(len(means)):
            mu, sig, w = means[i], stds[i], weights[i]
            pdf = w * (1/(sig * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sig)**2)
            scaled_pdf = pdf * len(f_vals) * bin_width
            total_pdf += scaled_pdf
            plt.plot(x, scaled_pdf, '--', color=colors[i%len(colors)], label=f"Comp {i} (u={mu:.2f})")
        plt.plot(x, total_pdf, 'k-', lw=1.5, label=f"Total Fit (FoM={fom:.2f})")
        
    plt.title(f"Best GMM Fit (Dynamic $T_0$) | {tag} | N = {len(f_vals)}\n$t_p$={best_p}ns, $t_{{tot}}$={best_t:.1f}$\\mu$s")
    plt.xlabel("F_prompt")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"GMM_Fit_{tag}.png"))
    plt.close()
    
    return best_fom

def process_and_optimize(args, target_channels):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    
    all_valid_waves = []
    all_ens = []
    all_peaks = []
    res_ns = 2.0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_fast, f, target_channels, args.ref_window_us): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting Dynamic Waveforms", unit="file"):
            res = future.result()
            if res[0] is not None:
                waves, q_t, r_ns, peaks = res
                all_valid_waves.append(waves)
                all_ens.append(q_t)
                all_peaks.append(peaks)
                res_ns = r_ns
                
    if not all_valid_waves: return
        
    full_data = np.vstack(all_valid_waves)
    all_ens = np.concatenate(all_ens)
    full_peaks = np.concatenate(all_peaks)
    
    SAVE_ROOT = os.path.join(args.input_dir, args.output_dir)
    if not os.path.exists(SAVE_ROOT): os.makedirs(SAVE_ROOT)
    
    plot_global_spectrum(all_ens, SAVE_ROOT, args.ref_window_us, args.bin_edges)
    
    if args.mode == 'spectrum':
        print(f"\n[SUCCESS] 'spectrum' mode finished. Halt active.")
        return
        
    for i in range(len(args.bin_edges)-1):
        e_min = args.bin_edges[i]
        e_max = args.bin_edges[i+1]
        slice_name = f"E_{int(e_min/1000)}k_{int(e_max/1000)}k"
        print(f"Processing {slice_name} ...")
        
        p_ax, t_ax, matrix, valid_tuple = run_grid_search(
            full_data, full_peaks, res_ns, e_min, e_max, args, slice_name, all_ens
        )
        
        if matrix is not None and np.max(matrix) > 0:
            slice_dir = os.path.join(SAVE_ROOT, slice_name)
            if not os.path.exists(slice_dir): os.makedirs(slice_dir)
            best_fom = generate_plots(p_ax, t_ax, matrix, valid_tuple, res_ns, slice_dir, slice_name, args.n_components)
            print(f"  -> Done. Max FoM = {best_fom:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default="PSD_Optimization_DynamicT0")
    parser.add_argument("--run_mode", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=['spectrum', 'optimize'], default='optimize')
    parser.add_argument("--prompt_range", nargs=3, type=float, default=[40.0, 200.0, 10.0])
    parser.add_argument("--total_range", nargs=3, type=float, default=[1.0, 10.0, 0.5])
    parser.add_argument("--ref_window_us", type=float, default=10.0)
    parser.add_argument("--bin_edges", nargs='+', type=float, default=[80000, 120000, 250000])
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("-mf", "--max_files", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    args.run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(args.run_mode)
    target_vuv_chs = sorted(list(ch_map['VUV']))
    
    print("--- QUARANTINED PSD OPTIMIZER (DYNAMIC T_0) ---")
    process_and_optimize(args, target_vuv_chs)