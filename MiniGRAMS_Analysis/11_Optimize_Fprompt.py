#!/usr/bin/env python3
"""
optimize_fprompt.py
Command-line tool for PSD Grid Search Optimization via GMM fitting.
Includes Multiprocessing, MB routing, Telemetry, and Real-Time Progress Bars.
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
from tqdm import tqdm  # <-- NEW IMPORT
from sklearn.mixture import GaussianMixture
import detector_config as dc 

# --- Motherboard VUV Mapping ---
MB_VUV_MAP = {
    'MB1': [0, 2, 4, 6, 8, 10],
    'MB2': [12, 14, 16, 18, 20, 22],
    'MB3': [24, 26, 28, 30]
}

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
    
    fwhm_ER = 2.355 * sig_ER
    fwhm_NR = 2.355 * sig_NR
    
    denominator = fwhm_ER + fwhm_NR
    if denominator == 0: return 0.0, None, None, None
    
    fom = np.abs(mu_NR - mu_ER) / denominator
    return fom, means, stds, weights

def process_file_data(filepath, target_channels, smart_sum_thresh):
    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean_mV' not in f: return None, 0.0, 2.0, 2.0
            
            dset_waves = f['waveforms_mV']
            dset_base = f['baseline_mean_mV']
            res_ns = f.attrs.get('resolution_ns', 2.0)
            pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 2.0)
            
            if not target_channels: return None, 0.0, res_ns, pre_trig_us
            
            n_events = dset_waves.shape[0]
            n_samples = dset_waves.shape[2]
            
            file_summed_vuv = np.zeros((n_events, n_samples), dtype=np.float32)
            chunk_size = 200 
            
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                corrected_chunk = np.zeros_like(wave_chunk[:, target_channels, :])
                for idx, ch in enumerate(target_channels):
                    corrected_chunk[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])

                if smart_sum_thresh > 0:
                    peaks = np.max(corrected_chunk, axis=2)
                    mask = (peaks > smart_sum_thresh).astype(np.float32)
                    masked_chunk = corrected_chunk * mask[:, :, np.newaxis]
                    chunk_sum = np.sum(masked_chunk, axis=1)
                else:
                    chunk_sum = np.sum(corrected_chunk, axis=1)
                
                file_summed_vuv[i:end_i] = chunk_sum
            
            cumsum_vuv = np.cumsum(file_summed_vuv, axis=1) * res_ns
            return cumsum_vuv, file_size_mb, res_ns, pre_trig_us
            
    except Exception as e:
        return None, 0.0, 2.0, 2.0

def plot_global_spectrum(all_energies, save_dir, ref_window_us, bin_edges):
    plt.figure(figsize=(10, 6))
    n_events = len(all_energies)
    plt.hist(all_energies, bins=100, range=(0, np.max(all_energies)), color='royalblue', alpha=0.7)
    plt.xlabel("Integrated Charge (mV*ns)")
    plt.ylabel("Counts")
    plt.title(f"Global Energy Spectrum (Ref: {ref_window_us} us) | N = {n_events}")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    for b in bin_edges:
        plt.axvline(b, color='red', linestyle='--', alpha=0.5)
        
    path = os.path.join(save_dir, "Global_Energy_Spectrum.png")
    plt.savefig(path)
    plt.close()

def run_grid_search(cumsum_data, res_ns, pre_trig_us, e_min, e_max, args, slice_name):
    prompt_vals = np.arange(args.prompt_range[0], args.prompt_range[1] + args.prompt_range[2], args.prompt_range[2])
    total_vals  = np.arange(args.total_range[0], args.total_range[1] + args.total_range[2], args.total_range[2])
    fom_matrix = np.zeros((len(total_vals), len(prompt_vals)))
    
    trig_idx = int((pre_trig_us * 1000) / res_ns)
    ref_samples = int((args.ref_window_us * 1000) / res_ns)
    end_idx = min(trig_idx + ref_samples, cumsum_data.shape[1] - 1)
    
    ref_energies = cumsum_data[:, end_idx] - cumsum_data[:, trig_idx] 
    mask = (ref_energies >= e_min) & (ref_energies < e_max)
    valid_data = cumsum_data[mask]
    
    if len(valid_data) < 50: return None, None, None, valid_data

    # --- TQDM WRAPPER ADDED HERE FOR THE GRID SEARCH ---
    for i, t_us in enumerate(tqdm(total_vals, desc=f"Scanning {slice_name}", leave=False, unit="row")):
        t_samples = int((t_us * 1000) / res_ns)
        t_end = min(trig_idx + t_samples, valid_data.shape[1] - 1)
        
        q_total = valid_data[:, t_end] - valid_data[:, trig_idx]
        mask_q = q_total > 100
        current_data = valid_data[mask_q]
        current_q_tot = q_total[mask_q]
        
        if len(current_data) < 50: continue

        for j, p_ns in enumerate(prompt_vals):
            p_samples = int(p_ns / res_ns)
            p_end = trig_idx + p_samples
            if p_end >= current_data.shape[1]: p_end = current_data.shape[1] - 1
            
            q_prompt = current_data[:, p_end] - current_data[:, trig_idx]
            f_vals = q_prompt / current_q_tot
            f_vals = f_vals[(f_vals >= 0) & (f_vals <= 1.2)]
            
            if len(f_vals) > args.n_components * 10:
                fom, _, _, _ = calculate_fom_and_params(f_vals, args.n_components)
                fom_matrix[i, j] = fom
                
    return prompt_vals, total_vals, fom_matrix, valid_data

def generate_plots(p_ax, t_ax, matrix, valid_data, res_ns, pre_trig_us, save_dir, tag, n_comp):
    n_events = len(valid_data)
    max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    best_t = t_ax[max_idx[0]]
    best_p = p_ax[max_idx[1]]
    best_fom = matrix[max_idx]
    
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(p_ax, t_ax, matrix, shading='auto', cmap='plasma')
    cbar = plt.colorbar()
    cbar.set_label('FoM')
    plt.plot(best_p, best_t, 'w*', markersize=20, markeredgecolor='k')
    plt.title(f"PSD Optimization | {tag} | N = {n_events}\nMax FoM: {best_fom:.3f} (P={best_p}ns, T={best_t:.1f}us)")
    plt.xlabel("Prompt (ns)")
    plt.ylabel("Total (us)")
    plt.savefig(os.path.join(save_dir, f"Heatmap_{tag}.png"))
    plt.close()
    
    trig_idx = int((pre_trig_us * 1000) / res_ns)
    p_end = trig_idx + int(best_p / res_ns)
    t_end = trig_idx + int((best_t * 1000) / res_ns)
    
    q_p = valid_data[:, p_end] - valid_data[:, trig_idx]
    q_t = valid_data[:, t_end] - valid_data[:, trig_idx]
    mask = q_t > 100
    f_vals = q_p[mask] / q_t[mask]
    f_vals = f_vals[(f_vals >= 0) & (f_vals <= 1.2)]
    
    fom, means, stds, weights = calculate_fom_and_params(f_vals, n_comp)
    
    plt.figure(figsize=(10, 6))
    plt.hist(f_vals, bins=100, density=True, alpha=0.5, color='gray', label=f'Data (N={len(f_vals)})')
    x = np.linspace(0, 1.2, 1000)
    total_pdf = np.zeros_like(x)
    colors = ['blue', 'red', 'green']
    
    if means is not None:
        for i in range(len(means)):
            mu, sig, w = means[i], stds[i], weights[i]
            pdf = w * (1/(sig * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sig)**2)
            total_pdf += pdf
            plt.plot(x, pdf, '--', color=colors[i%3], label=f"Comp {i} (u={mu:.2f})")
        plt.plot(x, total_pdf, 'k-', lw=1.5, label=f"Total Fit (FoM={fom:.2f})")
        
    plt.title(f"Best GMM Fit | {tag} | N = {n_events}")
    plt.xlabel("F_prompt")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"GMM_Fit_{tag}.png"))
    plt.close()
    
    return best_fom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize F_prompt using a GMM Grid Search.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the directory containing HDF5 files.")
    parser.add_argument("-o", "--output_dir", default="PSD_Smart_Scan", help="Subdirectory name for saving output plots.")
    parser.add_argument("--run_mode", type=str, default=None, help="Force run mode (e.g., OSAKA, COMBO). If None, auto-detects.")
    parser.add_argument("--mb", nargs='+', type=str, choices=['MB1', 'MB2', 'MB3'], help="Target specific motherboards (e.g., --mb MB1 MB2).")
    parser.add_argument("--channels", nargs='+', type=int, default=[], help="List of specific channel indices to sum.")
    parser.add_argument("--smart_sum_thresh", type=float, default=6.0, help="Threshold in mV for a channel to be included in the sum.")
    parser.add_argument("--bin_edges", nargs='+', type=float, default=[20000, 30000, 40000, 60000, 100000, 200000], help="Edges for energy slicing.")
    parser.add_argument("--prompt_range", nargs=3, type=float, default=[40, 300, 2], help="Start, Stop, Step for prompt window (ns).")
    parser.add_argument("--total_range", nargs=3, type=float, default=[1.0, 9.0, 0.1], help="Start, Stop, Step for total window (us).")
    parser.add_argument("--n_components", type=int, default=2, help="Number of Gaussians for the GMM fit.")
    parser.add_argument("--ref_window", type=float, default=9.0, dest="ref_window_us", help="Reference window (us) for fixed energy slicing.")
    parser.add_argument("-mf","--max_files", type=int, default=500, help="Maximum number of files to process.")
    parser.add_argument("-w","--workers", type=int, default=4, help="Number of CPU cores to use for file ingestion.")
    
    args = parser.parse_args()

    t_start = time.time()
    total_mb_processed = 0.0

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    if not files:
        print(f"No files found in {args.input_dir}.")
        exit()
        
    run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    
    target_channels = set(args.channels)
    if args.mb:
        for mb in args.mb:
            target_channels.update(MB_VUV_MAP[mb])
    
    if not target_channels:
        ch_map = dc.get_channel_map(run_mode)
        target_channels = set(ch_map['VUV'])
        
    target_channels = sorted(list(target_channels))
    
    print(f"--- GRAMS PSD OPTIMIZER ---")
    print(f"Mode: {run_mode} | Scanning up to {args.max_files} files...")
    print(f"Targeting Channels: {target_channels}")
    print(f"Using {args.workers} concurrent workers for I/O...")
    
    agg_cumsum = []
    res_ns, pre_trig = 2.0, 2.0
    
    # --- TQDM WRAPPER FOR FILE INGESTION ---
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_data, f, target_channels, args.smart_sum_thresh): f for f in files[:args.max_files]}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading Data", unit="file"):
            c, f_size, r, p = future.result()
            if c is not None:
                agg_cumsum.append(c)
                total_mb_processed += f_size
                res_ns, pre_trig = r, p

    if agg_cumsum:
        full_data = np.vstack(agg_cumsum)
        n_total_events = len(full_data)
        
        t_ingest = time.time() - t_start
        ingest_rate = total_mb_processed / t_ingest if t_ingest > 0 else 0
        print(f"\n[TELEMETRY] Ingested {total_mb_processed:.2f} MB ({n_total_events} events) in {t_ingest:.2f}s [{ingest_rate:.2f} MB/s]")
        
        trig = int((pre_trig*1000)/res_ns)
        ref = int((args.ref_window_us*1000)/res_ns)
        end = min(trig+ref, full_data.shape[1]-1)
        all_ens = full_data[:, end] - full_data[:, trig]
        
        SAVE_ROOT = os.path.join(args.input_dir, args.output_dir)
        if not os.path.exists(SAVE_ROOT): os.makedirs(SAVE_ROOT)
        
        plot_global_spectrum(all_ens, SAVE_ROOT, args.ref_window_us, args.bin_edges)
        print(f"Global spectrum saved. Commencing Grid Search...\n")
        
        for i in range(len(args.bin_edges)-1):
            e_min = args.bin_edges[i]
            e_max = args.bin_edges[i+1]
            slice_name = f"E_{int(e_min/1000)}k_{int(e_max/1000)}k"
            
            print(f"Processing {slice_name} ...")
            
            p_ax, t_ax, matrix, valid_data = run_grid_search(
                full_data, res_ns, pre_trig, e_min, e_max, args, slice_name
            )
            
            if matrix is not None and np.max(matrix) > 0:
                slice_dir = os.path.join(SAVE_ROOT, slice_name)
                if not os.path.exists(slice_dir): os.makedirs(slice_dir)
                
                best_fom = generate_plots(p_ax, t_ax, matrix, valid_data, res_ns, pre_trig, slice_dir, slice_name, args.n_components)
                print(f"  -> Done. Max FoM: {best_fom:.3f} (N={len(valid_data)})\n")
            else:
                n_skip = len(valid_data) if valid_data is not None else 0
                print(f"  -> Skipped. Insufficient stats (N={n_skip}).\n")
                
        t_total = time.time() - t_start
        print(f"[TELEMETRY] Mission Complete. Total Execution Time: {t_total:.2f}s")