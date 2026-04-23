#!/usr/bin/env python3
"""
plot_aggregate_correlation.py
Performs Dumb Sum on user-defined VUV and VIS motherboards. 
Uses GMM F_prompt bounds with decoupled sigma cuts to isolate Upper/Lower bands 
and plots VUV vs VIS correlation.
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import FormatStrFormatter

# --- Physical Channel Mapping ---
MB_MAP = {
    'MB1': {'VUV': [0, 2, 4, 6, 8, 10], 'VIS': [1, 3, 5, 7, 9, 11]},
    'MB2': {'VUV': [12, 14, 16, 18, 20, 22], 'VIS': [13, 15, 17, 19, 21, 23]},
    'MB3': {'VUV': [24, 26, 28, 30], 'VIS': [25, 27, 29, 31]}
}

def extract_dumb_sums(filepath, p_ns, t_us, e_min, e_max, vuv_chs, vis_chs):
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
            
            valid_events = []
            
            chunk_size = 200
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                # Extract and baseline-correct VUV
                vuv_corr = np.zeros((end_i - i, len(vuv_chs), wave_chunk.shape[2]))
                for idx, ch in enumerate(vuv_chs):
                    vuv_corr[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                # Extract and baseline-correct VIS
                vis_corr = np.zeros((end_i - i, len(vis_chs), wave_chunk.shape[2]))
                for idx, ch in enumerate(vis_chs):
                    vis_corr[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                # DUMB SUMMING (No Thresholds)
                vuv_summed = np.sum(vuv_corr, axis=1)
                vis_summed = np.sum(vis_corr, axis=1)
                
                # VUV Integration
                vuv_q_p = np.sum(vuv_summed[:, trig_idx:p_end], axis=1) * res_ns
                vuv_q_t = np.sum(vuv_summed[:, trig_idx:t_end], axis=1) * res_ns
                
                # VIS Integration (Total window ONLY)
                vis_q_t = np.sum(vis_summed[:, trig_idx:t_end], axis=1) * res_ns
                
                # VUV Global Energy Cut Mask
                mask = (vuv_q_t >= e_min) & (vuv_q_t < e_max) & (vuv_q_t > 0)
                
                for ev_idx in np.where(mask)[0]:
                    fp = vuv_q_p[ev_idx] / vuv_q_t[ev_idx]
                    if 0 <= fp <= 1.2:
                        valid_events.append((vuv_q_t[ev_idx], vis_q_t[ev_idx], fp))
                        
            return valid_events
    except:
        return []

def process_directory(directory, p_ns, t_us, e_min, e_max, vuv_chs, vis_chs, workers, max_files):
    files = sorted(glob.glob(os.path.join(directory, "*.h5")))[:max_files]
    agg_data = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_dumb_sums, f, p_ns, t_us, e_min, e_max, vuv_chs, vis_chs): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc=f"Scanning {os.path.basename(directory)}"):
            res = future.result()
            if res:
                agg_data.extend(res)
    return np.array(agg_data)

def isolate_bands(data_array, sig_lower_mult, sig_upper_mult, n_comp):
    vuv_e, vis_e, f_vals = data_array[:,0], data_array[:,1], data_array[:,2]
    
    gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=200)
    gmm.fit(f_vals.reshape(-1, 1))
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    
    sorted_indices = np.argsort(means)
    idx_low = sorted_indices[0]  
    idx_up = sorted_indices[-1]  
    
    mu_low, sig_low = means[idx_low], stds[idx_low] * sig_lower_mult
    mu_up, sig_up = means[idx_up], stds[idx_up] * sig_upper_mult
    
    mask_low = (f_vals >= (mu_low - sig_low)) & (f_vals <= (mu_low + sig_low))
    mask_up = (f_vals >= (mu_up - sig_up)) & (f_vals <= (mu_up + sig_up))
    
    return (vuv_e[mask_low], vis_e[mask_low], mu_low, stds[idx_low]), \
           (vuv_e[mask_up], vis_e[mask_up], mu_up, stds[idx_up])

def plot_correlation_panel(ax, vuv_e, vis_e, title, e_min, e_max):
    if len(vuv_e) < 10:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
        ax.set_title(title, fontsize=10)
        return

    # Expanded ranges for aesthetics
    x_range = [e_min - 10000, e_max + 10000]
    y_range = [-15000, 100000]
    
    h = ax.hist2d(vuv_e, vis_e, bins=[120, 120], range=[x_range, y_range], cmap='turbo', cmin=1, norm=matplotlib.colors.LogNorm())
    
    # Force scalar formatter and disable scientific notation on the colorbar
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    cbar = plt.colorbar(h[3], ax=ax, label='Counts', format=FormatStrFormatter('%g'))
    
    ax.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax.set_title(f"{title}\nN = {len(vuv_e)}", fontsize=11)
    ax.set_xlabel("VUV Charge (mV*ns)")
    ax.set_ylabel("VIS Charge (mV*ns)")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="VUV vs VIS Correlation with Decoupled Sigma Cuts.")
    parser.add_argument("-d0", "--dir_acq0", required=True)
    parser.add_argument("-d1", "--dir_acq1", required=True)
    parser.add_argument("-p", "--prompt_ns", type=float, default=64.0)
    parser.add_argument("-t", "--total_us", type=float, default=2.9)
    parser.add_argument("--e_min", type=float, default=100000)
    parser.add_argument("--e_max", type=float, default=200000)
    
    parser.add_argument("--mb_vuv", nargs='+', type=str, default=['MB2'], choices=['MB1', 'MB2', 'MB3'])
    parser.add_argument("--mb_vis", nargs='+', type=str, default=['MB2'], choices=['MB1', 'MB2', 'MB3'])
    
    parser.add_argument("-n", "--n_comp", type=int, default=2, help="Number of GMM components")
    
    parser.add_argument("-s0l", "--sig0_lower", type=float, default=1.0)
    parser.add_argument("-s0u", "--sig0_upper", type=float, default=1.0)
    parser.add_argument("-s1l", "--sig1_lower", type=float, default=0.5)
    parser.add_argument("-s1u", "--sig1_upper", type=float, default=0.5)
    
    parser.add_argument("-mf", "--max_files", type=int, default=480)
    parser.add_argument("--workers", type=int, default=8)
    
    args = parser.parse_args()
    
    target_vuv_chs = []
    for mb in args.mb_vuv: target_vuv_chs.extend(MB_MAP[mb]['VUV'])
    target_vis_chs = []
    for mb in args.mb_vis: target_vis_chs.extend(MB_MAP[mb]['VIS'])
        
    print("--- GRAMS CORRELATION ENGINE ---")
    data_acq0 = process_directory(args.dir_acq0, args.prompt_ns, args.total_us, args.e_min, args.e_max, target_vuv_chs, target_vis_chs, args.workers, args.max_files)
    data_acq1 = process_directory(args.dir_acq1, args.prompt_ns, args.total_us, args.e_min, args.e_max, target_vuv_chs, target_vis_chs, args.workers, args.max_files)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"VUV vs VIS Correlation | Energy: {int(args.e_min/1000)}k - {int(args.e_max/1000)}k\n(VUV: {args.mb_vuv}, VIS: {args.mb_vis} | $t_{{tot}}$ = {args.total_us} $\\mu$s)", fontsize=16)

    # Process ACQ0
    if len(data_acq0) > 50:
        low0, up0 = isolate_bands(data_acq0, args.sig0_lower, args.sig0_upper, args.n_comp)
        plot_correlation_panel(axes[0, 0], low0[0], low0[1], f"ACQ0 (Source) | Lower Band\nCut: u={low0[2]:.2f} +/- {args.sig0_lower}*sig", args.e_min, args.e_max)
        plot_correlation_panel(axes[0, 1], up0[0], up0[1], f"ACQ0 (Source) | Upper Band\nCut: u={up0[2]:.2f} +/- {args.sig0_upper}*sig", args.e_min, args.e_max)

    # Process ACQ1
    if len(data_acq1) > 50:
        low1, up1 = isolate_bands(data_acq1, args.sig1_lower, args.sig1_upper, args.n_comp)
        plot_correlation_panel(axes[1, 0], low1[0], low1[1], f"ACQ1 (Bkg) | Lower Band\nCut: u={low1[2]:.2f} +/- {args.sig1_lower}*sig", args.e_min, args.e_max)
        plot_correlation_panel(axes[1, 1], up1[0], up1[1], f"ACQ1 (Bkg) | Upper Band\nCut: u={up1[2]:.2f} +/- {args.sig1_upper}*sig", args.e_min, args.e_max)

    # Apply bounding box to protect the title and spread the rows
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(hspace=0.35)
    plt.savefig("VUV_VIS_Correlation.png", dpi=300)
    
    t_total = time.time() - t_start
    print(f"[TELEMETRY] Saved VUV_VIS_Correlation.png in {t_total:.1f}s")