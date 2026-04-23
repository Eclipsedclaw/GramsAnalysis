#!/usr/bin/env python3
"""
plot_3d_compton_scatter.py
Hunts for multi-site Compton scatters by correlating light yield across three optically isolated cells.
Performs single-pass integration to avoid NAS bottlenecks.
Includes low-statistics failsafe and explicit negative bounds for electronic cross-talk.
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

# --- Optical Cell Mapping ---
CELL_MAP = {
    'MB1': {'C1': [0, 2], 'C2': [4, 6], 'C3': [8, 10]},
    'MB2': {'C1': [12, 14], 'C2': [16, 18], 'C3': [20, 22]}
}

def extract_cell_energies(filepath, p_ns, t_us, e_min, e_max, cell_dict):
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
            
            all_vuv = cell_dict['C1'] + cell_dict['C2'] + cell_dict['C3']
            c1_len = len(cell_dict['C1'])
            c2_len = len(cell_dict['C2'])
            
            valid_events = []
            
            chunk_size = 200
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                # Baseline correct target channels
                vuv_corr = np.zeros((end_i - i, len(all_vuv), wave_chunk.shape[2]))
                for idx, ch in enumerate(all_vuv):
                    vuv_corr[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                # Global Integration for the Energy & F_prompt Cut
                global_sum = np.sum(vuv_corr, axis=1)
                global_q_p = np.sum(global_sum[:, trig_idx:p_end], axis=1) * res_ns
                global_q_t = np.sum(global_sum[:, trig_idx:t_end], axis=1) * res_ns
                
                # Cell-by-Cell Integration (Dumb Sum)
                c1_waves = np.sum(vuv_corr[:, 0:c1_len, :], axis=1)
                c2_waves = np.sum(vuv_corr[:, c1_len:c1_len+c2_len, :], axis=1)
                c3_waves = np.sum(vuv_corr[:, c1_len+c2_len:, :], axis=1)
                
                c1_q_t = np.sum(c1_waves[:, trig_idx:t_end], axis=1) * res_ns
                c2_q_t = np.sum(c2_waves[:, trig_idx:t_end], axis=1) * res_ns
                c3_q_t = np.sum(c3_waves[:, trig_idx:t_end], axis=1) * res_ns
                
                # Apply Global Energy Cut
                mask = (global_q_t >= e_min) & (global_q_t < e_max) & (global_q_t > 0)
                
                for ev_idx in np.where(mask)[0]:
                    fp = global_q_p[ev_idx] / global_q_t[ev_idx]
                    if 0 <= fp <= 1.2:
                        valid_events.append((fp, c1_q_t[ev_idx], c2_q_t[ev_idx], c3_q_t[ev_idx], global_q_t[ev_idx]))
                        
            return valid_events
    except:
        return []

def process_directory(directory, p_ns, t_us, e_min, e_max, cell_dict, workers, max_files):
    files = sorted(glob.glob(os.path.join(directory, "*.h5")))[:max_files]
    agg_data = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_cell_energies, f, p_ns, t_us, e_min, e_max, cell_dict): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Scanning Events"):
            res = future.result()
            if res:
                agg_data.extend(res)
    return np.array(agg_data)

if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="3D Compton Scatter Correlation Map.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("-p", "--prompt_ns", type=float, default=64.0)
    parser.add_argument("-t", "--total_us", type=float, default=2.9)
    parser.add_argument("--e_min", type=float, default=100000)
    parser.add_argument("--e_max", type=float, default=200000)
    
    parser.add_argument("--mb", type=str, default='MB2', choices=['MB1', 'MB2'], help="Motherboard to map into 3 cells")
    parser.add_argument("-n", "--n_comp", type=int, default=2)
    parser.add_argument("-s0l", "--sig0_lower", type=float, default=1.0, help="Sigma cut to isolate the Gamma band")
    
    parser.add_argument("-mf", "--max_files", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    print("--- GRAMS COMPTON SCATTER TRACKER ---")
    cell_target = CELL_MAP[args.mb]
    
    raw_data = process_directory(args.input_dir, args.prompt_ns, args.total_us, args.e_min, args.e_max, cell_target, args.workers, args.max_files)

    if len(raw_data) < 50:
        print("Insufficient data to perform correlation.")
        exit()

    # Isolate Lower Band (Gamma)
    f_vals = raw_data[:, 0]
    gmm = GaussianMixture(n_components=args.n_comp, random_state=42, max_iter=200)
    gmm.fit(f_vals.reshape(-1, 1))
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    idx_low = np.argmin(means)
    
    mu_low = means[idx_low]
    sig_cut = stds[idx_low] * args.sig0_lower
    
    mask_low = (f_vals >= (mu_low - sig_cut)) & (f_vals <= (mu_low + sig_cut))
    gamma_data = raw_data[mask_low]
    
    c1 = gamma_data[:, 1]
    c2 = gamma_data[:, 2]
    c3 = gamma_data[:, 3]
    global_e = gamma_data[:, 4]
    
    n_events = len(c1)
    print(f"\n[INFO] Isolated {n_events} Gamma Events (u={mu_low:.3f}). Plotting Kinematics...")
    
    # Plotting Setup
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Multi-Site Compton Scatter Phase Space | MB Target: {args.mb} (Gamma Band)\nN = {n_events} | Global Energy: {int(args.e_min/1000)}k - {int(args.e_max/1000)}k", fontsize=16)
    
    # Mathematical scaling to prevent Matplotlib offset text collisions
    scale_factor = 1e5
    c1_scaled = c1 / scale_factor
    c2_scaled = c2 / scale_factor
    c3_scaled = c3 / scale_factor
    global_e_scaled = global_e / scale_factor
    
    plot_min = -25000 / scale_factor
    plot_max = args.e_max / scale_factor
    unit_label = r"($\times 10^5$ mV$\cdot$ns)"
    
    # 3D Plot
    ax3d = fig.add_subplot(221, projection='3d')
    sc3d = ax3d.scatter(c1_scaled, c2_scaled, c3_scaled, c=global_e_scaled, cmap='turbo', alpha=0.6, s=15)
    cbar3d = fig.colorbar(sc3d, ax=ax3d, pad=0.1, label=f'Global Energy {unit_label}')
    cbar3d.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    
    ax3d.set_xlim(plot_min, plot_max)
    ax3d.set_ylim(plot_min, plot_max)
    ax3d.set_zlim(plot_min, plot_max)
    ax3d.set_xlabel(f'Cell 1 {cell_target["C1"]}\n{unit_label}', labelpad=14)
    ax3d.set_ylabel(f'Cell 2 {cell_target["C2"]}\n{unit_label}', labelpad=14)
    ax3d.set_zlabel(f'Cell 3 {cell_target["C3"]}\n{unit_label}', labelpad=14)
    ax3d.set_title('3D Phase Space', pad=20)
    ax3d.tick_params(labelsize=7) 
    
    # 2D Projections
    axes_2d = [fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    pairs = [
        (c1_scaled, c2_scaled, f'Cell 1 {cell_target["C1"]}', f'Cell 2 {cell_target["C2"]}'), 
        (c2_scaled, c3_scaled, f'Cell 2 {cell_target["C2"]}', f'Cell 3 {cell_target["C3"]}'), 
        (c1_scaled, c3_scaled, f'Cell 1 {cell_target["C1"]}', f'Cell 3 {cell_target["C3"]}')
    ]
    
    for ax, (x_data, y_data, x_label, y_label) in zip(axes_2d, pairs):
        # MATHEMATICAL FAILSAFE
        counts, _, _ = np.histogram2d(x_data, y_data, bins=100, range=[[plot_min, plot_max], [plot_min, plot_max]])
        safe_vmax = np.max(counts)
        if safe_vmax <= 1:
            safe_vmax = 10  
            
        h = ax.hist2d(x_data, y_data, bins=100, range=[[plot_min, plot_max], [plot_min, plot_max]], cmap='turbo', cmin=1, norm=matplotlib.colors.LogNorm(vmin=1, vmax=safe_vmax))
        cbar = fig.colorbar(h[3], ax=ax, label='Counts', format=FormatStrFormatter('%g'))
        
        # Explicitly set the 2D limits to match the 3D limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Draw explicit physical zero lines
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax.axvline(0, color='white', linestyle='--', alpha=0.5)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], color='white', linestyle=':', alpha=0.4)
        
        ax.set_xlabel(f'{x_label} {unit_label}')
        ax.set_ylabel(f'{y_label} {unit_label}')
        ax.set_title(f'{x_label} vs {y_label}')
        
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.savefig("Compton_Scatter_3D.png", dpi=300)
    
    t_total = time.time() - t_start
    print(f"[TELEMETRY] Mission Complete in {t_total:.1f}s. Saved as Compton_Scatter_3D.png")