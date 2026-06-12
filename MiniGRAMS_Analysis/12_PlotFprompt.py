#!/usr/bin/env python3
"""
plot_fprompt_grid.py
Generates 2D aggregated F_prompt and 1x6 channel-by-channel grids.
Includes TQDM progress bars, scientific notation, square aspect ratios, 
and dynamic colorbar scaling.
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

# MB2 VUV Specific Layout (1x6 row)
MB2_VUV_CHS = [12, 14, 16, 18, 20, 22]

def process_file_data(filepath, p_ns, t_us, smart_sum_thresh):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean_mV' not in f: return None
            
            dset_waves = f['waveforms_mV']
            dset_base = f['baseline_mean_mV']
            res_ns = f.attrs.get('resolution_ns', 2.0)
            pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 2.0)
            
            n_events = dset_waves.shape[0]
            trig_idx = int((pre_trig_us * 1000) / res_ns)
            p_end = trig_idx + int(p_ns / res_ns)
            t_end = trig_idx + int((t_us * 1000) / res_ns)
            
            ch_results = np.zeros((n_events, len(MB2_VUV_CHS), 2), dtype=np.float32)
            
            chunk_size = 200
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                wave_chunk = dset_waves[i:end_i]
                base_chunk = dset_base[i:end_i]
                
                corrected_chunk = np.zeros((end_i - i, len(MB2_VUV_CHS), wave_chunk.shape[2]))
                for idx, ch in enumerate(MB2_VUV_CHS):
                    corrected_chunk[:, idx, :] = -1 * (wave_chunk[:, ch, :] - base_chunk[:, ch][:, np.newaxis])
                
                prompt_slice = corrected_chunk[:, :, trig_idx:p_end]
                total_slice = corrected_chunk[:, :, trig_idx:t_end]
                
                ch_q_p = np.sum(prompt_slice, axis=2) * res_ns
                ch_q_t = np.sum(total_slice, axis=2) * res_ns
                
                safe_q_t = np.where(ch_q_t > 0, ch_q_t, 1)
                ch_f_p = np.where(ch_q_t > 0, ch_q_p / safe_q_t, 0)
                
                ch_results[i:end_i, :, 0] = ch_q_t
                ch_results[i:end_i, :, 1] = ch_f_p
                
            return ch_results
            
    except Exception as e:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate physical F_prompt grids with optimized windows.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to HDF5 files.")
    parser.add_argument("-p", "--prompt_ns", required=True, type=float, help="Optimized Prompt Window (ns).")
    parser.add_argument("-t", "--total_us", required=True, type=float, help="Optimized Total Window (us).")
    parser.add_argument("-o", "--output_dir", default="PSD_Validation", help="Output directory name.")
    parser.add_argument("--smart_sum_thresh", type=float, default=6.0, help="Smart sum threshold (mV).")
    parser.add_argument("-mf", "--max_files", type=int, default=500, help="Max files to process.")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU cores.")
    parser.add_argument("--vmax", type=int, default=None, help="Force maximum value for colorbar scale.")
    
    args = parser.parse_args()

    t_start = time.time()
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    
    if not files:
        print("No files found.")
        exit()

    print(f"--- GRAMS PSD VALIDATION ---")
    print(f"Targeting MB2 VUV Channels: {MB2_VUV_CHS}")
    print(f"Windows -> Prompt: {args.prompt_ns} ns | Total: {args.total_us} us")
    
    all_ch_results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_data, f, args.prompt_ns, args.total_us, args.smart_sum_thresh): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Processing Archives", unit="file"):
            res = future.result()
            if res is not None:
                all_ch_results.append(res)

    if not all_ch_results:
        print("No valid data extracted.")
        exit()

    stacked_ch = np.vstack(all_ch_results)
    n_events = len(stacked_ch)
    
    SAVE_ROOT = os.path.join(args.input_dir, args.output_dir)
    if not os.path.exists(SAVE_ROOT): os.makedirs(SAVE_ROOT)
    
    print(f"\nProcessing complete. Generating plots for {n_events} events...")

    # --- PLOT 1: Aggregate 2D F_prompt ---
    ch_e_flat = stacked_ch[:, :, 0].flatten()
    ch_fp_flat = stacked_ch[:, :, 1].flatten()
    
    valid_mask = (ch_e_flat > 50) & (ch_fp_flat >= 0) & (ch_fp_flat <= 1.2)
    valid_e = ch_e_flat[valid_mask]
    valid_fp = ch_fp_flat[valid_mask]
    n_points = len(valid_e)

    plt.figure(figsize=(10, 8))
    h = plt.hist2d(valid_e, valid_fp, bins=[150, 150], range=[[0, 200000], [0, 1.0]], cmap='turbo', cmin=1, vmax=args.vmax)
    plt.colorbar(h[3], label='Counts')
    plt.title(f"Aggregate F_prompt (All MB2 VUVs) | N = {n_points} hits\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)")
    plt.xlabel("Integrated Charge (mV*ns)")
    plt.ylabel("F_prompt Ratio")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_ROOT, "Aggregate_Fprompt_2D.png"), dpi=300)
    plt.close()

    # --- PLOT 2: MB2 VUV 1x6 Grid ---
    fig, axes = plt.subplots(1, 6, figsize=(26, 5), sharex=True, sharey=True)
    fig.suptitle(f"MB2 VUV F_prompt Map | N = {n_events} events\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)", fontsize=16)
    
    # Calculate global max for shared colorbar if vmax not provided
    if args.vmax is None:
        max_counts = 0
        for i, ch in enumerate(MB2_VUV_CHS):
            ch_e = stacked_ch[:, i, 0]
            ch_fp = stacked_ch[:, i, 1]
            ch_mask = (ch_e > 50) & (ch_fp >= 0) & (ch_fp <= 1.2)
            counts, _, _ = np.histogram2d(ch_e[ch_mask], ch_fp[ch_mask], bins=[100, 100], range=[[0, 200000], [0, 1.0]])
            max_counts = max(max_counts, counts.max())
        cmax = max_counts
    else:
        cmax = args.vmax

    im = None
    for i, ch in enumerate(MB2_VUV_CHS):
        ax = axes[i]
        ch_e = stacked_ch[:, i, 0]
        ch_fp = stacked_ch[:, i, 1]
        
        ch_mask = (ch_e > 50) & (ch_fp >= 0) & (ch_fp <= 1.2)
        im = ax.hist2d(ch_e[ch_mask], ch_fp[ch_mask], bins=[100, 100], range=[[0, 200000], [0, 1.0]], cmap='turbo', cmin=1, vmax=cmax)
        
        ax.set_title(f"MB2-CH{ch-11} (Ch {ch})", fontsize=10)
        ax.set_xlabel("Charge (mV*ns)")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_box_aspect(1) # Forces square aspect ratio
        if i == 0: ax.set_ylabel("F_prompt")

    # Adjust spacing to prevent title collision
    plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.1)

    if im:
        cbar = fig.colorbar(im[3], ax=axes.ravel().tolist(), pad=0.02, aspect=20)
        cbar.set_label('Counts')

    plt.savefig(os.path.join(SAVE_ROOT, "MB2_Fprompt_Grid_1x6.png"), dpi=300, bbox_inches='tight')
    plt.close()

    t_total = time.time() - t_start
    print(f"[TELEMETRY] Plots rendered and saved in {t_total:.2f} seconds.")