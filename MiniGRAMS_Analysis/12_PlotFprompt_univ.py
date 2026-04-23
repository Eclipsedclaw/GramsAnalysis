#!/usr/bin/env python3
"""
12_PlotFprompt.py (Universal Edition - MicroG 2x2 & Cut Box Support)
Generates 2D aggregated F_prompt and channel-by-channel grids.
Supports Single-Board (MicroG) 2x2 layouts and Multi-Board (MiniG) 1xN layouts.
Includes optional bounding box overlay for selection visualization.
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import detector_config as dc

def process_file_data(filepath, p_ns, t_us, target_channels):
    try:
        with h5py.File(filepath, 'r') as f:
            is_single = 'waveforms_mV' in f
            
            if is_single:
                if 'baseline_mean_mV' not in f: return None, None
                n_events = f['waveforms_mV'].shape[0]
                n_samples = f['waveforms_mV'].shape[2]
                res_ns = f.attrs.get('resolution_ns', 8.0)
                pre_trig_us = f.attrs.get('baseline_pre_trigger_us', 4.0)
            else:
                if 'Board_72' not in f or 'baseline_mean' not in f['Board_72']: return None, None
                n_events = f['Board_72']['waveforms'].shape[0]
                n_samples = f['Board_72'].attrs['n_samples']
                res_ns = f['Board_72'].attrs.get('sampling_period_ns', 2.0)
                pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)

            if not target_channels: return None, None
            
            trig_idx = int((pre_trig_us * 1000) / res_ns)
            p_end = trig_idx + int(p_ns / res_ns)
            t_end = trig_idx + int((t_us * 1000) / res_ns)
            
            agg_results = np.zeros((n_events, 2), dtype=np.float32)
            ch_results = np.zeros((n_events, len(target_channels), 2), dtype=np.float32)
            valid_event_mask = np.ones(n_events, dtype=bool)
            
            chunk_size = 200 
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                current_chunk_size = end_i - i
                
                corrected_chunk = np.zeros((current_chunk_size, len(target_channels), n_samples), dtype=np.float32)
                
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
                        valid_event_mask[i:end_i][drop_mask] = False

                    corrected_chunk[:, idx, :] = -1 * (wave - base[:, np.newaxis])

                prompt_slice_ch = corrected_chunk[:, :, trig_idx:p_end]
                total_slice_ch = corrected_chunk[:, :, trig_idx:t_end]
                
                ch_q_p = np.sum(prompt_slice_ch, axis=2) * res_ns
                ch_q_t = np.sum(total_slice_ch, axis=2) * res_ns
                
                safe_q_t_ch = np.where(ch_q_t > 0, ch_q_t, 1)
                ch_results[i:end_i, :, 0] = ch_q_t
                ch_results[i:end_i, :, 1] = np.where(ch_q_t > 0, ch_q_p / safe_q_t_ch, 0)
                
                summed_chunk = np.sum(corrected_chunk, axis=1)
                agg_q_p = np.sum(summed_chunk[:, trig_idx:p_end], axis=1) * res_ns
                agg_q_t = np.sum(summed_chunk[:, trig_idx:t_end], axis=1) * res_ns
                
                safe_q_t_agg = np.where(agg_q_t > 0, agg_q_t, 1)
                agg_results[i:end_i, 0] = agg_q_t
                agg_results[i:end_i, 1] = np.where(agg_q_t > 0, agg_q_p / safe_q_t_agg, 0)

            return agg_results[valid_event_mask], ch_results[valid_event_mask]
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Path to HDF5 files.")
    parser.add_argument("-p", "--prompt_ns", required=True, type=float)
    parser.add_argument("-t", "--total_us", required=True, type=float)
    parser.add_argument("-o", "--output_dir", default="PSD_Validation")
    parser.add_argument("--run_mode", type=str, default=None)
    parser.add_argument("-mf", "--max_files", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vmax", type=int, default=None)
    parser.add_argument("--emax", type=float, default=2500000)
    
    # NEW ARGUMENT FOR CUT BOX OVERLAY
    parser.add_argument("--cut_box", nargs=4, type=float, metavar=('E_MIN', 'E_MAX', 'FP_MIN', 'FP_MAX'),
                        help="Draw a bounding box. Format: E_min E_max fp_min fp_max")
    
    args = parser.parse_args()

    t_start = time.time()
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    if not files: exit()

    run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(run_mode)
    
    if run_mode == 'MICROG_PURITY_STUDY':
        target_channels = sorted(list(ch_map['VUV']) + list(ch_map['VIS']))
    else:
        target_channels = sorted(list(ch_map['VUV']))

    print(f"--- UNIVERSAL PSD 2D VISUALIZER ---")
    all_agg_results, all_ch_results = [], []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_data, f, args.prompt_ns, args.total_us, target_channels): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting Data"):
            agg, ch = future.result()
            if agg is not None:
                all_agg_results.append(agg), all_ch_results.append(ch)

    if not all_agg_results: exit()

    stacked_agg = np.vstack(all_agg_results)
    stacked_ch = np.vstack(all_ch_results)
    n_events = len(stacked_agg)
    
    SAVE_ROOT = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    # --- PLOT 1: Aggregate F_prompt ---
    agg_e = stacked_agg[:, 0]
    agg_fp = stacked_agg[:, 1]
    
    valid_mask = (agg_e > 50) & (agg_fp >= 0) & (agg_fp <= 1.2)
    valid_e = agg_e[valid_mask]
    valid_fp = agg_fp[valid_mask]

    plt.figure(figsize=(10, 8))
    h = plt.hist2d(valid_e, valid_fp, bins=[200, 150], range=[[0, args.emax], [0, 1.0]], cmap='turbo', cmin=1, vmax=args.vmax)
    plt.colorbar(h[3], label='Counts')
    plt.title(f"Aggregate Summed VUV F_prompt | N = {len(valid_e)} events\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)")
    plt.xlabel("Integrated Charge (mV*ns)")
    plt.ylabel("F_prompt Ratio")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid(True, alpha=0.3)
    
    # --- APPLY AESTHETIC CUT BOX ---
    if args.cut_box:
        e_min, e_max, fp_min, fp_max = args.cut_box
        # Vertical energy boundaries
        plt.axvline(e_min, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.axvline(e_max, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        # Bright Selection Box
        rect = Rectangle((e_min, fp_min), e_max - e_min, fp_max - fp_min,
                         linewidth=2.5, edgecolor='lime', facecolor='none', linestyle='-')
        plt.gca().add_patch(rect)

    plt.savefig(os.path.join(SAVE_ROOT, "Aggregate_Fprompt_2D.png"), dpi=300)
    plt.close()

    # --- PLOT 2: Channel-by-Channel Grid ---
    n_chs = len(target_channels)
    
    if args.vmax is None:
        max_counts = 0
        for i in range(n_chs):
            ch_e = stacked_ch[:, i, 0]
            ch_fp = stacked_ch[:, i, 1]
            mask = (ch_e > 50) & (ch_fp >= 0) & (ch_fp <= 1.2)
            if np.sum(mask) > 0:
                counts, _, _ = np.histogram2d(ch_e[mask], ch_fp[mask], bins=[100, 100], range=[[0, args.emax], [0, 1.0]])
                max_counts = max(max_counts, counts.max())
        cmax = max_counts if max_counts > 0 else 10
    else: cmax = args.vmax

    im = None
    
    if run_mode == 'MICROG_PURITY_STUDY' and n_chs == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig.suptitle(f"MicroGRAMS F_prompt Maps | N = {n_events} events\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)", fontsize=16)
        
        layout = [(0, 0), (0, 1), (1, 0), (1, 1)]
        labels = ["VUV (Ch 0)", "VUV (Ch 1)", "VIS (Ch 2)", "VIS (Ch 3)"]
        
        for i in range(4):
            ax = axes[layout[i]]
            ch_e = stacked_ch[:, i, 0]
            ch_fp = stacked_ch[:, i, 1]
            
            ch_mask = (ch_e > 50) & (ch_fp >= 0) & (ch_fp <= 1.2)
            im = ax.hist2d(ch_e[ch_mask], ch_fp[ch_mask], bins=[100, 100], range=[[0, args.emax], [0, 1.0]], cmap='turbo', cmin=1, vmax=cmax)
            
            # --- APPLY AESTHETIC CUT BOX TO INDIVIDUAL CHANNELS ---
            if args.cut_box:
                e_min, e_max, fp_min, fp_max = args.cut_box
                ax.axvline(e_min, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.axvline(e_max, color='red', linestyle='--', alpha=0.5, linewidth=1)
                rect = Rectangle((e_min, fp_min), e_max - e_min, fp_max - fp_min,
                                 linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='-')
                ax.add_patch(rect)

            ax.set_title(labels[i], fontsize=12, fontweight='bold')
            if layout[i][0] == 1: ax.set_xlabel("Charge (mV*ns)")
            if layout[i][1] == 0: ax.set_ylabel("F_prompt")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.set_box_aspect(1)
            
        plt.subplots_adjust(top=0.88, bottom=0.10, right=0.85, wspace=0.1, hspace=0.15)
        if im:
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            fig.colorbar(im[3], cax=cbar_ax).set_label('Counts')

    else:
        fig_width = min(26, 4 * n_chs)
        fig, axes = plt.subplots(1, n_chs, figsize=(fig_width, 5), sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f"Individual VUV F_prompt Maps | N = {n_events} events\n($t_p$ = {args.prompt_ns} ns, $t_{{tot}}$ = {args.total_us} $\\mu$s)", fontsize=16)
        
        for i, ch_idx in enumerate(target_channels):
            ax = axes[i]
            ch_e = stacked_ch[:, i, 0]
            ch_fp = stacked_ch[:, i, 1]
            
            ch_mask = (ch_e > 50) & (ch_fp >= 0) & (ch_fp <= 1.2)
            im = ax.hist2d(ch_e[ch_mask], ch_fp[ch_mask], bins=[100, 100], range=[[0, args.emax], [0, 1.0]], cmap='turbo', cmin=1, vmax=cmax)
            
            if args.cut_box:
                e_min, e_max, fp_min, fp_max = args.cut_box
                ax.axvline(e_min, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.axvline(e_max, color='red', linestyle='--', alpha=0.5, linewidth=1)
                rect = Rectangle((e_min, fp_min), e_max - e_min, fp_max - fp_min,
                                 linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='-')
                ax.add_patch(rect)

            ax.set_title(f"Channel {ch_idx}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Charge (mV*ns)")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.set_box_aspect(1)
            if i == 0: ax.set_ylabel("F_prompt")

        plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.1)
        if im: fig.colorbar(im[3], ax=axes.tolist(), pad=0.02, aspect=20).set_label('Counts')

    plt.savefig(os.path.join(SAVE_ROOT, "Channel_Fprompt_Grid.png"), dpi=300, bbox_inches='tight')
    plt.close()

    t_total = time.time() - t_start
    print(f"[TELEMETRY] Plots rendered and saved in {t_total:.2f} seconds.")