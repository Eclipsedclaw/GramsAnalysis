#!/usr/bin/env python3
"""
12b_PlotFprompt_DynamicT0.py (QUARANTINED)
Generates 2D aggregated F_prompt vs Energy grids.
Uses Dynamic T_0 (Peak Finding) to bypass hardware trigger errors.
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
            else:
                if 'Board_72' not in f or 'baseline_mean' not in f['Board_72']: return None, None
                n_events = f['Board_72']['waveforms'].shape[0]
                n_samples = f['Board_72'].attrs['n_samples']
                res_ns = f['Board_72'].attrs.get('sampling_period_ns', 2.0)

            if not target_channels: return None, None
            
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

            # Filter valid events
            valid_chunk = corrected_chunk[valid_event_mask]
            summed_waves = np.sum(valid_chunk, axis=1)
            
            # --- DYNAMIC T_0 ALIGNMENT ---
            peak_indices = np.argmax(summed_waves, axis=1)
            p_samples = int(p_ns / res_ns)
            t_samples = int((t_us * 1000) / res_ns)
            
            n_valid = len(valid_chunk)
            global_e = np.zeros(n_valid)
            global_fp = np.zeros(n_valid)
            channel_data = np.zeros((n_valid, len(target_channels), 2)) # [E, Fp] per channel
            
            for i in range(n_valid):
                p_idx = peak_indices[i]
                p_end = min(p_idx + p_samples, n_samples)
                t_end = min(p_idx + t_samples, n_samples)
                
                # Global (Summed) Math
                q_p = np.sum(summed_waves[i, p_idx:p_end]) * res_ns
                q_t = np.sum(summed_waves[i, p_idx:t_end]) * res_ns
                
                global_e[i] = q_t
                global_fp[i] = q_p / q_t if q_t > 0 else 0
                
                # Individual Channel Math (Anchored to Global Peak)
                for ch_idx in range(len(target_channels)):
                    ch_qp = np.sum(valid_chunk[i, ch_idx, p_idx:p_end]) * res_ns
                    ch_qt = np.sum(valid_chunk[i, ch_idx, p_idx:t_end]) * res_ns
                    channel_data[i, ch_idx, 0] = ch_qt
                    channel_data[i, ch_idx, 1] = ch_qp / ch_qt if ch_qt > 0 else 0
            
            global_data = np.column_stack((global_e, global_fp))
            return global_data, channel_data
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def plot_2d_fprompt(args, target_channels):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))[:args.max_files]
    all_global = []
    all_channels = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_data, f, args.prompt_ns, args.total_us, target_channels): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting Dynamic 2D Data"):
            g_data, c_data = future.result()
            if g_data is not None:
                all_global.append(g_data)
                all_channels.append(c_data)

    if not all_global: return

    stacked_global = np.vstack(all_global)
    stacked_ch = np.vstack(all_channels)
    
    SAVE_ROOT = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # --- 1. GLOBAL AGGREGATE PLOT ---
    plt.figure(figsize=(10, 8))
    g_e = stacked_global[:, 0]
    g_fp = stacked_global[:, 1]
    
    valid_mask = (g_e > 50) & (g_fp >= 0) & (g_fp <= 1.2)
    h = plt.hist2d(g_e[valid_mask], g_fp[valid_mask], bins=[200, 150], range=[[0, args.emax], [0, 1.0]], cmap='turbo', cmin=1)
    plt.colorbar(h[3], label='Counts')
    
    if args.cut_box:
        e_min, e_max, fp_min, fp_max = args.cut_box
        plt.axvline(e_min, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        plt.axvline(e_max, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        rect = Rectangle((e_min, fp_min), e_max - e_min, fp_max - fp_min,
                         linewidth=2, edgecolor='lime', facecolor='none', linestyle='-', label="Selection Box")
        plt.gca().add_patch(rect)
        plt.legend(loc='upper right')

    plt.title(f"Aggregated VUV F_prompt vs Energy (Dynamic $T_0$) | Mode: {args.run_mode}\n$t_p$: {args.prompt_ns} ns | $t_{{tot}}$: {args.total_us} $\\mu$s | N: {len(g_e[valid_mask])}")
    plt.xlabel("Total Integrated Charge (mV*ns)")
    plt.ylabel("F_prompt")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    out_global = os.path.join(SAVE_ROOT, "Fprompt_2D_Global_DynamicT0.png")
    plt.savefig(out_global, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Dynamic Global 2D Plot saved to: {out_global}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default="PSD_Validation_DynamicT0")
    parser.add_argument("-p", "--prompt_ns", type=float, default=200.0)
    parser.add_argument("-t", "--total_us", type=float, default=10.0)
    parser.add_argument("--emax", type=float, default=1000000, help="Max energy for X-axis")
    parser.add_argument("--cut_box", nargs=4, type=float, metavar=('E_MIN', 'E_MAX', 'FP_MIN', 'FP_MAX'), default=None)
    parser.add_argument("--run_mode", type=str, default=None)
    parser.add_argument("-mf", "--max_files", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.run_mode = args.run_mode if args.run_mode else dc.detect_run_mode(args.input_dir)
    ch_map = dc.get_channel_map(args.run_mode)
    target_vuv_chs = sorted(list(ch_map['VUV']))
    
    print("--- QUARANTINED 2D F_PROMPT PLOTTER (DYNAMIC T0) ---")
    plot_2d_fprompt(args, target_vuv_chs)