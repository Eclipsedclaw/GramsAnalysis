"""
40_Charge_Pixel_Heatmap_Targeted.py
LArTPC 2D Pixel "Track Persist" Mapper
- Solves the X/Y ambiguity "Ghost Pixel" problem using temporal matrix dot products.
- NEW: --target_event flag isolates a specific event ID for single-track visualization.
- UPDATED FOR RUN 7/8: Board 85 is Slave 1, Board 75 is Slave 2.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import os
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import detector_config as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- STYLE ---
plt.style.use('dark_background')
cmap_choice = 'inferno'

def get_charge_trace(f, event_idx, global_ch):
    if global_ch < 32: return None, None
    elif global_ch < 96: grp, local_ch = f['Board_85'], global_ch - 32
    else: grp, local_ch = f['Board_75'], global_ch - 96
        
    dt_ns = grp.attrs['sampling_period_ns']
    trace = grp['waveforms'][event_idx, local_ch, :].copy() 
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
        trace -= grp['baseline_mean'][event_idx, local_ch]
    return trace, dt_ns

def apply_filters(trace, res_ns, sg_win_ns, gauss_sigma_ns):
    win_samples = int(sg_win_ns / res_ns)
    if win_samples % 2 == 0: win_samples += 1
    if win_samples < 3: win_samples = 3
    
    try: sg_v = savgol_filter(trace, window_length=win_samples, polyorder=2)
    except ValueError: return np.zeros_like(trace)
        
    deriv = np.gradient(sg_v)
    return gaussian_filter1d(deriv, sigma=(gauss_sigma_ns / res_ns))

def check_light_trigger(f, event_idx, light_thresh):
    b72 = f['Board_72']
    for ch in range(32):
        trace = b72['waveforms'][event_idx, ch, :].copy()
        if 'baseline_mean' in b72 and not np.isnan(b72['baseline_mean'][event_idx, ch]):
            trace -= b72['baseline_mean'][event_idx, ch]
        if np.nanmin(trace) < -abs(light_thresh):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="2D Coincident Pixel Heatmap Generator")
    parser.add_argument('input_target', help="Directory containing synced HDF5 files")
    parser.add_argument('--events', '-e', type=int, default=0, help="Max events to process (Bulk mode)")
    parser.add_argument('--target_event', '-t', type=int, default=-1, help="Specific Event ID to process exclusively")
    parser.add_argument('--sg_x', type=float, required=True)
    parser.add_argument('--gauss_x', type=float, required=True)
    parser.add_argument('--sg_y', type=float, required=True)
    parser.add_argument('--gauss_y', type=float, required=True)
    parser.add_argument('--charge_thresh', '-c', type=float, default=0.001, help="Min induced current (arb)")
    parser.add_argument('--light_thresh', '-l', type=float, default=75.0, help="Min SiPM deviation (mV)")
    args = parser.parse_args()

    if os.path.isfile(args.input_target):
        files, acq_name = [args.input_target], os.path.basename(args.input_target).replace(".h5", "")
    else:
        files = sorted(glob.glob(os.path.join(args.input_target, "*.h5")))
        acq_name = os.path.basename(os.path.normpath(args.input_target))
        
    if not files: return print("No HDF5 files found.")

    mapping = dc.get_channel_map("MINIG")
    x_chans = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"]
    y_chans = mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]
    
    nx, ny = len(x_chans), len(y_chans)
    master_grid = np.zeros((ny, nx), dtype=np.float64)
    
    mode_str = f"SINGLE EVENT {args.target_event}" if args.target_event >= 0 else f"BULK ({args.events if args.events > 0 else 'All'} events)"
    
    print(f"\n=== 2D Pixel Track Map | {mode_str} ===")
    print(f"Charge Cut: > {args.charge_thresh} (Current amp)")
    print("-" * 50)

    events_processed, golden_tracks = 0, 0
    
    for filepath in files:
        if args.target_event < 0 and args.events > 0 and events_processed >= args.events: break
        
        with h5py.File(filepath, 'r') as f:
            if 'baseline_mean' not in f['Board_72']: continue
            n_events = f['Board_72']['waveforms'].shape[0]
            n_samples = f['Board_85'].attrs['n_samples']
            res_ns = f['Board_85'].attrs['sampling_period_ns']
            
            X_mat = np.zeros((nx, n_samples), dtype=bool)
            Y_mat = np.zeros((ny, n_samples), dtype=bool)
            
            with tqdm(total=n_events, desc=os.path.basename(filepath), unit="evt") as pbar:
                for i in range(n_events):
                    pbar.update(1)
                    if args.target_event >= 0 and i != args.target_event: continue
                    if args.target_event < 0 and args.events > 0 and events_processed >= args.events: continue
                    
                    if np.isnan(f['Board_75']['waveforms'][i, 0, 0]): continue
                    if not check_light_trigger(f, i, args.light_thresh): continue
                    
                    has_x = False
                    for idx_x, ch in enumerate(x_chans):
                        raw, _ = get_charge_trace(f, i, ch)
                        if np.all(np.isnan(raw)): continue
                        current = apply_filters(raw, res_ns, args.sg_x, args.gauss_x)
                        X_mat[idx_x, :] = current > args.charge_thresh
                        if np.any(X_mat[idx_x, :]): has_x = True
                    
                    has_y = False
                    if has_x:
                        for idx_y, ch in enumerate(y_chans):
                            raw, _ = get_charge_trace(f, i, ch)
                            if np.all(np.isnan(raw)): continue
                            current = apply_filters(raw, res_ns, args.sg_y, args.gauss_y)
                            Y_mat[idx_y, :] = current > args.charge_thresh
                            if np.any(Y_mat[idx_y, :]): has_y = True
                            
                    if has_x and has_y:
                        event_pixels = Y_mat.astype(float) @ X_mat.T.astype(float)
                        master_grid += event_pixels * (res_ns / 1000.0)
                        golden_tracks += 1
                        
                    events_processed += 1
                    X_mat.fill(False); Y_mat.fill(False)

    if np.max(master_grid) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(master_grid, cmap=cmap_choice, origin='lower', 
                       norm=LogNorm(vmin=0.01, vmax=np.max(master_grid)), interpolation='nearest')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Total Coincidence Time (µs)', rotation=270, labelpad=20, fontsize=12)
        
        title_evt = f"Event {args.target_event}" if args.target_event >= 0 else f"{golden_tracks} Tracks Accumulated"
        ax.set_title(f"2D Pixel Track Map | {acq_name}\n({title_evt})", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("X-Anode Strips", fontsize=12); ax.set_ylabel("Y-Anode Strips", fontsize=12)
        
        for pos in [13.5, 26.5]: ax.axvline(pos, color='gray', ls='--', lw=1, alpha=0.5)
        for pos in [13.5, 28.5]: ax.axhline(pos, color='gray', ls='--', lw=1, alpha=0.5)
        
        plt.tight_layout()
        out_dir = os.path.join(os.path.dirname(args.input_target) if os.path.isfile(args.input_target) else args.input_target, "Plots")
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"Evt{args.target_event}" if args.target_event >= 0 else "Bulk"
        out_path = os.path.join(out_dir, f"{acq_name}_PixelMap_{suffix}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"\n[SUCCESS] Track Map saved to: {out_path}")
    else:
        print("\n[WARNING] No tracks survived the coincidence cuts.")

if __name__ == "__main__": main()