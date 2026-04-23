"""
06_Charge_2D_Heatmap.py
LArTPC 2D Charge Response Visualizer
- Averages waveforms to suppress periodic switching noise.
- Reconstructs a perfect top-down physical view of the Anode plane (45x45).
- Explicitly models unrouted N/C channels as physical voids in the matrix.
- Computes the geometric mean of orthogonal strips (X, Y).
- Upgraded: Uses 'viridis' perceptually uniform colormap for publication-ready figures.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import detector_config as dc
import grams_tools as gt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def detect_config(h5_file, filepath):
    path_config = dc.detect_run_mode(filepath)
    if path_config != 'LIGHT_ONLY' and path_config != 'MINIG':
        return path_config
    if 'run_config' in h5_file.attrs:
        return h5_file.attrs['run_config']
    return 'MINIG' 

def get_trace(f, event_idx, global_ch):
    if global_ch < 32: 
        grp = f['Board_72']; local_ch = global_ch
    elif global_ch < 96: 
        grp = f['Board_85']; local_ch = global_ch - 32
    else: 
        grp = f['Board_75']; local_ch = global_ch - 96
        
    dt_ns = grp.attrs['sampling_period_ns']
    trace = grp['waveforms'][event_idx, local_ch, :].copy() 
    
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
        trace -= grp['baseline_mean'][event_idx, local_ch]
        
    return dt_ns, trace

def build_axis_info(mapping, bank_names):
    """
    Constructs the exact 45-channel spatial mapping.
    Forces padding for unrouted N/C channels to maintain true physical geometry.
    """
    chans = []
    labels = []
    for bank in bank_names:
        bank_chans = mapping.get(f"Charge_{bank}", [])
        
        # If all 15 channels are present
        if len(bank_chans) == 15:
            chans.extend(bank_chans[::-1])
            labels.extend([f"{bank}{i}" for i in range(14, -1, -1)])
            
        # If only 14 channels are present (Run 9 setup: SIG0 is dropped on B, D, F)
        elif len(bank_chans) == 14:
            chans.extend(bank_chans[::-1])
            labels.extend([f"{bank}{i}" for i in range(14, 0, -1)])
            # Inject dummy channel marker (-1) for the missing SIG0
            chans.append(-1)
            labels.append(f"{bank}0 (N/C)")
            
    return chans, labels

def plot_heatmap(matrix, title, cbar_label, out_filepath, x_labels, y_labels):
    """Generates and saves the 45x45 2D heatmap figure."""
    fig, ax = plt.subplots(figsize=(15, 13))
    
    # Copy viridis map and explicitly set NaN (missing) values to dark void
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#111111')
    
    # origin='upper' forces Bank A14 / D14 to the absolute top-left physical corner
    im = ax.imshow(matrix.T, origin='upper', cmap=cmap, aspect='auto')
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    # Set literal tick labels 
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=10, fontweight='bold')
    ax.set_xlabel("X-Anode Channels (Banks D, E, F) [Left -> Right]", fontsize=14, fontweight='bold', labelpad=10)
    
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel("Y-Anode Channels (Banks A, B, C) [Top -> Bottom]", fontsize=14, fontweight='bold', labelpad=10)
    
    # Draw perfect pixel box outlines
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    plt.tight_layout()
    plt.savefig(out_filepath, dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="LArTPC 2D Top-Down Heatmap Generator")
    parser.add_argument('input_target', help="HDF5 file or directory")
    parser.add_argument('--events', '-e', type=int, default=100, help="Number of events to average")
    parser.add_argument('--plot_integral', action='store_true', help="Generate the Riemann Sum Heatmaps")
    parser.add_argument('--plot_peak', action='store_true', help="Generate the Peak Amplitude Heatmap")
    parser.add_argument('--plot_both', action='store_true', help="Generate all Heatmaps")
    args = parser.parse_args()

    if not args.plot_integral and not args.plot_peak and not args.plot_both:
        args.plot_both = True

    do_integral = args.plot_integral or args.plot_both
    do_peak = args.plot_peak or args.plot_both

    h5_files = []
    if os.path.isfile(args.input_target): 
        h5_files.append(args.input_target)
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5_files.extend([os.path.join(root, f) for f in files if f.endswith('.h5')])
    h5_files.sort()

    if not h5_files:
        print("[FAIL] No HDF5 files found.")
        return

    with h5py.File(h5_files[0], 'r') as f:
        config_mode = detect_config(f, h5_files[0])
        mapping = dc.get_channel_map(config_mode)

    # Reconstruct 45x45 spatial axes (SIG14 -> SIG0)
    x_chans, x_labels = build_axis_info(mapping, ["D", "E", "F"])
    y_chans, y_labels = build_axis_info(mapping, ["A", "B", "C"])

    # Only accumulate real channels (skip the -1 dummy indices)
    all_charge_chans = [ch for ch in (x_chans + y_chans) if ch != -1]
    accumulated_traces = {ch: [] for ch in all_charge_chans}
    resolution_ns = None

    events_collected = 0
    print(f"\nExtracting and averaging {args.events} events...")
    
    with tqdm(total=args.events, desc="Processing") as pbar:
        for f_path in h5_files:
            if events_collected >= args.events: break
            try:
                with h5py.File(f_path, 'r') as f:
                    if 'Board_85' in f and 'baseline_mean' not in f['Board_85']: continue
                    
                    is_single_board = 'waveforms_mV' in f
                    n_in_file = f['waveforms_mV'].shape[0] if is_single_board else f['Board_72']['waveforms'].shape[0]
                    grab_amount = min(n_in_file, args.events - events_collected)

                    for i in range(grab_amount):
                        for ch in all_charge_chans:
                            dt_ns, v = get_trace(f, i, ch)
                            if not np.all(np.isnan(v)):
                                accumulated_traces[ch].append(v)
                                if resolution_ns is None: resolution_ns = dt_ns

                        events_collected += 1
                        pbar.update(1)
            except Exception as e:
                continue

    if events_collected == 0:
        print("\n[FAIL] No valid events collected.")
        return

    print("\nCalculating Geometric Means for 45x45 Matrices...")
    
    avg_traces = {}
    for ch in all_charge_chans:
        if accumulated_traces[ch]:
            avg_traces[ch] = np.mean(accumulated_traces[ch], axis=0)
        else:
            avg_traces[ch] = np.zeros(10) 

    # Initialize matrices with NaN to represent true physical voids
    matrix_pos = np.full((len(x_chans), len(y_chans)), np.nan)
    matrix_neg = np.full((len(x_chans), len(y_chans)), np.nan)
    matrix_peak = np.full((len(x_chans), len(y_chans)), np.nan)

    # Compute orthogonal intersections
    for i, x_ch in enumerate(x_chans):
        for j, y_ch in enumerate(y_chans):
            # Leave the pixel as NaN if either strip is N/C
            if x_ch == -1 or y_ch == -1:
                continue
                
            trace_x = avg_traces[x_ch]
            trace_y = avg_traces[y_ch]
            
            if do_integral:
                matrix_pos[i, j] = gt.calc_geometric_mean_pos_integral(trace_x, trace_y, resolution_ns)
                matrix_neg[i, j] = gt.calc_geometric_mean_neg_integral(trace_x, trace_y, resolution_ns)
            if do_peak:
                matrix_peak[i, j] = gt.calc_geometric_mean_peak(trace_x, trace_y)

    base_dir = os.path.dirname(h5_files[0]) if os.path.isfile(args.input_target) else args.input_target
    save_dir = os.path.join(base_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    acq_name = os.path.basename(os.path.normpath(args.input_target))
    if not acq_name.startswith("acq"): acq_name = "Acq"

    if do_integral:
        plot_heatmap(matrix_pos, 
                     f"MiniGRAMS 45x45 Response: Positive Integral\n{events_collected} Events | Rising Edge Coupling", 
                     "Geometric Mean of Positive Integral (mV*μs)", 
                     os.path.join(save_dir, f"{acq_name}_45x45_Heatmap_PosIntegral.png"), x_labels, y_labels)

        plot_heatmap(matrix_neg, 
                     f"MiniGRAMS 45x45 Response: Negative Integral\n{events_collected} Events | Falling Edge Coupling", 
                     "Geometric Mean of Negative Integral (mV*μs)", 
                     os.path.join(save_dir, f"{acq_name}_45x45_Heatmap_NegIntegral.png"), x_labels, y_labels)

    if do_peak:
        plot_heatmap(matrix_peak, 
                     f"MiniGRAMS 45x45 Response: Peak Amplitude\n{events_collected} Events | Max Coupling", 
                     "Geometric Mean of Peak Voltage (mV)", 
                     os.path.join(save_dir, f"{acq_name}_45x45_Heatmap_Peak.png"), x_labels, y_labels)

    print(f"[SUCCESS] Heatmaps saved to: {save_dir}")

if __name__ == "__main__":
    main()