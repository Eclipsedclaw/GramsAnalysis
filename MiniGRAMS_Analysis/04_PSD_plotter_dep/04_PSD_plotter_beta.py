"""
04_PSD_plotter_univ.py
MiniGRAMS LArTPC Analysis Pipeline
----------------------------------
Generates aggregate PSD plots per Acquisition.
Updates: 
- MINIG CONFIG: Supports the multi-board Event Builder output (72, 85, 75).
- DUAL IMPEDANCE CORRECTION: Corrects Light (50 Ohm series -> 2.0x) and Charge (110 Ohm series -> 3.2x) independently.
- Independent Frequency Axes: CAEN 72 (500MS/s) and CAENs 75/85 (125MS/s) plotted correctly.
- NaN Immunity: Bypasses dropped synchronization traces.
- Telemetry: Integrated tqdm progress bars for channel processing.
- Visuals: Legends for dense charge panels moved completely outside the plot area.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
import time
from multiprocessing import Pool
from tqdm import tqdm
import detector_config as dc
import grams_tools as gt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_EVENTS = 100
BATCH_SIZE = 10

def detect_config(h5_file, filename):
    if 'run_config' in h5_file.attrs:
        return h5_file.attrs['run_config']
    return 'MINIG' 

def get_impedance_factor(global_ch, config_mode, ch_map):
    """
    Dynamically checks the physical map.
    Light (VUV/VIS): 50 Ohm series -> 2.0x
    Charge (Banks A-F): 110 Ohm series -> 3.2x
    """
    if config_mode == 'MINIG':
        light_chans = ch_map.get('VUV', []) + ch_map.get('VIS', [])
        if global_ch in light_chans:
            return 2.0 
        else:
            return 3.2 
    return 1.0

def get_board_info(global_ch):
    """Routes the absolute channel index to the correct hardware board and local index."""
    if global_ch < 32: return 'Board_72', global_ch
    elif global_ch < 96: return 'Board_85', global_ch - 32
    else: return 'Board_75', global_ch - 96

def process_acquisition_task(payload):
    acq_path, h5_files, target_events = payload
    acq_name = os.path.basename(acq_path)
    print(f"\n--> [START] Analyzing: {acq_name}")
    
    h5_files.sort()
    channel_data = {}
    board_freq_axis_mhz = {}
    board_f_dc_mhz = {}
    events_collected = 0
    config_mode = 'MINIG'
    representative_filename = "Unknown.h5"
    
    start_t = time.time()

    # --- DATA EXTRACTION ---
    for filepath in h5_files:
        if events_collected >= target_events: break
        try:
            with h5py.File(filepath, 'r') as f:
                filename = os.path.basename(filepath)
                if events_collected == 0:
                    config_mode = detect_config(f, filename)
                    representative_filename = filename

                ch_map = dc.get_channel_map(config_mode)
                active_channels = []
                for k in ['VUV', 'VIS', 'Charge_A', 'Charge_B', 'Charge_C', 'Charge_D', 'Charge_E', 'Charge_F']:
                    if k in ch_map: active_channels.extend(ch_map[k])
                active_channels = sorted(list(set(active_channels)))
                
                is_single_board = 'waveforms_mV' in f
                n_in_file = f['waveforms_mV'].shape[0] if is_single_board else f['Board_72']['waveforms'].shape[0]
                
                needed = target_events - events_collected
                grab_total = min(needed, n_in_file)

                with tqdm(total=len(active_channels), desc=f"Extracting PSDs ({grab_total} evts)", unit="ch") as pbar:
                    for ch in active_channels:
                        impedance_k = get_impedance_factor(ch, config_mode, ch_map)
                        
                        if is_single_board:
                            res_ns = f.attrs.get('resolution_ns', 8.0)
                            dset = f['waveforms_mV']
                            local_ch = ch
                            b_name = 'Single'
                        else:
                            b_name, local_ch = get_board_info(ch)
                            if b_name not in f: 
                                pbar.update(1)
                                continue
                            res_ns = f[b_name].attrs['sampling_period_ns']
                            dset = f[b_name]['waveforms']
                            
                        if local_ch >= dset.shape[1]: 
                            pbar.update(1)
                            continue
                            
                        n_samples = dset.shape[2]
                        
                        if b_name not in board_freq_axis_mhz:
                            freq_axis = gt.get_freq_axis(n_samples, res_ns)
                            board_freq_axis_mhz[b_name] = freq_axis / 1e6
                            window_sec = n_samples * res_ns * 1e-9
                            board_f_dc_mhz[b_name] = (1.0 / window_sec) / 1e6 if window_sec > 0 else 0.003
                        
                        if ch not in channel_data: channel_data[ch] = []
                        
                        for start_idx in range(0, grab_total, BATCH_SIZE):
                            end_idx = min(start_idx + BATCH_SIZE, grab_total)
                            batch_waves = dset[start_idx:end_idx, local_ch, :].astype(np.float32)
                            
                            valid_waves = [w for w in batch_waves if not np.any(np.isnan(w))]
                            if not valid_waves: continue
                            
                            valid_waves = np.array(valid_waves) * impedance_k
                            
                            for w in valid_waves:
                                val = gt.compute_psd_trace(w, res_ns, window_type='hanning')
                                channel_data[ch].append(val)
                        
                        pbar.update(1)
                events_collected += grab_total
        except Exception as e:
            print(f"    [WARN] Skipping {filepath}: {e}")
            continue

    if events_collected == 0: return

    # --- AVERAGING ---
    avg_psd_data = {}
    for ch, traces in channel_data.items():
        if traces:
            avg_psd_data[ch] = np.mean(np.vstack(traces), axis=0)

    # --- PLOTTING ---
    save_dir = os.path.join(os.path.dirname(acq_path) if os.path.isfile(acq_path) else acq_path, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        ch_map = dc.get_channel_map(config_mode)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        x_chans = ch_map.get("Charge_A", []) + ch_map.get("Charge_B", []) + ch_map.get("Charge_C", [])
        y_chans = ch_map.get("Charge_D", []) + ch_map.get("Charge_E", []) + ch_map.get("Charge_F", [])
        
        layout = [
            (axes[0, 0], 'VUV SiPMs', ch_map.get('VUV', [])),
            (axes[0, 1], 'VIS SiPMs', ch_map.get('VIS', [])),
            (axes[1, 0], 'Charge X-Direction (Banks A, B, C)', x_chans),
            (axes[1, 1], 'Charge Y-Direction (Banks D, E, F)', y_chans),
        ]

        cmap = matplotlib.colormaps['tab20']
        
        for ax, title, channels in layout:
            if not channels:
                ax.text(0.5, 0.5, "Not Installed / Empty", ha='center', va='center', color='grey')
                ax.set_title(title, color='grey')
                continue
            
            colors = cmap(np.linspace(0, 1, max(len(channels), 1)))
            plotted_b_name = None
            
            for i, ch in enumerate(channels):
                if ch in avg_psd_data:
                    b_name = 'Single' if 'Single' in board_freq_axis_mhz else get_board_info(ch)[0]
                    plotted_b_name = b_name
                    f_axis = board_freq_axis_mhz[b_name]
                    label_str = f"Ch {ch}"
                    ax.semilogx(f_axis, avg_psd_data[ch], color=colors[i % len(colors)], alpha=0.6, lw=1.0, label=label_str)
            
            if plotted_b_name and board_f_dc_mhz[plotted_b_name] > 0:
                f_min = board_f_dc_mhz[plotted_b_name]
                ax.axvline(x=f_min, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax.text(f_min * 1.2, ax.get_ylim()[0] + 10, r"$f_{min}$", color='red', fontsize=11, ha='left', va='bottom')

            ax.set_title(f"{title} (N={len([c for c in channels if c in avg_psd_data])})", fontweight='bold')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
            # Anchor legends outside to the right
            if len(channels) > 10: 
                ax.legend(fontsize='x-small', ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            else: 
                ax.legend(fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("PSD (dBm/Hz)")

        fig.suptitle(fr"PSD Analysis | Config: {config_mode}" + "\n" + f"Acquisition: {representative_filename}", fontsize=14, fontweight='bold')
        
        # Squeeze the layout to leave room for the external legends
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plot_name = f"{acq_name.replace('.h5', '')}_PSD_univ.png"
        
        plt.savefig(os.path.join(save_dir, plot_name), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        total_time = time.time() - start_t
        print(f"--> [SUCCESS] Saved {plot_name} in {total_time:.1f}s\n")

    except Exception as e:
        print(f"    [FAIL] Plotting error: {e}")

def main():
    parser = argparse.ArgumentParser(description="LArTPC Universal PSD Plotter")
    parser.add_argument('input_target', help="Input file or directory containing HDF5 files")
    parser.add_argument('--events', '-e', type=int, default=DEFAULT_EVENTS, help="Number of events to average")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Max concurrent processes")
    args = parser.parse_args()
    
    acq_groups = {}
    if os.path.isfile(args.input_target):
        acq_groups[args.input_target] = [args.input_target]
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5s = [os.path.join(root, f) for f in files if f.endswith('.h5')]
            if h5s: acq_groups[root] = h5s
            
    tasks = [(p, f, args.events) for p, f in acq_groups.items()]
    if tasks:
        with Pool(min(args.workers, len(tasks))) as p: 
            p.map(process_acquisition_task, tasks)

if __name__ == "__main__":
    main()