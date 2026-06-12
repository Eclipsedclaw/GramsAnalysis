"""
18_LightTriggerHeatMap.py
LArTPC Trigger Rate Spatial Mapper
- Updated for Run 7 CAEN Shuffle (MB3 Overflow is now on Board 75)
- File Limit feature for rapid prototyping.
- STRICT TRIGGER MASK: Explicitly labels excluded motherboards as "NT".
- ROBUST: Explicitly uses negative-going threshold logic.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- TACTICAL COLOR SCALE ---
HEATMAP_VMIN = 0.01   
HEATMAP_VMAX = 20.0   

# --- PHYSICAL MAPPING ENGINE ---
def get_board_and_ch(mb_num, logical_ch_0to11):
    if mb_num == 1:
        return 'Board_72', logical_ch_0to11
    elif mb_num == 2:
        return 'Board_72', 12 + logical_ch_0to11
    elif mb_num == 3:
        if logical_ch_0to11 < 8:
            return 'Board_72', 24 + logical_ch_0to11
        else:
            # CHANGED FOR RUN 7: Overflow light is now on Board 75 (Slave 2)
            return 'Board_75', 60 + (logical_ch_0to11 - 8)
    return None, None

def get_mb_label(mb_num, logical_ch_0to11):
    return f"MB{mb_num}-CH{logical_ch_0to11 + 1}"

# --- WORKER FUNCTION ---
def process_investigation(payload):
    filepath, hw_thresh, trigger_mask = payload
    local_trigger_counts = {f"{mb}_{ch}": 0 for mb in [1, 2, 3] for ch in range(12)}
            
    try:
        with h5py.File(filepath, 'r') as f:
            if 'Board_72' not in f or 'baseline_mean' not in f['Board_72']:
                return None
                
            # CHANGED FOR RUN 7: Extract Board 75 for the overflow light
            b72, b75 = f['Board_72'], f['Board_75']
            
            res_ns = b72.attrs.get('sampling_period_ns', 2.0)
            pre_trig_us = b72.attrs.get('baseline_pre_trigger_us', 16.0) 
            n_events = b72['waveforms'].shape[0]
            n_samples = b72['waveforms'].shape[2]
            
            expected_trig_idx = int((pre_trig_us * 1000) / res_ns)
            gate_margin = int(1000 / res_ns) 
            gate_start = max(0, expected_trig_idx - gate_margin)
            gate_end = min(n_samples, expected_trig_idx + gate_margin)
            
            # Explicitly force the threshold to be negative
            neg_threshold = -1.0 * abs(hw_thresh)

            for i in range(n_events):
                candidates = []
                w72, w75 = b72['waveforms'][i], b75['waveforms'][i]
                bl72, bl75 = b72['baseline_mean'][i], b75['baseline_mean'][i]
                
                # Skip partial/orphaned events
                if np.isnan(w72[0,0]) or np.isnan(w75[0,0]): continue

                for mb in trigger_mask:
                    for ch in range(12):
                        board_name, hdf5_ch = get_board_and_ch(mb, ch)
                        
                        # Route the extraction dynamically
                        if board_name == 'Board_72':
                            raw, bl = w72[hdf5_ch], bl72[hdf5_ch]
                        else:
                            raw, bl = w75[hdf5_ch], bl75[hdf5_ch]
                            
                        trace = raw - bl
                        
                        gate_slice = trace[gate_start:gate_end]
                        
                        # True negative-going hunt
                        crossings = np.where(gate_slice < neg_threshold)[0]
                        if len(crossings) > 0:
                            candidates.append((f"{mb}_{ch}", crossings[0]))
                
                # Crown the winner
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    winner_key = candidates[0][0]
                    local_trigger_counts[winner_key] += 1
                            
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return None

    return local_trigger_counts

# --- PLOTTING ---
def plot_trigger_heatmap(trigger_data, acq_name, total_evts, duration, trigger_mask, hw_thresh, save_path):
    grid_data = np.zeros((6, 6))
    mb_stack = [{'name': 'MB3 (Top)', 'mb': 3}, {'name': 'MB1 (Mid)', 'mb': 1}, {'name': 'MB2 (Bot)', 'mb': 2}]
    
    row_cursor = 0
    for layer in mb_stack:
        mb = layer['mb']
        for col, ch in enumerate([0, 2, 4, 6, 8, 10]):
            rate = trigger_data.get(f"{mb}_{ch}", 0) / duration
            grid_data[row_cursor, col] = rate if mb in trigger_mask else np.nan
        for col, ch in enumerate([1, 3, 5, 7, 9, 11]):
            rate = trigger_data.get(f"{mb}_{ch}", 0) / duration
            grid_data[row_cursor + 1, col] = rate if mb in trigger_mask else np.nan
        row_cursor += 2

    fig, ax = plt.subplots(figsize=(10, 10))
    
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('whitesmoke')
    
    im = ax.imshow(grid_data, cmap=cmap, interpolation='nearest', norm=LogNorm(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX))
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Trigger Rate (Hz)', rotation=270, labelpad=20, fontsize=12)
    
    for pos in [1.5, 3.5]:
        ax.axhline(y=pos, color='black', linewidth=3)
        ax.axvline(x=pos, color='black', linewidth=3)

    row_cursor = 0
    for layer in mb_stack:
        mb = layer['mb']
        for col, ch in enumerate([0, 2, 4, 6, 8, 10]):
            if mb not in trigger_mask:
                label = f"{get_mb_label(mb, ch)}\nNT"
                ax.text(col, row_cursor, label, ha="center", va="center", color="gray", fontsize=9, fontweight='bold')
            else:
                rate = trigger_data.get(f"{mb}_{ch}", 0) / duration
                text_color = 'black' if rate > np.sqrt(HEATMAP_VMIN * HEATMAP_VMAX) else 'white'
                label = f"{get_mb_label(mb, ch)}\n{rate:.2f} Hz"
                ax.text(col, row_cursor, label, ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
        
        for col, ch in enumerate([1, 3, 5, 7, 9, 11]):
            if mb not in trigger_mask:
                label = f"{get_mb_label(mb, ch)}\nNT"
                ax.text(col, row_cursor+1, label, ha="center", va="center", color="gray", fontsize=9, fontweight='bold')
            else:
                rate = trigger_data.get(f"{mb}_{ch}", 0) / duration
                text_color = 'black' if rate > np.sqrt(HEATMAP_VMIN * HEATMAP_VMAX) else 'white'
                label = f"{get_mb_label(mb, ch)}\n{rate:.2f} Hz"
                ax.text(col, row_cursor+1, label, ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
        row_cursor += 2

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Trigger Rate Heatmap (< -{abs(hw_thresh)}mV) | {acq_name}\nTop: MB3 | Bot: MB2 | Duration: {duration:.1f}s", fontsize=16)
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "Trigger_Heatmap_Hz.png"), dpi=200, bbox_inches='tight')

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="LArTPC Multi-Board Heatmap Generator")
    parser.add_argument('input_target', help="Directory containing synced HDF5 files")
    parser.add_argument('--threshold', '-t', type=float, default=75.0, help="Hardware Trigger Threshold magnitude in mV (e.g. 75.0)")
    parser.add_argument('--duration', '-d', type=float, required=True, help="Total acquisition duration in seconds (required)")
    parser.add_argument('--mask', '-m', nargs='+', type=int, default=[1], help="Motherboards active in hardware trigger (e.g., -m 1)")
    parser.add_argument('--file_limit', '-f', type=int, default=0, help="Max number of HDF5 files to process (0 = all files)")
    parser.add_argument('--workers', '-w', type=int, default=8, help="Max workers")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_target, "*.h5")))
    if not files:
        print("No HDF5 files found.")
        return

    if args.file_limit > 0:
        files = files[:args.file_limit]
        print(f"[NOTE] Processing limited to the first {args.file_limit} files.")

    acq_name = os.path.basename(os.path.normpath(args.input_target))
    save_path = os.path.join(args.input_target, 'Plots')
    
    print(f"=== Trigger Heatmap Generator (Run 7) ===")
    print(f"Acquisition: {acq_name}")
    print(f"Files:       {len(files)}")
    print(f"Threshold:   <-{abs(args.threshold)} mV (Negative-going)")
    print(f"Duration:    {args.duration} s")
    print(f"Active Mask: MB {args.mask}")
    print("-" * 45)
    
    agg_triggers = {f"{mb}_{ch}": 0 for mb in [1, 2, 3] for ch in range(12)}
    total_evts = 0
    
    tasks = [(f, args.threshold, args.mask) for f in files]
    
    with Pool(min(args.workers, len(tasks))) as p:
        for r in tqdm(p.imap_unordered(process_investigation, tasks), total=len(tasks), desc="Processing Files", unit="file"):
            if r is None: continue
            for key, count in r.items(): 
                agg_triggers[key] += count
                total_evts += count

    print("\nGenerating Spatial Map...")
    plot_trigger_heatmap(agg_triggers, acq_name, total_evts, args.duration, args.mask, args.threshold, save_path)
    print(f"Mission Complete. Heatmap saved to {save_path}")

if __name__ == "__main__":
    main()