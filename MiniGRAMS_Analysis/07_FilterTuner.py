import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
import os
import detector_config as dc

# --- STYLE ---
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def get_active_channels(waves, baselines, ch_map):
    """Identifies the top 4 charge channels with the most activity."""
    integrals = []
    ch_indices = []
    
    for ch in ch_map['Charge']:
        if ch < waves.shape[0]:
            trace = waves[ch] - baselines[ch]
            integ = np.sum(np.abs(trace))
            integrals.append(integ)
            ch_indices.append(ch)
            
    sorted_pairs = sorted(zip(integrals, ch_indices), key=lambda x: x[0], reverse=True)
    return [p[1] for p in sorted_pairs[:4]]

def process_and_plot(filename, event_id, window_ns, poly_order):
    print(f"--- FILTER TUNER v2: Event {event_id} ---")
    print(f"Target Window: {window_ns} ns | Poly Order: {poly_order}")
    
    with h5py.File(filename, 'r') as f:
        run_config = f.attrs.get('run_config', 'UPS')
        res_ns = f.attrs.get('resolution_ns', 8.0)
        ch_map = dc.get_channel_map(run_config)
        
        # Calculate Window in Samples
        win_samples = int(window_ns / res_ns)
        if win_samples % 2 == 0: win_samples += 1 
        if win_samples < 3: win_samples = 3
        
        print(f"Resolution: {res_ns} ns")
        print(f"SG Filter: {win_samples} samples")
        
        waves = f['waveforms_mV'][event_id]
        if 'baseline_mean_mV' in f:
            baselines = f['baseline_mean_mV'][event_id]
        else:
            baselines = np.mean(waves[:, :20], axis=1)
            
        active_chs = get_active_channels(waves, baselines, ch_map)
        t_axis = np.arange(waves.shape[1]) * (res_ns / 1000.0)
        
        # --- PLOTTING (4 Rows x 2 Cols) ---
        fig, axes = plt.subplots(len(active_chs), 2, figsize=(16, 3*len(active_chs)), sharex='col')
        if len(active_chs) == 1: axes = np.array([axes]) # Handle single channel case
        
        for i, ch in enumerate(active_chs):
            # Row i, Left Col (Voltage)
            ax_volts = axes[i, 0]
            # Row i, Right Col (Current)
            ax_curr = axes[i, 1]
            
            raw = waves[ch] - baselines[ch]
            
            # Filter
            filtered = savgol_filter(raw, window_length=win_samples, polyorder=poly_order)
            
            # Derivative (scaled to be visible? No, arbitrary units are fine for tuning)
            # We multiply by 1000 just to make the numbers on the Y-axis nicer (mV/us roughly)
            deriv = np.gradient(filtered) * 1000 
            
            # --- LEFT PLOT: VOLTAGE ---
            ax_volts.plot(t_axis, raw, color='gray', alpha=0.4, lw=1, label='Raw')
            ax_volts.plot(t_axis, filtered, color='tab:blue', lw=1.5, label=f'SG Filter')
            ax_volts.set_ylabel(f"Ch{ch}\nVoltage (mV)", fontweight='bold')
            ax_volts.legend(loc='upper right', fontsize='x-small')
            
            # --- RIGHT PLOT: DERIVATIVE ---
            ax_curr.plot(t_axis, deriv, color='tab:orange', lw=1.0, label='Derivative')
            ax_curr.axhline(0, color='black', alpha=0.5, lw=0.5)
            ax_curr.set_ylabel("Current (arb)", fontweight='bold')
            
            # Mark Trigger on both
            ax_volts.axvline(16.0, color='green', ls='--', alpha=0.5)
            ax_curr.axvline(16.0, color='green', ls='--', alpha=0.5)

            if i == 0:
                ax_volts.set_title(f"Event {event_id}: Voltage (Window={window_ns}ns)")
                ax_curr.set_title("Derivative (Slope)")

        axes[-1, 0].set_xlabel("Time (us)")
        axes[-1, 1].set_xlabel("Time (us)")
        
        plt.tight_layout()
        out_name = f"TunerV2_Evt{event_id}_W{int(window_ns)}.png"
        plt.savefig(out_name)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--event', '-e', type=int, default=96)
    parser.add_argument('--window', '-w', type=float, default=2000.0, help="Try 1000, 2000, 5000")
    parser.add_argument('--poly', '-p', type=int, default=2, help="Keep low (2 or 3)")
    args = parser.parse_args()
    
    process_and_plot(args.filename, args.event, args.window, args.poly)