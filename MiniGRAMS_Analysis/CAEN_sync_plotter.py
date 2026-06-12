import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def find_threshold_crossing(time_array, waveform, threshold=100.0):
    """
    Finds the exact sub-sample time a waveform crosses a threshold 
    (rising edge) using linear interpolation.
    """
    # Find the first index where the waveform goes from below to above the threshold
    crossings = np.where((waveform[:-1] < threshold) & (waveform[1:] >= threshold))[0]
    
    if len(crossings) == 0:
        return -1  # No crossing found
        
    idx = crossings[0]
    
    # Linear interpolation for sub-sample precision
    t1, t2 = time_array[idx], time_array[idx+1]
    y1, y2 = waveform[idx], waveform[idx+1]
    
    # t = t1 + (t2 - t1) * (threshold - y1) / (y2 - y1)
    t_cross = t1 + (t2 - t1) * (threshold - y1) / (y2 - y1)
    
    return t_cross

def analyze_and_plot(h5_file, ch_master, ch_slave1, ch_slave2, event=0):
    print(f"\n--- INITIATING DELAY ANALYSIS ON EVENT {event} ---")
    
    with h5py.File(h5_file, 'r') as hf:
        w_72 = hf['Board_72/waveforms'][event, ch_master, :]
        dt_72 = hf['Board_72'].attrs['sampling_period_ns']
        
        w_75 = hf['Board_75/waveforms'][event, ch_slave1, :]
        dt_75 = hf['Board_75'].attrs['sampling_period_ns']
        
        w_85 = hf['Board_85/waveforms'][event, ch_slave2, :]
        dt_85 = hf['Board_85'].attrs['sampling_period_ns']
        
    # Generate Time Arrays (in nanoseconds)
    t_72 = np.arange(len(w_72)) * dt_72
    t_75 = np.arange(len(w_75)) * dt_75
    t_85 = np.arange(len(w_85)) * dt_85
    
    # Calculate exact 100 mV crossing times
    trigger_threshold = 100.0
    time_72 = find_threshold_crossing(t_72, w_72, trigger_threshold)
    time_75 = find_threshold_crossing(t_75, w_75, trigger_threshold)
    time_85 = find_threshold_crossing(t_85, w_85, trigger_threshold)
    
    # Calculate Delays relative to the Master
    delay_75 = time_75 - time_72
    delay_85 = time_85 - time_72
    
    print(f"[INTEL] Master (72, V2730) 100mV Crossing : {time_72:.2f} ns")
    print(f"[INTEL] Slave 1 (75, V2740B) 100mV Crossing: {time_75:.2f} ns -> Delay = {delay_75:.2f} ns")
    print(f"[INTEL] Slave 2 (85, V2740B) 100mV Crossing: {time_85:.2f} ns -> Delay = {delay_85:.2f} ns")

    # --- Stacked Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Plot Master
    ax1.plot(t_72, w_72, color='blue', label=f'Board 72 (Master, V2730) - Ch {ch_master}')
    if time_72 > 0: 
        ax1.axvline(time_72, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Amplitude (mV)')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot Slave 1
    ax2.plot(t_75, w_75, color='orange', label=f'Board 75 (Slave 1, V2740B) - Ch {ch_slave1}')
    if time_75 > 0: 
        ax2.axvline(time_75, color='red', linestyle='--', alpha=0.7)
        ax2.text(time_75 + 15, 0, f"Delay: {delay_75:.2f} ns", color='red', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    ax2.set_ylabel('Amplitude (mV)')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    # Plot Slave 2
    ax3.plot(t_85, w_85, color='green', label=f'Board 85 (Slave 2, V2740B) - Ch {ch_slave2}')
    if time_85 > 0: 
        ax3.axvline(time_85, color='red', linestyle='--', alpha=0.7)
        ax3.text(time_85 + 15, 0, f"Delay: {delay_85:.2f} ns", color='red', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    ax3.set_xlabel('Time (nanoseconds)')
    ax3.set_ylabel('Amplitude (mV)')
    ax3.legend(loc='lower right')
    ax3.grid(True)
    
    # Zoom window: 200ns before master trigger, 800ns after
    if time_72 > 0:
        ax3.set_xlim(time_72 - 200, time_72 + 800)
    
    plt.suptitle("Hardware Trigger Sync - Propagation Delay Measurement")
    plt.tight_layout()
    plt.savefig('trigger_delay_measurement_v3.png')
    print("\n[SUCCESS] Plot saved to 'trigger_delay_measurement_v3.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file', help="Path to the generated HDF5 file")
    parser.add_argument('--ch_m', type=int, required=True, help="Channel index of square wave on Master (72)")
    parser.add_argument('--ch_s1', type=int, required=True, help="Channel index of square wave on Slave 1 (75)")
    parser.add_argument('--ch_s2', type=int, required=True, help="Channel index of square wave on Slave 2 (85)")
    args = parser.parse_args()
    
    analyze_and_plot(args.h5_file, args.ch_m, args.ch_s1, args.ch_s2)