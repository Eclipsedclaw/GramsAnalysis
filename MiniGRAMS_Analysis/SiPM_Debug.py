import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def check_duplicates(filename, num_events):
    print(f"--- SCANNING: {filename} ---")
    
    try:
        with h5py.File(filename, 'r') as f:
            waveforms = f['waveforms_mV']
            # Hardcoded UPS SiPM indices for now to force the check
            sipm_indices = [0, 1, 2, 3] 
            
            print(f"Checking SiPM Indices: {sipm_indices}")
            
            for evt_idx in range(min(num_events, waveforms.shape[0])):
                print(f"\n[Event {evt_idx}]")
                traces = {ch: waveforms[evt_idx, ch, :] for ch in sipm_indices}
                
                # Check for Identical Arrays
                dupes = False
                for ch_a in sipm_indices:
                    for ch_b in sipm_indices:
                        if ch_a >= ch_b: continue
                        
                        if np.array_equal(traces[ch_a], traces[ch_b]):
                            print(f"  !!! CRITICAL: Ch {ch_a} == Ch {ch_b} (IDENTICAL)")
                            dupes = True
                        else:
                            # Print the sum of differences to see how close they are
                            diff = np.sum(np.abs(traces[ch_a] - traces[ch_b]))
                            print(f"  Compare {ch_a} vs {ch_b}: Delta = {diff:.1f}")
                
                if not dupes:
                    print("  > No exact duplicates.")

                # PLOT
                plt.figure(figsize=(10, 6))
                colors = ['r', 'g', 'b', 'purple']
                for i, ch in enumerate(sipm_indices):
                    # Offset traces slightly so we can see overlaps
                    offset = 0 if i == 0 else 0
                    plt.plot(traces[ch] + offset, label=f"Ch{ch}", color=colors[i], alpha=0.7, lw=1)
                
                plt.title(f"Event {evt_idx} - Duplicate Check")
                plt.xlim(1500,3000)
                plt.legend()
                # Save instead of show, since we are in terminal
                out_name = f"debug_evt{evt_idx}.png"
                plt.savefig(out_name)
                print(f"  > Saved plot to {out_name}")
                plt.close()

    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_duplicates.py <path_to_h5_file>")
    else:
        check_duplicates(sys.argv[1], 5)