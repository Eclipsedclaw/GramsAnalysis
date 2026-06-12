import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import os
import sys
import detector_config as dc

def analyze_correlations_aggregate(input_dir, max_events='all'):
    # 1. Setup Output Directory
    save_dir = os.path.join(input_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Find all HDF5 files
    files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    files.sort() # Ensure chronological order if filenames have timestamps
    
    if not files:
        print(f"[ERROR] No HDF5 files found in {input_dir}")
        return

    print(f"--- ANALYZING ACQUISITION: {os.path.basename(input_dir)} ---")
    print(f"Found {len(files)} files to aggregate.")

    # 3. Initialize Global Data Containers
    global_vuv = []
    global_vis = []
    global_charge = []
    global_light_total = []
    
    events_processed = 0
    target_limit = float('inf') if max_events == 'all' else int(max_events)
    
    # Metadata placeholders (will be set by first file)
    run_config = None
    ch_map = None
    res_ns = 2.0 

    # 4. Loop Through Files
    for filename in files:
        if events_processed >= target_limit:
            break
            
        filepath = os.path.join(input_dir, filename)
        
        try:
            with h5py.File(filepath, 'r') as f:
                # --- METADATA (Once) ---
                if run_config is None:
                    run_config = f.attrs.get('run_config', 'UPS')
                    res_ns = f.attrs.get('resolution_ns', 2.0)
                    ch_map = dc.get_channel_map(run_config)
                    print(f"  > Config Detected: {run_config}")
                    print(f"  > Resolution: {res_ns} ns")
                    print(f"  > Channels: VUV={ch_map['VUV']}, VIS={ch_map['VIS']}, Charge={ch_map['Charge']}")

                # --- LOAD DATA ---
                waveforms = f['waveforms_mV']
                n_in_file = waveforms.shape[0]
                
                # Check for Baselines
                if 'baseline_mean_mV' in f:
                    baselines = f['baseline_mean_mV']
                    use_stored_baselines = True
                else:
                    print(f"  [WARN] {filename}: Missing baselines. Using first-20-sample subtraction.")
                    use_stored_baselines = False

                # Calculate how many events to grab from this file
                needed = target_limit - events_processed
                to_process = min(n_in_file, needed)
                
                # --- EVENT LOOP ---
                # processing in memory-safe chunks could be added here, 
                # but for simplicity we iterate row by row or small blocks.
                for i in range(to_process):
                    # A. RAW DATA
                    waves = waveforms[i] # (n_ch, n_samples)
                    
                    # B. BASELINE CORRECTION
                    if use_stored_baselines:
                        bls = baselines[i]
                        corrected_waves = waves - bls[:, np.newaxis]
                    else:
                        bls = np.mean(waves[:, :20], axis=1)
                        corrected_waves = waves - bls[:, np.newaxis]

                    # C. INVERT SiPMs
                    # Essential for correct integration area of negative pulses
                    for ch in ch_map['VUV'] + ch_map['VIS']:
                        corrected_waves[ch] *= -1.0

                    # D. INTEGRATE (Riemann Sum)
                    # Unit: mV * ns
                    
                    # VUV
                    q_vuv = 0.0
                    if len(ch_map['VUV']) > 0:
                        q_vuv = np.sum(corrected_waves[ch_map['VUV'], :]) * res_ns

                    # VIS
                    q_vis = 0.0
                    if len(ch_map['VIS']) > 0:
                        q_vis = np.sum(corrected_waves[ch_map['VIS'], :]) * res_ns

                    # Charge (No inversion usually)
                    q_charge = 0.0
                    if len(ch_map['Charge']) > 0:
                        q_charge = np.sum(corrected_waves[ch_map['Charge'], :]) * res_ns

                    # Append to Globals
                    global_vuv.append(q_vuv)
                    global_vis.append(q_vis)
                    global_charge.append(q_charge)
                    global_light_total.append(q_vuv + q_vis)

                events_processed += to_process
                print(f"  > Processed {to_process} events from {filename} (Total: {events_processed})")

        except Exception as e:
            print(f"  [FAIL] Error reading {filename}: {e}")
            continue

    # 5. Plotting (Only if we have data)
    if events_processed == 0:
        print("[ERROR] No events processed.")
        return

    print("--- GENERATING PLOTS ---")
    
    # Convert to numpy for plotting
    arr_vuv = np.array(global_vuv)
    arr_vis = np.array(global_vis)
    arr_charge = np.array(global_charge)
    arr_light_total = np.array(global_light_total)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Settings
    bins = 100
    cmap = 'viridis' # User requested viridis
    norm = LogNorm() 

    # Plot 1: VUV vs VIS
    h1 = axes[0, 0].hist2d(arr_vuv, arr_vis, bins=bins, cmap=cmap, norm=norm)
    axes[0, 0].set_title("1. VUV vs VIS")
    axes[0, 0].set_xlabel("Integrated VUV (mV*ns)")
    axes[0, 0].set_ylabel("Integrated VIS (mV*ns)")
    fig.colorbar(h1[3], ax=axes[0, 0])

    # Plot 2: Charge vs VUV
    h2 = axes[0, 1].hist2d(arr_charge, arr_vuv, bins=bins, cmap=cmap, norm=norm)
    axes[0, 1].set_title("2. Charge vs VUV")
    axes[0, 1].set_xlabel("Integrated Charge (mV*ns)")
    axes[0, 1].set_ylabel("Integrated VUV (mV*ns)")
    fig.colorbar(h2[3], ax=axes[0, 1])

    # Plot 3: Charge vs VIS
    h3 = axes[1, 0].hist2d(arr_charge, arr_vis, bins=bins, cmap=cmap, norm=norm)
    axes[1, 0].set_title("3. Charge vs VIS")
    axes[1, 0].set_xlabel("Integrated Charge (mV*ns)")
    axes[1, 0].set_ylabel("Integrated VIS (mV*ns)")
    fig.colorbar(h3[3], ax=axes[1, 0])

    # Plot 4: Charge vs Total Light
    h4 = axes[1, 1].hist2d(arr_charge, arr_light_total, bins=bins, cmap=cmap, norm=norm)
    axes[1, 1].set_title("4. Charge vs Total Light")
    axes[1, 1].set_xlabel("Integrated Charge (mV*ns)")
    axes[1, 1].set_ylabel("Total Light (mV*ns)")
    fig.colorbar(h4[3], ax=axes[1, 1])

    acq_name = os.path.basename(os.path.normpath(input_dir))
    plt.suptitle(f"Acquisition Correlations: {acq_name}\nEvents: {events_processed}", fontsize=15)
    plt.tight_layout()
    
    out_name = f"Correlations_Aggregated_{acq_name}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path)
    print(f"--- COMPLETE ---")
    print(f"Saved plot to: {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help="Directory containing HDF5 files")
    parser.add_argument('--events', default='all', help="Total events to process across all files")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print("Error: The provided path is not a directory.")
        sys.exit(1)
        
    analyze_correlations_aggregate(args.input_dir, args.events)