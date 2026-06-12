"""
05_Trigger_Rate_Analyzer.py
Hunts down anomalies by plotting the exact Trigger Rate (Hz) over the entire run.
- Reads only timestamps (blisteringly fast).
- Uses 10-second binning to expose Delta Spikes vs. Plateaus.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import argparse
import re
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    parser = argparse.ArgumentParser(description="Trigger Rate Anomaly Hunter")
    parser.add_argument('input_dir', help="Directory containing baseline-corrected HDF5 files")
    parser.add_argument('--resolution_sec', type=float, default=10.0, help="Histogram bin width in seconds")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")), key=natural_sort_key)
    # Filter out master integrals if they exist in the dir
    h5_files = [f for f in h5_files if "SPE_Master_Integrals" not in f]
    
    if not h5_files:
        print(f"[FAIL] No HDF5 files found in {args.input_dir}")
        return

    print(f"Hunting for Anomalies in: {args.input_dir}")
    
    # 125 MHz CAEN FPGA Clock
    clock_hz = 125000000.0 
    all_relative_seconds = []

    # Get T0
    with h5py.File(h5_files[0], 'r') as f:
        t0 = f['Board_72']['timestamps'][0]

    # Extract all timestamps
    for filepath in tqdm(h5_files, desc="Extracting Timestamps"):
        try:
            with h5py.File(filepath, 'r') as f:
                if 'Board_72' in f:
                    timestamps = f['Board_72']['timestamps'][:]
                    rel_sec = (timestamps - t0) / clock_hz
                    all_relative_seconds.extend(rel_sec)
        except Exception as e:
            pass

    all_relative_seconds = np.array(all_relative_seconds)
    total_runtime_sec = all_relative_seconds[-1]
    
    # Create the Histogram
    num_bins = int(total_runtime_sec / args.resolution_sec)
    counts, bin_edges = np.histogram(all_relative_seconds, bins=num_bins)
    
    # Convert counts per bin to Rate (Hz)
    rate_hz = counts / args.resolution_sec
    bin_centers_minutes = ((bin_edges[:-1] + bin_edges[1:]) / 2) / 60.0

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bin_centers_minutes, rate_hz, color='crimson', linewidth=1.5)
    ax.fill_between(bin_centers_minutes, rate_hz, color='crimson', alpha=0.3)
    
    # Mark the 6 Macro Time Bins to show where Bin 4 lives
    macro_bin_width = (total_runtime_sec / 6.0) / 60.0
    for i in range(1, 6):
        ax.axvline(i * macro_bin_width, color='black', linestyle='--', alpha=0.5)
        ax.text((i - 0.5) * macro_bin_width, np.max(rate_hz)*0.9, f"Bin {i-1}", ha='center', alpha=0.7, fontweight='bold')
    ax.text(5.5 * macro_bin_width, np.max(rate_hz)*0.9, "Bin 5", ha='center', alpha=0.7, fontweight='bold')

    ax.set_title(f"Global Trigger Rate vs. Time ({args.resolution_sec}-second resolution)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Run Time (Minutes)", fontweight='bold')
    ax.set_ylabel("Trigger Rate (Hz)", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    out_dir = os.path.join(args.input_dir, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "Trigger_Rate_Anomaly_Report.pdf")
    
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=200)
    plt.close(fig)

    print(f"\n[SUCCESS] Anomaly Report saved to: {out_pdf}")
    print(f"Peak Rate Detected: {np.max(rate_hz):.1f} Hz")
    print(f"Average Baseline Rate: {np.median(rate_hz):.1f} Hz")

if __name__ == "__main__":
    main()