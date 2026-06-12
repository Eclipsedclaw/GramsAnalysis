"""
PedDebug.py (Forensic Noise Analyzer)
-------------------------------------
Performs a deep-dive statistical analysis on a single channel.
- Extracts raw samples from first 100 events.
- Histograms the voltage distribution.
- Fits a Gaussian to determine True RMS vs Quantization Noise.
- Saves plot to 'Plots/' directory next to input file.

Usage: python PedDebug.py <HDF5_File> -c <Channel_ID>
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend (No X11/Window needed)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
import sys

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def forensic_analysis(filepath, channel_id):
    filename = os.path.basename(filepath)
    file_dir = os.path.dirname(filepath)
    save_dir = os.path.join(file_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)

    print(f"--> INITIATING FORENSIC ANALYSIS")
    print(f"    Target:  {filename}")
    print(f"    Channel: {channel_id}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # 1. RAM-Safe Data Extraction
            # We only need ~100 events to get 3 million samples. 
            limit = 100
            if 'waveforms_mV' not in f:
                print("[ERR] 'waveforms_mV' dataset not found!")
                return

            # Check if channel exists
            n_ch = f['waveforms_mV'].shape[1]
            if channel_id >= n_ch:
                print(f"[ERR] Channel {channel_id} out of bounds (Max {n_ch-1})")
                return

            print(f"    Loading data (First {limit} events)...")
            raw_traces = f['waveforms_mV'][:limit, channel_id, :]
            
            # 2. Flatten (RAM safe operation for this size)
            all_samples = raw_traces.flatten()
            
            # 3. Basic Stats ("The Lazy Math")
            calc_mean = np.mean(all_samples)
            calc_std = np.std(all_samples)
            min_val = np.min(all_samples)
            max_val = np.max(all_samples)
            
            print(f"\n--- RAW DATA STATISTICS ---")
            print(f"    Total Samples:   {len(all_samples)}")
            print(f"    Range:           [{min_val:.4f}, {max_val:.4f}] mV")
            print(f"    Calculated Mean: {calc_mean:.4f} mV")
            print(f"    Calculated RMS:  {calc_std:.4f} mV")

            # 4. QUANTIZATION CHECK
            # We verify the resolution of the data
            unique_vals = np.unique(all_samples[:10000]) # Sample subset for speed
            if len(unique_vals) < 2:
                print("[WARN] Data is constant! (Dead channel?)")
                step_size = 0
            else:
                steps = np.diff(np.sort(unique_vals))
                step_size = np.min(steps)

            # 5. VISUALIZATION
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot 1: The Raw Trace (Zoomed in on first event)
            time_axis = np.arange(raw_traces.shape[1])
            axes[0].plot(time_axis, raw_traces[0], color='black', lw=0.5, alpha=0.9)
            axes[0].set_title(f"Raw Trace: Event 0, Ch {channel_id}")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("Voltage (mV)")
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: The Histogram & Fit
            # Use high bin count to see quantization gaps
            n_bins = 150
            counts, bins, patches = axes[1].hist(all_samples, bins=n_bins, density=False, 
                                                 color='teal', alpha=0.6, label='Raw Samples')
            
            # Gaussian Fit
            bin_centers = (bins[:-1] + bins[1:]) / 2
            try:
                # Initial Guess
                p0 = [np.max(counts), calc_mean, calc_std]
                popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=p0)
                
                fit_amp, fit_mean, fit_sigma = popt
                fit_sigma = abs(fit_sigma)
                
                # Plot Fit
                x_interval = np.linspace(bins[0], bins[-1], 1000)
                # Fixed SyntaxWarning by escaping the backslash for LaTeX sigma
                axes[1].plot(x_interval, gaussian(x_interval, *popt), 
                             color='red', lw=2, linestyle='--', 
                             label=f'Gauss Fit\n$\\sigma$={fit_sigma:.4f} mV')
                
                print(f"\n--- GAUSSIAN FIT RESULTS ---")
                print(f"    Fit Sigma (True Noise): {fit_sigma:.4f} mV")
                
            except Exception as e:
                print(f"[WARN] Fit failed: {e}")
                fit_sigma = 0.0

            # Labels and Scale
            axes[1].set_title(f"Noise Distribution Analysis")
            axes[1].set_xlabel("Voltage (mV)")
            axes[1].set_ylabel("Count (Log Scale)")
            axes[1].set_yscale('log')
            axes[1].legend()
            axes[1].grid(True, which="both", ls="-", alpha=0.2)
            
            # Add Text Box with Stats
            stats_text = (f"Calc RMS: {calc_std:.3f} mV\n"
                          f"Fit Sigma: {fit_sigma:.3f} mV\n"
                          f"Min Step: {step_size:.4f} mV")
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right', bbox=props)

            plt.tight_layout()
            
            # Save
            save_name = f"Forensic_Ch{channel_id}_{filename}.png"
            save_path = os.path.join(save_dir, save_name)
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            
            print(f"\n--> [DONE] Plot saved to:")
            print(f"    {save_path}")
            
    except Exception as e:
        print(f"[FATAL] Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forensic Noise Analyzer")
    # Position argument for file
    parser.add_argument('file', help="Path to HDF5 file")
    # Flag argument for channel
    parser.add_argument('--channel', '-c', type=int, required=True, help="Channel ID (0-60)")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.file):
        forensic_analysis(args.file, args.channel)
    else:
        print("[ERR] File not found.")