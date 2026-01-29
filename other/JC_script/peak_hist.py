import numpy as np
import matplotlib.pyplot as plt
import os

# Load the data - File 1
npz_file1 = '/home/jiancheng/NAS/GAr_TPC_Runs/Run7/GArCombo5cmDrift_Run7_UPS_61ch_TPCHV50_acq6_20251124/acq6_chargetrig_peak_locations.npz'
data1 = np.load(npz_file1)
peak_x1 = data1['peak_x']
peak_y1 = data1['peak_y']

#TODO: Apply peak shift for actual data
peak_shift = 0
#peak_shift = -7.824
# Convert sample index to time in microseconds (8 ns per sample)
peak_x1_us = peak_x1 * 8 / 1000 - 16 - peak_shift
peak_y1_us = peak_y1 * 8 / 1000 - 16 - peak_shift

# Combine both arrays into one
peak_combined1 = np.concatenate([peak_x1_us, peak_y1_us])

# Filter for events > 0
peak_combined1 = peak_combined1[(peak_combined1 > -100) & (peak_combined1 < 200)]

# Load the data - File 2
npz_file2 = '/home/jiancheng/NAS/GAr_TPC_Runs/Run7/GArCombo5cmDrift_Run7_UPS_61ch_TPCHV150_acq4_20251124/acq4_chargetrigonly_peak_locations.npz' 
data2 = np.load(npz_file2)
peak_x2 = data2['peak_x']
peak_y2 = data2['peak_y']

# TODO: Apply peak shift for actual data
peak_shift = 0
#peak_shift = -7.824

# Convert sample index to time in microseconds
peak_x2_us = peak_x2 * 8 / 1000 - 16 - peak_shift
peak_y2_us = peak_y2 * 8 / 1000 - 16 - peak_shift

# Combine both arrays into one
peak_combined2 = np.concatenate([peak_x2_us, peak_y2_us])

# Filter for events > 0
peak_combined2 = peak_combined2[(peak_combined2 > -100) & (peak_combined2 < 200)]

bins = 50
counts1_raw, bin_edges1 = np.histogram(peak_combined1, bins=bins)
counts2_raw, bin_edges2 = np.histogram(peak_combined2, bins=bins)
bin_centers = (bin_edges1[:-1] + bin_edges1[1:]) / 2
bin_width = bin_edges1[1] - bin_edges1[0]

# Normalize by the number of selected events (probability, sums to 1)
counts1 = counts1_raw / counts1_raw.sum()
counts2 = counts2_raw / counts2_raw.sum()

plt.bar(bin_centers, counts1, width=bin_width*0.8, alpha=0.6, label=f'50V drift field (n={len(peak_combined1)})', color='blue', edgecolor='darkblue', linewidth=0.5)
plt.bar(bin_centers, counts2, width=bin_width*0.8, alpha=0.6, label=f'150V drift field (n={len(peak_combined2)})', color='orange', edgecolor='darkorange', linewidth=0.5)
plt.xlabel('Longest drift (μs)', fontsize=12, fontweight='bold')
plt.ylabel('Probability', fontsize=12, fontweight='bold')
plt.title('GAr Run7 - Comparison of Peak Locations for Charge Channels', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(bottom=1e-2)
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3, linestyle='--')

# Save in current folder
output_file = 'peak_locations_histogram_comparison.png'
plt.savefig(output_file, dpi=1000, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.show()