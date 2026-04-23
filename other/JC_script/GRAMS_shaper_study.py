import ROOT
import matplotlib.pyplot as plt
import numpy as np
from array import array
import pandas as pd
import os
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d

# Load single event from ROOT file
def single_event(file_address, entry_index, num_channels):
    file = ROOT.TFile(file_address)
    mytree = file.Get("tree")
    mytree.SetBranchStatus("waveform_samples", 1)
    raw_wf = array('f', [0]*(len(mytree.waveform_samples)-1))
    mytree.SetBranchAddress("waveform_samples", raw_wf)
    
    num_events = int(mytree.GetEntries()/num_channels)
    df = pd.DataFrame()
    
    for i in range(num_channels):
        mytree.GetEntry(entry_index + i*num_events)
        wfarray = raw_wf
        column_name = f"ch{mytree.channel}"
        df[column_name] = wfarray
    
    return df

# Baseline correction
def baseline_correction(baseline_array, pretrigger=1875):
    avg = np.average(baseline_array[0:pretrigger])
    corrected = np.array(baseline_array) - avg
    return corrected

# GRAMS shaper filter
def GRAMS_shaper_trial_1(raw_waveform, filter_order, critical_frequency, gaussian_sigma, gain):
    b, a = butter(filter_order, critical_frequency, 'high')
    high_pass_filtered = lfilter(b, a, raw_waveform)
    gaussian_filtered = gaussian_filter1d(high_pass_filtered, sigma=gaussian_sigma)
    gaussian_filtered = gaussian_filtered * gain
    
    # Align peaks using smoothed waveform to reduce noise sensitivity
    smoothed_original = gaussian_filter1d(raw_waveform, sigma=50)
    original_peak_idx = np.argmax(smoothed_original)
    filtered_peak_idx = np.argmax(gaussian_filtered)
    peak_shift = original_peak_idx - filtered_peak_idx
    gaussian_filtered = np.roll(gaussian_filtered, peak_shift)
    
    return gaussian_filtered, high_pass_filtered

# Main script
entry_ID = 9483
file_address = "/home/jiancheng/NAS/LAr_TPC_runs/Run26/LArComboFullDrift_Run26_EmmaTile_FilteredLAr0_90Deg_AnodeMeshV2_34chans_09052024/LArComboFullDrift_Run26_EmmaTile_FilteredLAr0_90Deg_AnodeMeshV2_34chans_09052024_dig2-usb51054_20240905161846-09.root"

# Load event data
event_data = single_event(file_address=file_address, entry_index=entry_ID, num_channels=34)

# Process and plot single channel (ch57)
baseline_array = baseline_correction(baseline_array=event_data['ch57'].to_numpy())
single_filtered_channel, high_pass_result = GRAMS_shaper_trial_1(baseline_array, filter_order=1, 
                                                critical_frequency=0.001, gaussian_sigma=250, gain=4)

# Create plot
plt.figure(figsize=(14, 8))
plt.plot(baseline_array, label='raw waveform')
plt.plot(high_pass_result, c='orange', label='high-pass filtered')
plt.plot(single_filtered_channel, c='red', label='after GRAMS filter')
plt.ylim(-30, 60)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (mV)')
plt.title(f'Event {entry_ID} - Channel 57')
plt.legend()
plt.grid()

# Save plot locally
output_path = f"event_{entry_ID}_ch57.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.abspath(output_path)}")
plt.show()