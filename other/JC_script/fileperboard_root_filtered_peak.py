import numpy as np
import matplotlib.pyplot as plt
import ROOT
import sys
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from tqdm import tqdm
import time
import re
from Event_class import Event # required for handling Robin-style TTrees
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm


def GRAMS_shaper_trial_1(raw_waveform, b, a, gaussian_sigma, gain):
    """Apply GRAMS filter with pre-computed coefficients for speed."""
    high_pass_filtered = lfilter(b, a, raw_waveform)
    gaussian_filtered = gaussian_filter1d(high_pass_filtered, sigma=gaussian_sigma)
    return gaussian_filtered * gain

def GRAMS_high_pass(raw_waveform, filter_order, critical_frequency):
    b, a = butter(filter_order, critical_frequency, 'high')
    high_pass_filtered = lfilter(b, a, raw_waveform)
    return high_pass_filtered


# Create a PathCompleter for file path tab-completion
completer = PathCompleter()

# Ask the user to input the file path
file_path = prompt('Please enter the file path to the ROOT file: ', completer=completer)
print(f'You selected: {file_path}')

# Convert to an absolute path if a relative path is provided
file_path = os.path.abspath(file_path)

# Extract just the file name from the file path
file_name = os.path.basename(file_path)
pedestal_name = "pedestal_"+os.path.splitext(file_name)[0]+".png"
# Ask the user to input the save path for the plot
save_path = prompt('Please enter the path where you want to save the plot(default same as root file directory): ', completer=completer)
if(save_path == ''):
    save_path = os.path.dirname(file_path)
print(f'You selected: {save_path}')

# Define the new folder name
new_folder = 'filtered_waveform_TEST'

# Create the full path including the new folder
full_save_path = os.path.join(save_path, new_folder)

# Create the directory if it doesn't exist
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)

print("Save filtered waveform plot to: ", full_save_path)

# Pre-compute filter coefficients once (much faster than computing for each waveform)
b, a = butter(1, 0.001, 'high')

# Pre-define channel categories as sets (O(1) lookup instead of O(n) list)
charge_cat_x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
charge_cat_y = {34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62}

# Load the number of entries
infile = ROOT.TFile(file_path)  # Open the ROOT file
tree = infile.Get("test_tree")  # Load tree
total_events_in_acq = tree.GetEntries()
print(f"There were {total_events_in_acq} events in this acquisition.")
infile.Close() # closing file after reading total number of events 

peak_x = []
peak_y = []

for event_num in tqdm(range(total_events_in_acq)):

    hit_channels_x = 0
    hit_channels_y = 0

    # Loading up the event object
    event_obj = Event()
    event_obj.load_basics(event_num, file_path)
    event_obj.load_actives(event_num, file_path)
    event_obj.load_data(event_num, file_path)

    waveform_data = event_obj.waveform_data_2D
    
    event_id = event_obj.event_num[0]

    maximum_peak_location_x = -1
    maximum_peak_location_y = -1
    for chan in waveform_data: # for key in dict
        peak_location_x = -1
        peak_location_y = -1

        # all waveforms are baseline corrected
        base_corr_waveform = waveform_data[chan] - np.mean(waveform_data[chan][:1500]) # 1500 because of our pre-trigger time, this is hard-coded and needs to be modified for different pre-trigger times
        
        #### checking for interesting events ####

        # Check both X and Y charge channels
        if chan in charge_cat_x or chan in charge_cat_y:

            # apply GRAMS filter with pre-computed coefficients
            base_corr_waveform = GRAMS_shaper_trial_1(
                    base_corr_waveform,
                    b, a,
                    gaussian_sigma=250,
                    gain=4
            )

            rms = np.sqrt(np.mean(base_corr_waveform[:-int(event_obj.num_samples[0] / 3)] ** 2))
            if np.max(base_corr_waveform) > 5 * rms:
                peak_location = np.argmax(base_corr_waveform)
                
                if chan in charge_cat_x:
                    hit_channels_x += 1
                    if peak_location > maximum_peak_location_x:
                        maximum_peak_location_x = peak_location
                elif chan in charge_cat_y:
                    hit_channels_y += 1
                    if peak_location > maximum_peak_location_y:
                        maximum_peak_location_y = peak_location
    if maximum_peak_location_x != -1:
        peak_x.append(maximum_peak_location_x)
    if maximum_peak_location_y != -1:
        peak_y.append(maximum_peak_location_y)

# Save arrays to the same directory as input file
input_dir = os.path.dirname(file_path)
base_name = os.path.splitext(file_name)[0]
output_file = os.path.join(input_dir, f"{base_name}_peak_locations.npz")

np.savez(output_file, peak_x=np.array(peak_x), peak_y=np.array(peak_y))
print(f"Peak location arrays saved to: {output_file}")
print(f"  - peak_x: {len(peak_x)} events")
print(f"  - peak_y: {len(peak_y)} events")
print("Done")
