"""
Plot all waveform from CAEN generated root file
author: Jiancheng Zeng
Date: Feb 3, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import ROOT
import sys
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from tqdm import tqdm
import time
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import pandas as pd

def GRAMS_shaper_trial_1(raw_waveform, filter_order, critical_frequency, gaussian_sigma, gain, shaping_time=5, sampling_rate=125):
    b, a = butter(filter_order, critical_frequency, 'high')
    #plt.plot(raw_waveform, alpha = 0.7, label = 'original')
    high_pass_filtered = lfilter(b, a, raw_waveform)
    #plt.plot(high_pass_filtered, alpha = 0.6, label = 'high pass filtered')
    
    # This is for lifting baseline, not trustworthy now
    #high_pass_filtered = (high_pass_filtered - np.min(high_pass_filtered)) / (np.max(high_pass_filtered) - np.min(high_pass_filtered)) * np.max(high_pass_filtered)

    # Gaussian filter
    gaussian_filtered = gaussian_filter1d(high_pass_filtered, sigma=gaussian_sigma)
    """
    if(np.max(gaussian_filtered)>3.5):
        gaussian_filtered = gaussian_filtered/np.max(gaussian_filtered)
    else:
        gaussian_filtered = 0.1*gaussian_filtered/np.max(gaussian_filtered)
    """

    # This is for generate a fake normal distribution, also not trustworthy
    """
    x = np.arange(0, len(gaussian_filtered))
    location = np.array(gaussian_filtered).argmax()  # Mean (μ)
    scale = shaping_time * sampling_rate
    # Calculate the PDF of the normal distribution
    pdf = gain * np.max(gaussian_filtered) * norm.pdf(x, loc=location, scale=scale)
    """
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
new_folder = 'filtered_waveform'

# Create the full path including the new folder
full_save_path = os.path.join(save_path, new_folder)

# Create the directory if it doesn't exist
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)

print("Save raw waveform plot to: ", full_save_path)

# Ask the user to input the number of channels
num_channels = int(input("Please enter the number of channels: "))

print(f"Absolute file path: {file_path}")
print(f"Absolute save path: {save_path}")
print(f"Number of channels: {num_channels}")

"""
args = sys.argv

if len(sys.argv) == 5:
    print("file directory = ", args[1])
    filename = args[1]+args[2]
    print("file name = ", filename)
    print("number of channels = ", args[3])
    num_channels = int(args[3])
    EntryID = args[4]
else: print("*** incorrect inout arguments ***")
"""

fig = plt.figure(1)
fig.set_size_inches(8, 6)

# num_channels = 34
# NbTraces = 501
# rootfile = ROOT.TFile("./LArComboFullDrift_EmmaTile_DirtyLAr8_90Deg_34chans_05292024_dig2-usb22575_20240529110213-10.root")
rootfile = ROOT.TFile(file_path)
tree = rootfile.Get("tree")
tree.Print()

entries = tree.GetEntries()
NbTraces = int(entries/num_channels)
if(entries%num_channels != 0): print("error in number of channels or traces")
print("total entries = ",entries)
print("number of traces = ",NbTraces)

# Read the first entry to determine the number of points
tree.GetEntry(0)
# Assuming the waveform data is stored in a branch called 'waveform' or similar
# You'll need to check the actual branch name from tree.Print() output
waveform_branch_name = "waveform_samples"  # Change this to your actual branch name
waveform = getattr(tree, waveform_branch_name)
num_points = len(waveform)
print(f"Number of points per waveform: {num_points}")

# Update the time array based on the actual number of points
time = [n*0.008 for n in range(num_points)]


repeat = NbTraces

# First, load your channel mapping with flags
channel_mapping = {} 
# Read the CSV file (assuming it's named 'channel_mapping.csv')
import csv
with open('channel_mapping.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        channel_mapping[row['Channel']] = {
            'label': row['Label'],
            'flag': row['flag'],
            'comment': row['comment']
        }
    
# Function to check if a channel is valid (not NULL/nan)
def is_valid_channel(channel_name):
    if channel_name not in channel_mapping:
        return False
    channel_data = channel_mapping[channel_name]
    return not (pd.isna(channel_data['label']) and pd.isna(channel_data['extra']))


for n in tqdm(range(repeat)):
    interesting_event = False
    
    # Create figure with adjusted subplot heights
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 3])  # 1:3 height ratio (SiPM:CSP)
    
    ax_sipm = plt.subplot(gs[0])
    ax_csp = plt.subplot(gs[1], sharex=ax_sipm)  # Share x-axis

    
    for x in range(num_channels):
        tree.GetEntry(NbTraces*x+n)
        if(x != 0 and event_id != tree.event_id):
            print("different event id")
        event_id = tree.event_id
        channel = tree.channel
        if(x != 0 and timestamp != tree.timestamp):
            print("different timestamp")
        timestamp = tree.timestamp
        resolution = tree.resolution
        numberOfSamples = tree.numberOfSamples
        waveform_samples = np.array(tree.waveform_samples)

        base_label = f"Ch{channel}"
        chan_info = channel_mapping.get(base_label, {})
        label = chan_info.get('label', base_label)
        flag = chan_info.get('flag', '')

        waveform_samples = waveform_samples - np.mean(waveform_samples[:1500])

        # only apply filter to charge channels
        if flag == "CSP" :
            waveform_samples = GRAMS_shaper_trial_1(waveform_samples, filter_order=1, 
                                                    critical_frequency=0.001, gaussian_sigma=250,
                                                    shaping_time=5, gain=4, sampling_rate=125)
            
            rms = np.sqrt(np.mean(waveform_samples[:-int(num_points/3)]**2))
            #print("rms is:", rms)
            #print("max is:",np.max(waveform_samples))
            if np.max(waveform_samples) > 5 * rms:
                interesting_event = True

        # Skip NULL labels
        if label == 'NULL':
            continue
            
        # Plot based on flag
        if flag == 'SiPM':
            ax_sipm.plot(time, waveform_samples, label=label)
        elif flag == 'CSP':
            ax_csp.plot(time, waveform_samples, label=label)
        else:
            # Plot unclassified channels on CSP plot (or create third subplot if needed)
            ax_csp.plot(time, waveform_samples, label=label)
    
    # Configure SiPM plot (top)
    ax_sipm.set_title(f'SiPM Channels - Event ID {event_id}')
    ax_sipm.set_ylabel('Output [mV]')
    ax_sipm.set_xlim(0, 300)
    ax_sipm.set_ylim(-50, 10)
    ax_sipm.legend(ncol=2, loc='lower right', fontsize=10)
    #ax_sipm.grid(True)
    
    # Configure CSP plot (bottom)
    ax_csp.set_title('CSP Channels')
    ax_csp.set_xlabel('Time [μs]')
    ax_csp.set_ylabel('Output [mV]')
    ax_csp.set_xlim(0, 300)
    ax_csp.set_ylim(-300, 300)
    ax_csp.legend(ncol=6, loc='lower right', fontsize=10)
    #ax_csp.grid(True)
    
    plt.tight_layout()
    
    # Save figure based on interesting_event status
    if interesting_event:
        plt.savefig(full_save_path+'/***'+str(n).zfill(4)+"_"+str(event_id).zfill(4)+".png")
    else:
        plt.savefig(full_save_path+'/'+str(n).zfill(4)+"_"+str(event_id).zfill(4)+".png")
    
    plt.close(fig)  # Use plt.close() instead of clf() when working with figures

print("Done")