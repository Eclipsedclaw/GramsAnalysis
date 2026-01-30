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
new_folder = 'raw_waveform_TEST'

# Create the full path including the new folder
full_save_path = os.path.join(save_path, new_folder)

# Create the directory if it doesn't exist
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)

print("Save raw waveform plot to: ", full_save_path)

# Load the number of entries
infile = ROOT.TFile(file_path)  # Open the ROOT file
tree = infile.Get("test_tree")  # Load tree
total_events_in_acq = tree.GetEntries()
print(f"There were {total_events_in_acq} events in this acquisition.")
infile.Close() # closing file after reading total number of events 

for event_num in tqdm(range(total_events_in_acq)):

    hit_channels = 0 # how many channels saw a real hit in this event?

    # Loading up the event object
    event_obj = Event()
    event_obj.load_basics(event_num, file_path)
    event_obj.load_actives(event_num, file_path)
    event_obj.load_data(event_num, file_path)

    # ------ Creating a matplotlib figure - using JC's code here ------
    # Create a 2x2 grid
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1])
    
    # Define subplots
    ax_sipm_vis = plt.subplot(gs[0, 0])  # Top-left (SiPM_VIS)
    ax_sipm_vuv = plt.subplot(gs[0, 1])  # Top-right (SiPM_VUV)
    ax_csp_x = plt.subplot(gs[1, 0])     # Bottom-left (CSP_x)
    ax_csp_y = plt.subplot(gs[1, 1])     # Bottom-right (CSP_y)

    # Share x-axis between CSP plots (optional)
    ax_csp_y.sharex(ax_csp_x)
    
    # Move SIPM_VUV and CSP_y y-axis to the right
    ax_sipm_vuv.yaxis.tick_right()
    ax_sipm_vuv.yaxis.set_label_position("right")
    
    ax_csp_y.yaxis.tick_right()
    ax_csp_y.yaxis.set_label_position("right")

    offset_value_x = 0
    offset_value_y = 0 # this isn't needed right now since we only have one axis
    offset_bin = 25
    # ------ end of JC's code snippet -------

    waveform_data = event_obj.waveform_data_2D
    time_array = [n*event_obj.resolution[0]*1e-3 for n in range(event_obj.num_samples[0])] # test this

    # adding vertical line
    ax_csp_x.vlines(16, -50, 550, ls="--", color="black", alpha=0.8)
    ax_csp_y.vlines(16, -50, 550, ls="--", color="black", alpha=0.8)

    # catalog for light and charge channels - sourced from xwiki
    light_vis_cat = [32] # even suffixes in description, CAEN CHANNEL NUMBERS
    light_vuv_cat = [] # odd suffixes in description,  CAEN CHANNEL NUMBERS
    charge_cat_x = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48] # they are all on the same axis, x or y
    charge_cat_y = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

    event_id = event_obj.event_num[0]

    for chan in waveform_data: # for key in dict

        # all waveforms are baseline corrected
        base_corr_waveform = waveform_data[chan] - np.mean(waveform_data[chan][:1500]) # 1500 because of our pre-trigger time, this is hard-coded and needs to be modified for different pre-trigger times
        
        #### checking for interesting events ####

        # start with charge channels
        if (chan in charge_cat_x or chan in charge_cat_y):
            # First, check if it's interesting
            # ------ this bit from JC's code ------
            rms = np.sqrt(np.mean(base_corr_waveform[:-int(event_obj.num_samples[0] / 3)] ** 2))
            if np.mean(base_corr_waveform[2200:][np.argpartition(base_corr_waveform[2200:], -5000)[-5000:]]) - np.mean(base_corr_waveform[:1500]) > 4 * rms and rms > 1:
                hit_channels += 1
            # ------ end ------

            # then plot on the x or y axes
            if chan in charge_cat_x:
                offset_base_corr_waveform = base_corr_waveform + offset_value_x
                line, = ax_csp_x.plot(time_array, offset_base_corr_waveform, alpha=0.8) # Robin added alpha here to make lines translucent
                offset_value_x += offset_bin

            elif chan in charge_cat_y:
                offset_base_corr_waveform = base_corr_waveform + offset_value_y
                line, = ax_csp_y.plot(time_array, offset_base_corr_waveform, alpha=0.8)
                offset_value_y += offset_bin
        
        # moving on to light channels
        # ------ code from JC ------
        if chan in light_vis_cat:
            line = ax_sipm_vis.plot(time_array, base_corr_waveform, alpha=0.8)[0]  # Note [0] here <-- JC's comment
        elif chan in light_vuv_cat:
            line = ax_sipm_vuv.plot(time_array, base_corr_waveform, alpha=0.8)[0]
        # ------ end -------

    # We're out of the channel loop, now we do some final cleaning up for the event graph
    # ------ From JC's code ------
    # Configure SiPM plots (top row)
    for ax_sipm in [ax_sipm_vis, ax_sipm_vuv]:
        ax_sipm.set_ylabel('Output [mV]')
        #ax_sipm.set_xlim(0, 150)
        ax_sipm.set_ylim(-150, 10)
        ax_sipm.legend(ncol=1, loc='lower right', fontsize=10)
    
    ax_sipm_vis.set_title('SiPM (VIS)')
    ax_sipm_vuv.set_title('SiPM (VUV)')
    
    # Configure CSP plots (bottom row)
    for ax_csp in [ax_csp_x, ax_csp_y]:
        ax_csp.set_xlabel('Time [μs]')
        ax_csp.set_ylabel('Output [mV]')
        #ax_csp.set_xlim(0, 150)
        ax_csp.set_ylim(-50, 550)
        #ax_csp.legend(ncol=1, loc='lower right', fontsize=10)
    
    ax_csp_x.set_title('CSP (X-axis)')
    ax_csp_y.set_title('CSP (Y-axis)')
    
    fig.suptitle(f"{file_name}\nEvent {event_num:04d}_{event_id:04d}")

    plt.tight_layout()
    
    # Save figure
    if hit_channels > 0:
        plt.savefig(f"{full_save_path}/***_{hit_channels}_{event_num:04d}_{event_id:04d}.png")
    else:
        plt.savefig(f"{full_save_path}/{event_num:04d}_{event_id:04d}.png")
    
    plt.close(fig)
    # ------ end ------
         
print("Done")
