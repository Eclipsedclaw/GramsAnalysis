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
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

def GRAMS_shaper_trial_1(raw_waveform, filter_order, critical_frequency, gaussian_sigma, gain, shaping_time=5, sampling_rate=125):
    b, a = butter(filter_order, critical_frequency, 'low')
    #plt.plot(raw_waveform, alpha = 0.7, label = 'original')
    filtered = filtfilt(b, a, raw_waveform, padtype="constant")
    #plt.plot(high_pass_filtered, alpha = 0.6, label = 'high pass filtered')
    
    # This is for lifting baseline, not trustworthy now
    #high_pass_filtered = (high_pass_filtered - np.min(high_pass_filtered)) / (np.max(high_pass_filtered) - np.min(high_pass_filtered)) * np.max(high_pass_filtered)

    # Gaussian filter
    gaussian_filtered = gaussian_filter1d(filtered, sigma=gaussian_sigma)
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

def fit_fn(x, A, freq, phase):
    return A*np.sin(x*(2*np.pi*freq) + phase)

def subtract_sinusoid(data, cutoff_freq, order=4):
    b, a = butter(order, cutoff_freq, btype='low')
    filtered = filtfilt(b, a, data, padtype="constant")
    remainder = data - filtered
    guess = [10, 1/5000, 0]
    x_vals = np.arange(len(data))
    popt, _ = curve_fit(fit_fn, x_vals, remainder, p0=guess)
    fitted = fit_fn(x_vals, *popt)
    res = data - fitted

    # subtract baseline
    res = res - np.mean(res[:1500])

    return res


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
new_folder = 'WV_filtered_waveform'

# Create the full path including the new folder
full_save_path = os.path.join(save_path, new_folder)

# Create the directory if it doesn't exist
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)

print("Save filtered waveform plot to: ", full_save_path)

# Load the number of entries
infile = ROOT.TFile(file_path)  # Open the ROOT file
tree = infile.Get("test_tree")  # Load tree
total_events_in_acq = tree.GetEntries()
print(f"There were {total_events_in_acq} events in this acquisition.")
#infile.Close() # closing file after reading total number of events 

print("Before loading event_obj\n")
event_obj = Event()    # Loading up the event object. Only needed once!

offset_bin = 25

light_vis_cat = [2, 4]
light_vuv_cat = [19, 35]

# using DB50->DB62 converter board (not DB50->2xDB37 directly)
charge_cat_x = [21, 5, 39, 22, 6, 40, 23, 7, 41, 24, 8, 42, 25, 9, 26] # A
charge_cat_y = [10, 44, 28, 11, 45, 29, 12, 46, 30, 13, 47, 31, 14, 48, 27] #B
#charge_cat_x = [21, 5, 39, 22, 6, 40, 23, 7, 41, 24, 8, 42, 25, 9, 26] # bottom
#charge_cat_y = [43, 27, 10, 44, 28, 11, 45, 29, 12, 46, 30, 13, 47, 31, 14, 48] # top



for event_num in tqdm(range(total_events_in_acq)):

    hit_channels = 0 # how many channels saw a real hit in this event?

    # Loading up the event object
    event_obj.load_basics(event_num, tree)
    event_obj.load_actives(event_num, tree)
    event_obj.load_data(event_num, tree)

    waveform_data = event_obj.waveform_data_2D
    time_array = [n*event_obj.resolution[0]*1e-3 for n in range(event_obj.num_samples[0])] # test this

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

    event_id = event_obj.event_num[0]

    try:
        for chan in waveform_data: # for key in dict

            # all waveforms are baseline corrected
            base_corr_waveform = waveform_data[chan] - np.mean(waveform_data[chan][:1500]) # 1500 because of our pre-trigger time, this is hard-coded and needs to be modified for different pre-trigger times

            #### checking for interesting events ####

            # start with charge channels
            if chan in charge_cat_x or chan in charge_cat_y:

                # apply GRAMS filter
                base_corr_waveform = subtract_sinusoid(base_corr_waveform, .8*0.0001985) # freq from sinusoid fit)
                base_corr_waveform = GRAMS_shaper_trial_1(
                        base_corr_waveform,
                        filter_order=1,
                        critical_frequency=0.0002,
                        gaussian_sigma=250,
                        shaping_time=5,
                        gain=1,
                        sampling_rate=125
                )
                # First, check if it's interesting
        
                if np.max(base_corr_waveform) > 20:
                    hit_channels += 1
                # ------ end ------

                # then plot on the x or y axes # freq from sinusoid fit
                if chan in charge_cat_x:
                    offset_value = offset_bin * charge_cat_x.index(chan)
                    offset_base_corr_waveform = base_corr_waveform + offset_value
                    line, = ax_csp_x.plot(time_array, offset_base_corr_waveform, alpha=0.8) # Robin added alpha here to make lines translucent

                elif chan in charge_cat_y:
                    offset_value = offset_bin * charge_cat_y.index(chan)
                    offset_base_corr_waveform = base_corr_waveform + offset_value
                    line, = ax_csp_y.plot(time_array, offset_base_corr_waveform, alpha=0.8)
            
            # moving on to light channels
            # ------ code from JC ------
            if chan in light_vis_cat:
                line = ax_sipm_vis.plot(time_array, base_corr_waveform, alpha=0.8, lw=2)[0]  # Note [0] here <-- JC's comment
            elif chan in light_vuv_cat:
                line = ax_sipm_vuv.plot(time_array, base_corr_waveform, alpha=0.8, lw=2)[0]
            # ------ end -------

        # We're out of the channel loop, now we do some final cleaning up for the event graph
        # ------ From JC's code ------
        # Configure SiPM plots (top row)
        for ax_sipm in [ax_sipm_vis, ax_sipm_vuv]:
            ax_sipm.set_ylabel('Output [mV]')
            #ax_sipm.set_xlim(0, 150)
            ax_sipm.set_ylim(-150, 10)
            #ax_sipm.legend(ncol=1, loc='lower right', fontsize=10)
            ax_sipm.grid(axis="both", alpha=0.5, ls="--")
        
        ax_sipm_vis.set_title('SiPM (VIS)')
        ax_sipm_vuv.set_title('SiPM (VUV)')
        
        # Configure CSP plots (bottom row)
        for ax_csp in [ax_csp_x, ax_csp_y]:
            ax_csp.set_xlabel('Time [μs]')
            ax_csp.set_ylabel('Output [mV]')
            #ax_csp.set_xlim(0, 150)
            ax_csp.set_ylim(-50, 550)
            #ax_csp.legend(ncol=1, loc='lower right', fontsize=10)
            ax_csp.axvline(16, ls="--", c="gray", alpha=0.8)
            ax_csp.axvline(109, ls="--", c="gray", alpha=0.8)
            ax_csp.grid(axis="both", alpha=0.5, ls="--")

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
    except RuntimeError: # in case fitting sinusoid fails
        continue

infile.Close() # closing TFile here since Event_class doesn't handle that anymore.
print("Done")
