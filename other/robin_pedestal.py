"""
Analyze CAEN root file pedestal for Aramaki Lab
author: Jiancheng Zeng
Date: Oct 4, 2024
"""

import ROOT
import matplotlib.pyplot as plt
import numpy as np
from array import array
import pandas as pd
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from Event_class import Event
from tqdm import tqdm # For a progress bar!

# Create a PathCompleter for file path tab-completion
completer = PathCompleter()

from statistics import stdev



def GRAMS_RMS(baseline_array):
    #print("Calculating RMS")
    """
    Calculate the Root Mean Square (RMS) of a given array.
    
    Parameters:
    arr (list or np.array): Input array of numerical values.
    
    Returns:
    float: The RMS value of the array.
    """
    baseline_array = np.array(baseline_array)  # Ensure input is a numpy array
    return np.sqrt(np.mean(baseline_array**2))


def GRAMS_Pedestal(file_address):

    event_obj = Event()

    file = ROOT.TFile(file_address)

    treename = "test_tree"

    mytree = file.Get(treename)

    num_events = mytree.GetEntries()
    
    fluctuation_rms = []
    df = pd.DataFrame()
    print("Reading root file..")

    channel_data = {}  # Temporary storage for channel data
    channel_means = {}   # Temporary storage for fluctuation_rms

    # This loads data into the channel_data dictionary
    for acq in tqdm(range(num_events)):
        #print(f"Working on event {acq}...")
        event_obj.load_basics(acq, file_address, treename)  # Loads basic info like event_num, timestamp etc
        event_obj.load_actives(acq, file_address, treename) # Loads active channel map
        event_obj.load_data(acq, file_address, treename)    # Loads waveform data into a dictionary {2:[...], 3:[...], 17:[...], ..., 56:[...]} keys are active channel number

        for ch in range(event_obj.num_channels[0]):
            real_ch_number = event_obj.actives[ch]
            try:
                np.append(channel_data[real_ch_number], event_obj.waveform_data_2D[real_ch_number])
                # like extend([data already in channel data for ch x], [data newly read for ch x])
                np.append(channel_means[real_ch_number], np.average(event_obj.waveform_data_2D[real_ch_number]))
            except KeyError:
                channel_data[real_ch_number] = event_obj.waveform_data_2D[real_ch_number]
                channel_means[real_ch_number] = np.average(event_obj.waveform_data_2D[real_ch_number])

    channel_rms = {}

    print("Loading rms values...")

    for ch in channel_means:
        try:
            channel_rms[ch] = stdev(channel_means[ch])/np.sqrt(len(channel_means))
        except:
            channel_rms[ch] = 0
            print("fluctuation got a weird result")
    
    # Add columns to the DataFrame in ascending order of channel numbers
    fluctuation_rms = []  # Reset fluctuation_rms to follow the sorted order
    for channel in sorted(channel_data.keys()):
        column_name = f"Ch{channel}"  # Create the column name from the sorted channel
        df[column_name] = channel_data[channel]
        fluctuation_rms.append(channel_rms[channel])  # Append rms in the same order

    print(df.head())
    return df, fluctuation_rms, event_obj.num_channels[0]


def baseline_correction(baseline_array):
    print("Performing baseline correction")
    avg = np.average(baseline_array)
    #print("first ", pretrigger, " samples has an average:", avg, "mV")
    corrected = np.array(baseline_array) - avg
    return corrected


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


# Convert to an absolute path if a relative path is provided
save_path = os.path.abspath(save_path)
save_name = os.path.join(save_path, pedestal_name)
print("Save output pedestal plot to: ", save_name)

# Ask the user to input the number of channels
# num_channels = int(input("Please enter the number of channels: ")) # no need for this with event based tree

print(f"Absolute file path: {file_path}")
print(f"Absolute save path: {save_path}")
#print(f"Number of channels: {num_channels}")


RMS_data = []
data = GRAMS_Pedestal(file_address=file_path)
data_wf = data[0]
RMS_errors = data[1]
num_channels = data[2]  # so that your script below can work

for i in range(num_channels):
    corrected_data = baseline_correction(data_wf.iloc[:,i])
    #plt.plot(corrected_data)
    RMS_data.append(GRAMS_RMS(corrected_data))
    print("Processing CAEN channel "+ str(data_wf.columns[i])+" ("+str(i+1)+"/"+str(num_channels)+")")

##### Modification by Robin - Save the raw data, RMS_data and RMS_errors for later analysis since it doesn't show the actual values in the plot #####
#print(RMS_data)
#print("\n")
#print(RMS_errors)
#print(type(RMS_data))
#print(type(RMS_errors))

choice = str(input("Do you want to save the raw RMS data/errors into a file? (Y/N) ")).lower()
if choice=="y":
    data_path = prompt("Please enter the path where the raw data should be saved. ", completer=completer)
    if data_path=="":
        data_path=os.path.dirname(file_path)
    data_path = os.path.abspath(data_path)
    data_name = os.path.join(data_path, "raw_data.txt")
    print("Save raw arrays to: ", data_name)
    HV_val = str(input("HV value?: "))
    rms_data_str = str(RMS_data)
    rms_err_str = str(RMS_errors)
    with open(data_name, "a") as array_file:
        array_file.write(f"HV:{HV_val}\n{rms_data_str}\n{rms_err_str}\n")
    array_file.close()
    print("Data written to file.")
elif choice=="n":
    pass
else:
    print("Something strange happened.")
    pass
#######################################################################################################################################################

# Create the bar plot
plt.figure(figsize=(10, 6))  # Set figure size

# Customize error bar appearance using error_kw
error_kw = {'ecolor': 'red', 'capsize': 5, 'elinewidth': 2, 'alpha': 0.8}
plt.bar(range(len(RMS_data)), RMS_data,  yerr=RMS_errors, capsize=5, error_kw=error_kw)  # Generate the bar plot
# Custom labels for x-axis (one for each element, can be any label like strings)
custom_labels = data_wf.columns  # Get the column names from the DataFrame

# Customize x-axis ticks
plt.xticks(ticks=range(len(RMS_data)), labels=custom_labels, rotation=45, fontsize=10)  # Set custom labels, rotate, and adjust font size
plt.xlabel('channel number')  # Label for x-axis
plt.ylabel('RMS/mV')          # Label for y-axis

# Set plot title using the extracted file name
plt.title(f'RMS - {file_name}', fontsize=14)
plt.grid(True, linestyle='--')  # Add grid lines to the plot


plt.savefig(save_name, format='png', dpi=300, bbox_inches='tight')  # Save the figure

plt.show()  # Display the plot
