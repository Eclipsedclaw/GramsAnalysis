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
new_folder = 'raw_waveform'

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

# TODO: make updates to read the data to determine the actually points
time = [n*0.008 for n in range(37500)]


repeat = NbTraces


for n in tqdm(range(repeat)):
    interesting_event = False
    for x in range(num_channels):
        tree.GetEntry(NbTraces*x+n)                     # read entry 20
        if(x != 0 and event_id != tree.event_id):
            print("different event id")                         # check if in the same event id
        event_id = tree.event_id                                # event ID, WD2 skipped some numbers
        channel = tree.channel                                  # channel number, 30-
        if(x != 0 and timestamp != tree.timestamp):             # check if in the same timestamp
            print("different timestamp")
        timestamp = tree.timestamp                              # start time
        resolution = tree.resolution                            # 8 ns
        numberOfSamples = tree.numberOfSamples
        waveform_samples = np.array(tree.waveform_samples)      # 1D array with 37500 elements [mV]
        rms = np.sqrt(np.mean(waveform_samples[:1900]**2))
        waveform_samples = waveform_samples - np.mean(waveform_samples[:1900]) # baseline correction
        #print("max value is: ",max(abs(waveform_samples[2200:])))
        #print("rms value is: ", rms)
        if (max(abs(waveform_samples[2200:]-np.mean(waveform_samples[:1900]))) > 5 * rms):
            interesting_event = True
        label = "Ch" + str(channel)
        plt.plot(time,waveform_samples,label=label)
        plt.legend(ncol=12, loc = "upper center",fontsize=5)
    plt.title('Waveform for Event ID = '+ str(event_id))
    plt.xlabel('time [us]')
    plt.ylabel('output [mV]')
    plt.xlim(0,300)
    plt.ylim(-300,300)
    if(interesting_event==True):
        plt.savefig(full_save_path+'/***'+str(n).zfill(4)+"_"+str(event_id).zfill(4)+".png")
    else:
        plt.savefig(full_save_path+'/'+str(n).zfill(4)+"_"+str(event_id).zfill(4)+".png")
    #plt.show()
    plt.clf()

print("Done")