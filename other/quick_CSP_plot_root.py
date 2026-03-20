import numpy as np
import ROOT
import matplotlib.pyplot as plt
import os
from Event_class import Event
from tqdm import tqdm
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# to make figure hi res
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# to improve performance...this code kind of sucks
import matplotlib.style as mplstyle
mplstyle.use('fast')

# ---------- prompts for user -- copied from fileperboard_root_plot_raw.py --------------
# Create a PathCompleter for file path tab-completion
completer = PathCompleter()

# Ask the user to input the file path
file_path = prompt('Please enter the file path to the ROOT file: ', completer=completer)
print(f'You selected: {file_path}')

# Convert to an absolute path if a relative path is provided
file_path = os.path.abspath(file_path)

# Extract just the file name from the file path
file_name = os.path.basename(file_path)
pedestal_name = "heatmap_"+os.path.splitext(file_name)[0]+".png"

# Ask the user to input the save path for the plot
save_path = prompt('Please enter the path where you want to save the plot(default same as root file directory): ', completer=completer)
if(save_path == ''):
    save_path = os.path.dirname(file_path)
print(f'You selected: {save_path}')

# Define the new folder name
new_folder = 'heatmap'

# Create the full path including the new folder
full_save_path = os.path.join(save_path, new_folder)

# Create the directory if it doesn't exist
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)

print("Save waveform heatmap to: ", full_save_path)
# ---------- prompts for user -- copied from fileperboard_root_plot_raw.py --------------

# ---------- trying to sample every 12th point --------------
def trim(waveform, factor):
    trimmed = [waveform[n] for n in range(0, len(waveform), factor)]
    return trimmed
# ---------- trying to sample every 12th point --------------

# counting total number of events
data_filename = file_path
tfile = ROOT.TFile(data_filename)
tree = tfile.Get("test_tree")
num_events = tree.GetEntries()
#tfile.Close() # don't close yet
print(f"There are {num_events} events.")

# prep work
csp_event = Event()
csp_event.load_basics(1, tree)
csp_event.load_actives(1, tree)

# clean up channel_list by removing light channels
light_chans = [30, 31, 32, 33]

channel_dict = csp_event.channel_dict

#offset = 4 # needed to remove light channels <- is hacky, pls fix, future me

charge_channel_dict = {ind-4: channel_dict[ind] for ind in range(4, len(channel_dict))}

#print("Old dict\n", csp_event.channel_dict, "\n")
#print("New dict\n", charge_channel_dict)

time_array = [n*csp_event.resolution[0]*1e-3 for n in range(csp_event.num_samples[0])]

fig, axs = plt.subplots(4, 8) # create an 8x4 grid, not location based currently; just a quick&dirty plot for now
fig.suptitle("CSP responses")

#test_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#test_y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#prepping figure
for ind in range(0, len(axs.flat)):
    try:
        axs.flat[ind].set_xlabel("Time (us)", fontsize=8)
        axs.flat[ind].set_ylabel("Amplitude (mV)", fontsize=6)
        axs.flat[ind].set_ylim(-75, 300)
        #axs.flat[ind].plot(test_x, test_y)
        axs.flat[ind].label_outer()
        axs.flat[ind].set_title(f"Channel {charge_channel_dict[ind]}", fontsize=6)
    except KeyError: # i have more graphs than charge channels; this acq has only 29 charge channels. That's why I need this break
        axs.flat[ind].set_facecolor("black")

# filling figure
for times in tqdm(range(0, 100)): # event loop
    csp_event.load_basics(times, tree)
    csp_event.load_actives(times, tree)
    csp_event.load_data(times, tree)

    event_dict = csp_event.waveform_data_2D
    
    for ind in range(0, len(charge_channel_dict)): # plotting all channels in one event but skipping light channels. that's why I have the 4 hardcoded
        channel_waveform = event_dict[charge_channel_dict[ind]]
        axs.flat[ind].plot(trim(time_array, 20), trim(channel_waveform, 20), alpha=0.3, color="#3e68e6")

tfile.Close() # finally close the file

# all things plotted
fig.savefig(f"{full_save_path}/heatmap.png")
fig.close() # closing this to prevent memory leaks
