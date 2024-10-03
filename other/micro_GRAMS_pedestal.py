import ROOT
import matplotlib.pyplot as plt
import numpy as np
from array import array
import pandas as pd
import os
import pathlib
import readline


def complete_path(text, state):
    incomplete_path = pathlib.Path(text)
    if incomplete_path.is_dir():
        completions = [p.as_posix() for p in incomplete_path.iterdir()]
    elif incomplete_path.exists():
        completions = [incomplete_path]
    else:
        exists_parts = pathlib.Path('.')
        for part in incomplete_path.parts:
            test_next_part = exists_parts / part
            if test_next_part.exists():
                exists_parts = test_next_part

        completions = []
        for p in exists_parts.iterdir():
            p_str = p.as_posix()
            if p_str.startswith(text):
                completions.append(p_str)
    return completions[state]


# we want to treat '/' as part of a word, so override the delimiters
readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete_path)
#print(input('tab complete a filename: '))

def GRAMS_Pedestal(file_address, num_channels):
    file = ROOT.TFile(file_address)

    mytree = file.Get("tree")

    branches = mytree.GetListOfBranches()

    for branch in branches:
        print(branch.GetName())

    mytree.SetBranchStatus("waveform_samples", 1)
    raw_wf = array('f', [0]*(len(mytree.waveform_samples)-1))
    mytree.SetBranchAddress("waveform_samples", raw_wf)

    num_events = int(mytree.GetEntries()/num_channels)

    df = pd.DataFrame()
    print("Reading root file..")
    for i in range(num_channels):
        wfarray = []
        for j in range(num_events):
            mytree.GetEntry(i*num_events + j)
            #wfarray = np.random.random(len(mytree.waveform_samples))
            wfarray.extend(raw_wf)
            #print("wfarray length is: ", len(wfarray))
            column_name = f"ch{mytree.channel}"
        df[column_name] = wfarray


    print(df.head())
    return df


def baseline_correction(baseline_array):
    avg = np.average(baseline_array)
    #print("first ", pretrigger, " samples has an average:", avg, "mV")
    corrected = np.array(baseline_array) - avg
    return corrected


def GRAMS_RMS(baseline_array):
    """
    Calculate the Root Mean Square (RMS) of a given array.
    
    Parameters:
    arr (list or np.array): Input array of numerical values.
    
    Returns:
    float: The RMS value of the array.
    """
    baseline_array = np.array(baseline_array)  # Ensure input is a numpy array
    return np.sqrt(np.mean(baseline_array**2))


# Ask the user to input the file path
file_path = input("Please enter the file path to the ROOT file: ")

# Convert to an absolute path if a relative path is provided
file_path = os.path.abspath(file_path)

# Extract just the file name from the file path
file_name = os.path.basename(file_path)
pedestal_name = "pedestal_"+os.path.splitext(file_name)[0]+".png"
# Ask the user to input the save path for the plot
save_path = input("Please enter the path where you want to save the plot: ")

# Convert to an absolute path if a relative path is provided
save_path = os.path.abspath(save_path)
save_name = os.path.join(save_path, pedestal_name)
print("Save output pedestal plot to: ", save_name)

# Ask the user to input the number of channels
num_channels = int(input("Please enter the number of channels: "))

print(f"Absolute file path: {file_path}")
print(f"Absolute save path: {save_path}")
print(f"Number of channels: {num_channels}")


RMS_data = []
data = GRAMS_Pedestal(file_address=file_path, num_channels=num_channels)

for i in range(num_channels):
    corrected_data = baseline_correction(data.iloc[:,i])
    #plt.plot(corrected_data)
    RMS_data.append(GRAMS_RMS(corrected_data))
    print("Processing CAEN channel "+ str(data.columns[i])+" ("+str(i+1)+"/"+str(num_channels)+")")

# Create the bar plot
plt.figure(figsize=(10, 6))  # Set figure size
plt.bar(range(len(RMS_data)), RMS_data)  # Generate the bar plot
# Custom labels for x-axis (one for each element, can be any label like strings)
custom_labels = data.columns  # Get the column names from the DataFrame
# Customize x-axis ticks
plt.xticks(ticks=range(len(RMS_data)), labels=custom_labels, rotation=45, fontsize=10)  # Set custom labels, rotate, and adjust font size
plt.xlabel('channel number')  # Label for x-axis
plt.ylabel('RMS/mV')          # Label for y-axis
# Set plot title using the extracted file name
plt.title(f'RMS performance for all CSP channels - {file_name}', fontsize=14)
plt.grid(True, linestyle='--')  # Add grid lines to the plot


plt.savefig(save_name, format='png', dpi=300, bbox_inches='tight')  # Save the figure

plt.show()  # Display the plot
