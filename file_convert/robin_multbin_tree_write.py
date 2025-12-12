import struct   # Needed to read the binary file
import ROOT     # Needed to use TTrees 
from array import array    # Needed to package data so that it can be stored in a TTree, weird hold-over because ROOT needs C++ pointers
import os       # For converting rel. paths to abs.
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from pathlib import Path # to read in multiple binary files at once
from time import time

completer = PathCompleter()

file_path = prompt("Please enter the absolute path to the binary file directory: ", completer=completer)
print(f"You selected: {file_path}")

bin_dir_path = Path(file_path)

save_path = prompt("Please enter the absolute path where you would like the root file to be saved: ", completer=completer)
root_file_name = str(input("Please enter a file name for the new root file: "))

save_name = os.path.join(save_path, root_file_name+".root")

if save_path == "":
    save_path_base = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    save_path = os.path.join(save_path_base, file_name)
    save_name = os.path.join(save_path, os.path.splitext(file_name)[0]+".root")
    print(f"save name is {save_name}")

print(f"You are saving to this path: {save_path}.")

start_time = time() # in seconds

data_file_list = list(bin_dir_path.glob("*.bin")) # This gathers all files in bin_dir_path with the .bin extension!

writefile = ROOT.TFile(save_name, "RECREATE")
if (writefile.IsOpen() == False):
    print("Error in opening file.")

t = ROOT.TTree("test_tree", "test_tree")

# initializing some branch variables
event_num = array("i", [0])
timestamp = array("L", [0])
num_of_samples = array("i", [0])
one_sample_in_ns = array("i", [0])
channels = array("i", [0])
num_channels, num_samples = None, None

print(f"Preparing tree {root_file_name} and branches")

# reading only first event from first file for prep
with open(data_file_list[0], "rb") as prep_file:
    prep_file.read(12) # reading past first 12 bytes
    num_samples = struct.unpack("I", prep_file.read(4))[0]
    prep_file.read(8) # reading past next 8 bytes
    num_channels = struct.unpack("i", prep_file.read(4))[0]
    print(f"Samples: {num_samples}, channels: {num_channels}")

actives = array("i", num_channels*[0])
waveform_data = array("f", num_channels*num_samples*[0.0])

# actually creating the branches
t.Branch("event_num", event_num, "event_num/I")
t.Branch("timestamp", timestamp, "timestamp/l")
t.Branch("num_of_samples", num_of_samples, "num_of_samples/I")
t.Branch("resolution", one_sample_in_ns, "resolution/I")
t.Branch("num_of_channels", channels, "num_of_channels/I")
t.Branch("active_channels", actives, "active_channels[" + str(num_channels) + "]/I")
t.Branch("waveform_data", waveform_data, "waveform_data[" + str(num_channels*num_samples) + "]/F")

event_count = 1     
err_flag = 0 # means no errors. If not zero, we ran into some errors.
file_flag = 0 # just counts all the files
##### TO DO: Test this with multiple files like pedestal, maybe 60 events per file #####
for filename in data_file_list:
    file_flag += 1
    print(f"----- Reading file {file_flag}/{len(data_file_list)} -----")
    with open(filename, "rb") as file:
        while(True):
            try:
                event_num[0] = struct.unpack("I", file.read(4))[0] # reading 4 bytes out to an unsigned int value (which is what "I" means) # event_num is 4-byte unsigned int
                timestamp[0] = struct.unpack("Q", file.read(8))[0] # timestamp is 8-byte unsigned int # "Q" is an unsigned long long, which is 8 bytes
                num_of_samples[0] = struct.unpack("I", file.read(4))[0] # num_of_samples is 4-byte unsigned int
                one_sample_in_ns[0] = struct.unpack("Q", file.read(8))[0] # this is also 8-byte unsigned int
                channels[0] = struct.unpack("i", file.read(4))[0] # this is a 4-byte signed int

                for ch in range(channels[0]):
                    channel_number = struct.unpack("h", file.read(2))[0]
                    actives[ch] = channel_number
                    for times in range(num_of_samples[0]):
                        waveform_sample = struct.unpack("f", file.read(4))[0] # this is a 4-byte float
                        waveform_data[(num_of_samples[0] * ch) + times] = waveform_sample
                
                print(f"Filling event {event_count} in the ROOT TTree")
                t.Fill()
                event_count+=1

            except struct.error as e: # TO-DO: EOF should be handled more gracefully...
               print(e, f", probably EOF for file {file_flag}.")
               break # This stops us from reading data from a fully read file. Breaking here returns control flow to the main for loop that iterates over the .bin files.
            
            except Exception as e2: # If a strange exception occurs, write the tree and close the root file.
                print(e2)
                err_flag += 1
                t.Write()
                writefile.Close()
                break
        
if err_flag == 0:
    t.Write()
    writefile.Close()

run_time = round(time() - start_time, 3)
print(f"\n*******\nTree done.\nTotal events read: {event_count-1}.\nTime taken to process: {time() - start_time} seconds.\n*******")

