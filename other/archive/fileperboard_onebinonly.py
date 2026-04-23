import struct   # Needed to read the binary file
import ROOT     # Needed to use TTrees 
from array import array    # Needed to package data so that it can be stored in a TTree, weird hold-over because ROOT needs C++ pointers
import os       # For converting rel. paths to abs.
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

completer = PathCompleter()

file_path = prompt("Please enter the absolute path to the binary file: ", completer=completer)
print(f"You selected: {file_path}")

save_path = prompt("Please enter the absolute path where you would like the root file to be saved: ", completer=completer)
root_file_name = str(input("Please enter a file name for the new root file: "))

save_name = os.path.join(save_path, root_file_name+".root")

if save_path == "":
    save_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path + "_pedestal")
    save_name = os.path.join(save_path, os.path.splitext(file_name)[0]+".root")

with open(file_path, "rb") as file:

    # Creating a ROOT TFile here. YOU CANNOT USE THE with KEYWORD HERE.
    writefile = ROOT.TFile(save_name, "RECREATE")

    # Catching TFile opening errors here
    if (writefile.IsOpen() == False):
        print("Error in opening TFile.")
        
    # defining tree and branches here
    t = ROOT.TTree("test_tree", "test_tree")

    # initializing some branch variables
    event_num = array("i", [0])
    timestamp = array("L", [0])
    num_of_samples = array("i", [0])
    one_sample_in_ns = array("i", [0])
    channels = array("i", [0])

    # now we need an array, so we must read out the first header
    initialize = 0
    event_num[0] = struct.unpack("I", file.read(4))[0]
    timestamp[0] = struct.unpack("Q", file.read(8))[0]
    num_of_samples[0] = struct.unpack("I", file.read(4))[0]
    one_sample_in_ns[0] = struct.unpack("Q", file.read(8))[0]
    channels[0] = struct.unpack("i", file.read(4))[0]

    # now we can initialize the data branches
    actives = array("i", channels[0]*[0])
    waveform_data = array("f", channels[0]*num_of_samples[0]*[0.0])

    # actually creating the branches
    t.Branch("event_num", event_num, "event_num/I")
    t.Branch("timestamp", timestamp, "timestamp/l")
    t.Branch("num_of_samples", num_of_samples, "num_of_samples/I")
    t.Branch("resolution", one_sample_in_ns, "resolution/I")
    t.Branch("num_of_channels", channels, "num_of_channels/I")
    t.Branch("active_channels", actives, "active_channels[" + str(channels[0]) + "]/I")
    t.Branch("waveform_data", waveform_data, "waveform_data[" + str(channels[0]*num_of_samples[0]) + "]/F")
    
    flag = 0 # means no errors. If not zero, we ran into some errors.
    event_count = 1

    while(True): # infinite loop, be careful!!
        try:
            # reading from bin file
            if initialize == 0:
                #event_num[0] = struct.unpack("I", file.read(4))[0] # reading 4 bytes out to an unsigned int value (which is what "I" means) # event_num is 4-byte unsigned int
                #timestamp[0] = struct.unpack("Q", file.read(8))[0] # timestamp is 8-byte unsigned int # "Q" is an unsigned long long, which is 8 bytes
                #num_of_samples[0] = struct.unpack("I", file.read(4))[0] # num_of_samples is 4-byte unsigned int
                #one_sample_in_ns[0] = struct.unpack("Q", file.read(8))[0] # this is also 8-byte unsigned int
                #channels[0] = struct.unpack("i", file.read(4))[0] # this is a 4-byte signed int
                #event_dict = {}
            
                for ch in range(0, channels[0]):
                    channel_number = struct.unpack("h", file.read(2))[0]
                    actives[ch] = channel_number
                    for times in range(0, num_of_samples[0]):
                        waveform_sample = struct.unpack("f", file.read(4))[0] # this is a 4-byte float
                        waveform_data[(num_of_samples[0]*ch)+times] = waveform_sample # so you have [[ch0data], [ch1data], ..., [lastchdata]] where each chdata array is num_samples long and the whole array is flattened.
                    #event_dict[channel_number] = waveform_array
                initialize = 1

            elif initialize == 1: # did we move onto the second event in the file?
                event_num[0] = struct.unpack("I", file.read(4))[0] # reading 4 bytes out to an unsigned int value (which is what "I" means) # event_num is 4-byte unsigned int
                timestamp[0] = struct.unpack("Q", file.read(8))[0] # timestamp is 8-byte unsigned int # "Q" is an unsigned long long, which is 8 bytes
                num_of_samples[0] = struct.unpack("I", file.read(4))[0] # num_of_samples is 4-byte unsigned int
                one_sample_in_ns[0] = struct.unpack("Q", file.read(8))[0] # this is also 8-byte unsigned int
                channels[0] = struct.unpack("i", file.read(4))[0] # this is a 4-byte signed int
                #event_dict = {}
            
                for ch in range(0, channels[0]):
                    channel_number = struct.unpack("h", file.read(2))[0]
                    actives[ch] = channel_number
                    for times in range(0, num_of_samples[0]):
                        waveform_sample = struct.unpack("f", file.read(4))[0] # this is a 4-byte float
                        waveform_data[(num_of_samples[0]*ch)+times] = waveform_sample

            print(f"Filling event {event_count} in the ROOT TTree")
            t.Fill()
            event_count+=1
        except struct.error as e:
            print(e, ", probably EOF")
            t.Write()
            flag+=1
            writefile.Close() # close file in case of errors, because like I said, bad things happen if you don't do this!!!!!
            break 

    if flag == 0: # else is handled by the except statement
        t.Write()
        writefile.Close() # if you do not close, bad bad things happen :((
