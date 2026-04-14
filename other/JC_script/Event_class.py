import struct                       # Needed to read the binary file
import ROOT                         # Needed to use ROOT functionality
from array import array             # Needed to package data so that it can be stored in a TTree, weird hold-over because ROOT needs C++ pointers
import os                           # For converting rel. paths to abs.
import numpy as np                  # For converting lists to np arrays
from prompt_toolkit import prompt   # Prompt completion
from prompt_toolkit.completion import PathCompleter

class Event:
    def __init__(self):
        self.event_num = array("i", [0])        # These are all initialized in this weird way so that they can act as C++ pointers
        self.timestamp = array("L", [0])
        self.num_channels = array("i", [0])
        self.num_samples = array("i", [0])
        self.resolution = array("i", [0])
        self.actives = None     # needs to be initialized
        self.raw_data = None    # needs to be initialized
        self.channel_dict = {0: None}
        self.waveform_data_2D = {} # needs to be initialized


    def load_basics(self, event_num, rootfilename, treename="test_tree"):
        """load_basics(self, event_num, rootfilename, treename)
        
        Provide the name of the ROOT file and the TTree name and this function will extract all the basic information of the run as below:
        - event number
        - timestamp
        - number of active channels
        - resolution of the data in nanoseconds
        - number of samples in a single waveform

        You can recreate the x-axis in time units with this info.
        """
        infile = ROOT.TFile(rootfilename)           # Open the ROOT file
        tree = infile.Get(treename)                 # Get the name of the tree
        tree.SetBranchAddress("event_num", self.event_num)
        tree.SetBranchAddress("timestamp", self.timestamp)
        tree.SetBranchAddress("num_of_samples", self.num_samples)
        tree.SetBranchAddress("resolution", self.resolution)
        tree.SetBranchAddress("num_of_channels", self.num_channels)

        tree.GetEntry(event_num)                            # Just get the first entry since this info is the same across all events
        infile.Close()                              # Close the ROOT file!!!


    def load_actives(self, event_num, rootfilename, treename="test_tree"):
        """load_actives(self, event_num, rootfilename, treename)

        Run load_basics() first for this function to initialize these values properly.

        Provide the name of the ROOT file and the TTree name to load the active channels and create a dictionary.

        The created dictionary has keys from 0 - num_of_channels and values as the physical CAEN channel numbers. 
        """
        infile = ROOT.TFile(rootfilename)
        tree = infile.Get(treename)

        # Initialize the arrays (pseudo-pointers) with the info from load_basics()
        self.actives = array("i", self.num_channels[0]*[0])
        tree.SetBranchAddress("active_channels", self.actives)

        tree.GetEntry(event_num)
        infile.Close()

        # Creating the dictionary
        for ch in range(0, self.num_channels[0]):
            self.channel_dict[ch] = self.actives[ch]
        
        #print(f"Channel dictionary: {self.channel_dict}")


    def load_data(self, event_num, rootfilename, treename="test_tree"):
        """load_data(self, event_num, rootfilename, treename)

        Loads data for a given event_num from the ROOT TTree into a dictionary. Event_num is indexed from 0.

        Run load_basics() and load_actives() before this!

        The dictionary key is the CAEN channel number and the value is the waveform as a np.array()
        """
        
        infile = ROOT.TFile(rootfilename)
        tree = infile.Get(treename)

        self.raw_data = array("f", self.num_channels[0]*self.num_samples[0]*[0.0])
        tree.SetBranchAddress("waveform_data", self.raw_data)

        tree.GetEntry(event_num)
        infile.Close()

        for ch in range(0, self.num_channels[0]):
            ch_id = self.channel_dict[ch]   # reading CAEN channel ID from the channel dictionary
            waveform = []
            for times in range(0, self.num_samples[0]):
                sample = self.raw_data[(self.num_samples[0] * ch) + times]  # Parsing the format that the data is stored in
                waveform.append(sample)
            self.waveform_data_2D[ch_id] = np.array(waveform)


        if len(self.waveform_data_2D) == self.num_channels[0]:
            pass
            #print("2D waveform dictionary loaded.")

    def print_summary(self):
        """print_summary(self)
        
        Prints a quick summary of the event."""

        print("\n")
        print(" -------------- Event summary -------------- ")
        print(f" Event number: {self.event_num[0]}")
        print(f" Timestamp: {self.timestamp[0]}")
        print(f" Resolution: {self.resolution[0]} ns")
        print(f" Number of samples: {self.num_samples[0]}")
        print(f" Number of active channels: {self.num_channels[0]}")
        print(f" Channel dictionary: {self.channel_dict}")
        if (len(self.waveform_data_2D) == 1):
            print(f" Data loaded? No.")
        elif (len(self.waveform_data_2D) == self.num_channels[0] and self.num_channels[0] != 0):
            print(f" Data loaded? Yes.")
        else:
            print(f" Data loaded? Uncertain state.")
        print(" ------------------------------------------- ")
        print("\n")

## Code to test ## 
if __name__ == "__main__":
    test = Event()
    test.print_summary() # should not be initialized
    test.load_basics("/NAS/Robin/CAEN_TTree_test/var_tree.root", "test_tree")
    test.print_summary() # basics initialized
    test.load_actives("/NAS/Robin/CAEN_TTree_test/var_tree.root", "test_tree")
    test.print_summary() # everything except data initialized
    test.load_data(0, "/NAS/Robin/CAEN_TTree_test/var_tree.root", "test_tree")
    print(len(test.waveform_data_2D))
    test.print_summary() # everything loaded.
    print("End.")
## end of code to test ##
