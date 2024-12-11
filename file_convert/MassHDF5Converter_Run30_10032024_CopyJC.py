import matplotlib.pyplot as plt
import matplotlib.transforms as tfrms
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import poisson
from rich.progress import track
import h5py
import sys, os, time, re
import struct
import array
import numpy as np
import timeit
import gc
import json
from sys import getsizeof

# path = os.path.abspath(h5py.__file__)
# print(path)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def pts_per_trace(_):   # Creates time array given record length and sampling rate, e.g. elements are spaced by 8 (2) ns if SR is 125 (500) MSa/s
    pts = _ + 1 
    trace_length = np.arange(0, pts, 2)  
    return trace_length, len(trace_length)

class Event:
    def __init__(self, event_number, event_index, timestamp, sampling_rate, time_resolution, waveform):
        self.EventNumber = event_number
        self.EventIndex = event_index
        self.Timestamp = timestamp
        self.SamplingRate = sampling_rate
        self.TimeResolution = time_resolution
        self.Waveform = waveform
        
class ReadRawFile:
    def __init__(self, input_raw_file):
        self.input_raw_file = input_raw_file

    def read_raw_data(self, N, file_name):
        with open(self.input_raw_file, 'rb') as myfile:
            n=0
            while True:
                try:
                    event_index = n 
                    event_number = struct.unpack("I", myfile.read(4))[0]
                    myfile.seek(0, 1)
                    timestamp = struct.unpack("Q", myfile.read(8))[0]
                    myfile.seek(0, 1)
                    sampling_rate = struct.unpack("I", myfile.read(4))[0]
                    myfile.seek(0, 1)
                    time_resolution = struct.unpack("Q", myfile.read(8))[0]

                    wf = []
                    for _ in range(pts_per_trace(record_length)[1]): 
                        myfile.seek(0, 1)
                        wf.append(struct.unpack("f", myfile.read(4))[0])
                    waveform = np.array(wf)
            
                    event_object = Event(event_number, event_index, timestamp, sampling_rate, time_resolution, waveform)
                    n+=1
                    yield event_object
                except struct.error:
                    break

# Runtime start timer
thyme = time.time()

# Define path from which script is run
mast_path = os.getcwd() + '/'
dir_obj = os.scandir(mast_path)
# Define path to data
data_path = '/'
data_dir_obj = os.scandir(data_path)

HDD = 'HDD2'  # Choose hard drive where desired data lives
run_pref = 'Run30_SiPMData'  # Choose the run prefix or identifier for directory containing binary files


for entry in data_dir_obj:
    if entry.is_dir() and HDD in entry.name:
        data_path = f'{data_path}{entry.name}'
        dir_obj = os.scandir(data_path)
        for entry in dir_obj:
            if entry.is_dir() and run_pref in entry.name:
                data_path = f'{data_path}/{entry.name}'
os.chdir(data_path)
data_dir_obj = os.scandir(data_path)

print(f'Current working directory: {mast_path}')
print(f'Accessing data from: {data_path}')
print()

# Access each acquisition directory to produce HDF5 files from binary data
acq_id = 'LArSiPMTesting'  # Define identifier for acquisition (Use common identifier to convert all binaries from given run)
acq_ct = 1
for entry in data_dir_obj:
    if entry.is_dir() and acq_id in entry.name:
        acq_path = f'{data_path}/{entry.name}'
        print('Processing binary data from acquisition path:')
        print(acq_path)
        acq_dir_obj = os.scandir(acq_path)
        # Initialize metadata and file path dictionary
        chan_data_dict = {}
        acq_times = []
        chan_ct = 0
        for entry in acq_dir_obj:
            if entry.is_dir():
                if 'CH' in entry.name:
                    chan_data_dict[entry.name] = []
                    chan_data_path = f'{acq_path}/{entry.name}/'
                    chan_obj = os.scandir(chan_data_path)
                    for file in chan_obj:
                        if file.is_file():
                            if not 'DS_Store' in file.name:
                                chan_data_dict[entry.name]+=[file.name]
                                chan_data_dict[entry.name].sort(key=natural_keys)
                    file_num = int(len(chan_data_dict[entry.name]))
                    #print(f'No. of files created per channel in acquisition {int(acq_ct)}: {file_num}')
                
                    if chan_ct == 0:     # Only extract acquistion times through first set of binary files (i.e. first channel in channel dictionary) 
                        for i in enumerate(chan_data_dict[entry.name]):
                            _, start, stop = 0,0,0
                            acqtime = []
                            for j in enumerate(i[1]):
                                if j[1]=='_':
                                    _+=1
                                    if _ == 7:
                                        start = j[0]
                                if j[1]=='-':
                                    stop=j[0]
                            for k in enumerate(i[1]):
                                if k[0] in range(start+1,stop):
                                    acqtime.append(k[1])
                            acqtime = "".join(acqtime)
                            acq_times.append(acqtime)
                    #print(f'Binary file IDs by timestamp: {acq_times}')
                    chan_ct += 1
        #print(chan_data_dict)
        acq_ct+=1

        # Create directories for HDF5 files and their backups
        HDF5_genpath = f'{acq_path}/HDF5'
        HDF5b_genpath = f'{acq_path}/HDF5_backup'

        if not os.path.exists(HDF5_genpath):
            os.mkdir(HDF5_genpath)
        if not os.path.exists(HDF5b_genpath):
            os.mkdir(HDF5b_genpath)
                                                         
        #Create list of CAEN channel ID's in ascending numerical order             
        channel_key = list(chan_data_dict.keys())
        channel_key.sort(key=natural_keys)
        print('CAEN (WV2) Channel Key: ' +  str(channel_key))
        print()
        SiPM_chans = set()
        Q_chans = set()
        for key in range(len(channel_key)):
            if channel_key[key] == 'CH0' or channel_key[key]== 'CH2' or channel_key[key]== 'CH3':
                SiPM_chans.add(channel_key[key])
            else:
                Q_chans.add(channel_key[key])      # Warning: unique for this dataset, injection calibration only has charge channels for now
   
        ##### Eventually build information from lines into into HDF5 file metadata
        record_length = 79998       # Given in ns
        pretrig_length = 15998       # Given in ns

        t = pts_per_trace(record_length)[0]       # Construct time array, this should be applicable to all traces from all channels in a given dataset 
        samples = pts_per_trace(record_length)[1]
        print(samples)

        ################## Initialize numpy matrices ################## 
        ################### Build numpy waveform matrix  ################## 
        for j in range(file_num):
            for i in track(range(len(channel_key)),description=f'Processing events from file {j+1}/{file_num}'):
                filename = f'{acq_path}/{channel_key[i]}/{chan_data_dict[channel_key[i]][j]}'
                events = ReadRawFile(filename)
                compiled_events = events.read_raw_data(0, events.input_raw_file)   # Zero used as place holder, use select_events to specify
                if i==0:
                    num_events = 0
                    for event in compiled_events:
                        num_events = event.EventIndex
                    data_shape = (num_events+1, len(channel_key), pts_per_trace(record_length)[1]+4)    
                    event_shape = (len(channel_key), pts_per_trace(record_length)[1]+4)                 
                    trace_shape = (pts_per_trace(record_length)[1]+4) 
                    print()
                    print(f'Data Shape for file {j+1}/{file_num}: {data_shape}')
                    print(f'Building numpy matrix from file {j+1}/{file_num}...')
                    data = np.empty(data_shape)
                events = ReadRawFile(filename)
                compiled_events = events.read_raw_data(0, events.input_raw_file)
                for event in compiled_events:
                    event_ind = event.EventIndex
                    t_stamp = event.Timestamp
                    samp_rate = event.SamplingRate
                    t_res = event.TimeResolution
                    wf = event.Waveform
                    wf = np.insert(wf, (0,0,0,0), (event_ind, t_stamp, samp_rate, t_res))
                    data[event_ind][i] = wf
            with h5py.File(f'{HDF5_genpath}/{acq_times[j]}.h5', 'w') as hdf:
                t_data = hdf.create_group('Time')
                t_data.create_dataset('CAEN Time', data=t)             
                for n in track(range(len(data)), description=f'Writing data from acquisition file {j+1}/{file_num} to HDF5 format'):
                    # Create HDF5 dataset representing all events recorded by a given channel 
                    event = hdf.create_group('Event '+ str(n))
                    for i in range(len(channel_key)):
                        chan = event.create_dataset(channel_key[i]+'_WaveformData', data=data[n][i])
                        #Create attribute for each channel based on the detector type
                        if channel_key[i] in SiPM_chans:
                            chan.attrs['Detector Type'] = 'SiPM'
                        elif channel_key[i] in Q_chans:
                            chan.attrs['Detector Type'] = 'Charge'
                hdf.close()
print()
elapsed = time.time() - thyme
print('Total Runtime: '+str(elapsed)+' seconds')