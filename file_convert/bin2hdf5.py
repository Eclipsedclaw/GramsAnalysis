import numpy as np
import struct
import time
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from tqdm import tqdm
import h5py

## field sizes in bytes - in this order
L_EVENT = 28 # 4 + 8 + 4 + 8 + 4 bytes

def read_header(file_path, header_len=L_EVENT):
    """Open file and read first few bytes to determine event structure"""
    path = Path(file_path)
    with open(path, "rb") as file:
        header = file.read(header_len)
    _, _, n_samps, res, n_chans = struct.unpack("<IQIQi", header)
    print(f"Read event info from first event in {path.name}:")
    print(f" {n_chans} channels")
    print(f" {n_samps} samples")
    print(f" {res} ns resolution")
    return n_chans, n_samps

def bin_dtype(n_chans, n_samps):
    """dtype for reading bin files"""
    return np.dtype([
        ("event_num",       "u4"),
        ("timestamp",       "u8"),
        ("num_of_samples",  "u4"),
        ("resolution",      "u8"),
        ("num_of_channels", "i4"),
        ("waveforms", [("active_channels", "i2"),
                       ("waveform_data", "f4", n_samps)], n_chans)
    ])

def intermediate_dtype(n_chans, n_samps):
    """Intermediate dtype for writing to TTree"""
    return np.dtype([
        ("event_num",       "u4"),
        ("timestamp",       "u8"),
        ("num_of_samples",  "u4"),
        ("resolution",      "u8"),
        ("num_of_channels", "i4"),
        ("active_channels", "i2", (n_chans)),
        ("waveform_data",   "f4", (n_chans,n_samps))
    ])

def header_to_dtype(file_path):
    """Generate numpy dtype from first event in binary header """
    path = Path(file_path)
    with open(path, "rb") as file:
        header = file.read(L_EVENT)
    _, _, n_samps, res, n_chans = struct.unpack("<IQIQi", header)
    print(f"Read event info from first event in {path.name}:")
    print(f" {n_chans} channels")
    print(f" {n_samps} samples")
    print(f" {res} ns resolution")
    return np.dtype([
        ("event_num",       "u4"),
        ("timestamp",       "u8"),
        ("num_of_samples",  "u4"),
        ("resolution",      "u8"),
        ("num_of_channels", "i4"),
        ("waveforms", [("active_channels", "i2"),
                       ("waveform_data", "f4", n_samps)], n_chans)
    ])

def transform_array(data):
    """Convert from binary file dtype to TTree dtype"""
    n_evts, n_chans, n_samps = data["waveforms"]["waveform_data"].shape
    new_dtype = intermediate_dtype(n_chans, n_samps)
    new_data = np.recarray((n_evts,), dtype=new_dtype)
    identical_fields = ["event_num", "timestamp", "num_of_samples", "resolution", "num_of_channels"]
    new_data[identical_fields] = data[identical_fields]
    new_data["active_channels"] = data["waveforms"]["active_channels"]
    new_data["waveform_data"] = data["waveforms"]["waveform_data"].reshape((n_evts, n_chans, n_samps))
    return new_data

def npy_to_h5(bin_files, h5_path, dataset_name='data'):
    start = time.time()
    # Get event structure by reading first event in first bin file
    n_chans, n_samps = read_header(bin_files[0])
    dtype = bin_dtype(n_chans, n_samps)
    # n_evts = 0
    # Create and write to the .h5 file
    with h5py.File(h5_path, 'w') as f:
        # for bin in bin_files:
            # data = np.fromfile(bin, dtype=dtype)
            # print(f"Writing {len(data)} events from {bin.name} to {h5_path.name}")
        # Create a dataset in the HDF5 file and write the data
        # f.create_dataset(dataset_name, data=transform_array(data))
        for i in range(len(bin_files)):
            data = np.fromfile(bin_files[i], dtype=dtype)
            print(f"Writing {len(data)} events from {bin_files[i].name} to {h5_path.name}")
            dt = transform_array(data) 
            if i is 0:
                dset = f.create_dataset(dataset_name, data=dt, maxshape=(None,), chunks=True)
            else:
                current_shape = dset.shape # Get the current shape
                new_shape = (current_shape[0] + dt.shape[0],) # Determine the new shape
                dset.resize(new_shape) # Resize the dataset (this extends the space in the file)
                dset[current_shape[0]:] = dt # Assign the new data to the extended portion
    end = time.time()
    # print(f"Successfully converted {bin_files} to {h5_path} in {end-start:0.2f} seconds")

if __name__ == "__main__":

    completer = PathCompleter()
    bin_dir = prompt("Please enter the absolute path to the binary file directory: ",
                     completer=completer)
    
    bin_path = Path(bin_dir)
    bin_files = []
    if bin_path.is_dir():
        bin_files = sorted(bin_path.glob("*.bin"))
    elif bin_path.suffix == ".bin":
        if bin_path.exists():
            bin_files = [bin_path]
            bin_path = bin_files[0].parent
    if len(bin_files) == 0:
        raise OSError("No bin files found in directory")

    h5_file_name = str(input("h5 file will be saved in the same directory. " \
                               "Please enter a name for the new h5 file: "))
    if not h5_file_name: # if no input / empty string, use the directory name as the h5 filename
        h5_file_name = bin_path.stem
    if not h5_file_name.endswith(".h5"):
        h5_file_name += ".h5"
    h5_file_path = bin_path / h5_file_name

    print(f"{len(bin_files)} bin files found.")

    npy_to_h5(bin_files, h5_file_path, dataset_name='my_dataset')
    #bins_to_h5(bin_files, h5_file_path)
    # Check the h5 file   
    with h5py.File(h5_file_path, 'r') as f:
        print(f"Checking h5 file ...")
        print(f"Keys in h5 file: {list(f.keys())}")
        print(f"Shape of 'dataset': {f['my_dataset'].shape}")
    print("Done")  
