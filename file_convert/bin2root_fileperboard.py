import uproot
import numpy as np
import struct
import time
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from tqdm import tqdm

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
        ("waveform_data",   "f4", (n_chans*n_samps))
    ])

def ttree_branch_types(n_chans, n_samps):
    """Branch names and type definitions for writing to TTree"""
    return {
        "event_num" :       np.dtype("i4"),
        "timestamp" :       np.dtype("u8"),
        "num_of_samples" :  np.dtype("i4"),
        "resolution" :      np.dtype("i4"),
        "num_of_channels" : np.dtype("i4"),
        "active_channels" : np.dtype(("i4", n_chans)),
        "waveform_data" :   np.dtype(("f4", n_chans*n_samps))
    }

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
    new_data["waveform_data"] = data["waveforms"]["waveform_data"].reshape((n_evts, n_chans*n_samps))
    return new_data

def split_array(array, max_bytes=None):
    """Splits the array into multiple arrays of (approximately equal) size less than max_bytes.
    If max_bytes is None, or smaller than a single event, maximally split array (one array per
    event).
    """
    n_entries = len(array)
    n_bytes = array.nbytes
    if max_bytes is None:
        splits = n_entries
    else:
        splits = n_bytes // max_bytes
        if splits > n_entries:
            splits = n_entries
    return np.array_split(array, int(splits))

def bins_to_root(bin_files, root_path, max_basket_size=None):
    """Take list of bin file paths and write all data to a single root file with TBaskets of size
    max_basket_size (in bytes). If max_basket_size is None, then split into one TBasket per event"""
    start = time.time()

    # Get event structure by reading first event in first bin file
    n_chans, n_samps = read_header(bin_files[0])
    dtype = bin_dtype(n_chans, n_samps)
    n_evts = 0
    with uproot.recreate(root_path) as outfile:
        outfile.mktree("test_tree", ttree_branch_types(n_chans, n_samps), title="test_tree")
        for bin in bin_files:
            data = np.fromfile(bin, dtype=dtype)
            print(f"Writing {len(data)} events from {bin.name} to {root_path.name}")
            t = tqdm(total=len(data))
            split_data = split_array(data, max_basket_size)
            for d in split_data:
                outfile["test_tree"].extend(transform_array(d))
                n_evts += len(d)
                t.update(len(d))
            t.close()
    end = time.time()
    print(f"{n_evts} total events written to {root_path} in {end-start:0.2f} seconds")


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

    root_file_name = str(input("ROOT file will be saved in the same directory. " \
                               "Please enter a name for the new ROOT file: "))
    if not root_file_name: # if no input / empty string, use the directory name as the root filename
        root_file_name = bin_path.stem
    if not root_file_name.endswith(".root"):
        root_file_name += ".root"
    root_file_path = bin_path / root_file_name

    print(f"{len(bin_files)} bin files found.")

    bins_to_root(bin_files, root_file_path)
