import uproot
import numpy as np
import struct
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# hardcode data path for testing for now
#filename = ("/NAS/GAr_TPC_Runs/Run5/GArCombo5cmDrift_Run5_UPS_60ch_TPCHV0_pedestal1_20251022/"
#           "GArCombo5cmDrift_Run5_UPS_60ch_TPCHV0_pedestal1_20251022_dig2-usb51054_20251022140656-04.bin")

## field sizes in bytes - in this order
L_EVENT = 28 # 4 + 8 + 4 + 8 + 4 bytes


def header_to_dtype(file_path):
    """Generate numpy dtype from first event in binary header """

    path = Path(file_path)
    with open(path, "rb") as file:
        ## read out a portion of the first event to get channel & sampling info
        header = file.read(L_EVENT)

    _, _, n_samps, res, n_chans = struct.unpack("<iqiqi", header)

    print(f"Read event info from first event in {path.name}:")
    print(f" {n_chans} channels")
    print(f" {n_samps} samples")
    print(f" {res} ns resolution")

    return np.dtype([
        ("event_num", "i4"),
        ("timestamp", "i8"),
        ("num_of_samples", "i4"),
        ("resolution", "i8"),
        ("num_of_channels", "i4"),
        ("active_channels", "i2", (n_chans)),
        ("waveform_data", "f4", (n_chans*n_samps))
    ])





def bin_to_array(file_path, dtype):
    """Read a single binary file into numpy array"""

    path = Path(file_path)
    #print(f"Reading data from {path.name} into numpy array")
    data = np.fromfile(file_path, dtype=dtype)
    #print(f"{len(data)} events read")

    return data


def bins_to_root(bin_files, root_filename):
    dtype = header_to_dtype(bin_files[0])
    ttree_branch_types = {
        "event_num" :       "i4",
        "timestamp" :       "i8",
        "num_of_samples" :  "i4",
        "resolution" :      "i4",
        "num_of_channels" : "i4",
        "active_channels" : dtype["active_channels"],
        "waveform_data" :   dtype["waveform_data"]
    }
    with uproot.recreate(root_filename) as outfile:
        outfile.mktree("test_tree", ttree_branch_types, title="test_tree")
        n_evts = 0
        all_arrays = []
        for bin in bin_files:
            data = bin_to_array(bin, dtype)
            all_arrays.append(data)
            n_evts += len(data)
            print(f"Writing {len(data)} events from {bin.name} to {root_filename.name}")
            #outfile["test_tree"].extend(data)
        all_data = np.concatenate(all_arrays)
        outfile["test_tree"].extend(all_data)
        print(f"{n_evts} total events written to {root_filename.name}")
        

if __name__ == "__main__":

    completer = PathCompleter()

    bin_dir = prompt("Please enter the absolute path to the binary file directory: ", completer=completer)
    root_file_name = str(input("Please enter a file name for the new root file: "))

    if not root_file_name.endswith(".root"):
        root_file_name += ".root"

    bin_dir_path = Path(bin_dir)
    bin_files = sorted(bin_dir_path.glob("*.bin"))
    root_file_path = bin_dir_path / root_file_name

    bins_to_root(bin_files, root_file_path)

 

