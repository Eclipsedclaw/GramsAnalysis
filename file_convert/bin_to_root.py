import uproot
import numpy as np
import struct
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# hardcode data path for testing for now
#filename = ("/NAS/GAr_TPC_Runs/Run5/GArCombo5cmDrift_Run5_UPS_60ch_TPCHV0_pedestal1_20251022/"
#           "GArCombo5cmDrift_Run5_UPS_60ch_TPCHV0_pedestal1_20251022_dig2-usb51054_20251022140656-04.bin")

completer = PathCompleter()

filename = prompt("Please enter the absolute path to the binary file: ", completer=completer)


## field sizes in bytes - in this order
L_header = 28 # 4 + 8 + 4 + 8 + 4 bytes

with open(filename, "rb") as file:
    ## read out a portion of the first event to get channel & sampling info
    header_bin = file.read(L_header)
    _, _, n_samps, res, n_chans = struct.unpack("<iqiqi", header_bin)

    print("Read event info from first event:")
    print(f" {n_chans} channels")
    print(f" {n_samps} samples")
    print(f" {res} ns resolution")

file_read_dtype = np.dtype([
    ("event_num", "i4"),
    ("timestamp", "i8"),
    ("num_of_samples", "i4"),
    ("resolution", "i8"),
    ("num_of_channels", "i4"),
    ("active_channels", "i2", (n_chans)),
    ("waveform_data", "f4", (n_chans*n_samps))
])

print("Reading data into numpy array")
data = np.fromfile(filename, dtype=file_read_dtype)
print(f"{len(data)} events read")

## create type specification for numpy structured array, which uproot will
## convert to TTree branches
ttree_write_dtype = [
    ("event_num", "i4"),
    ("timestamp", "i8"),
    ("num_of_samples", "i4"),
    ("resolution", "i4"),
    ("num_of_channels", "i4"),
    ("active_channels", f"{n_chans} * i4"),
    ("waveform_data", f"{n_chans*n_samps} * f8")
]

print("Writing TTree to ROOT file")
#out_data = data.astype(np.dtype(ttree_write_dtype))

outfile_name = "drew_test.root"

ttree_branch_types = {
    "event_num" :       "i4",
    "timestamp" :       "i8",
    "num_of_samples" :  "i4",
    "resolution" :      "i4",
    "num_of_channels" : "i4",
    "active_channels" : np.dtype(("i4", n_chans)),
    "waveform_data" :   np.dtype(("i4", n_chans*n_samps))
}

with uproot.recreate(outfile_name) as outfile:
    #outfile["test_tree"] = out_data
    outfile.mktree("test_tree", ttree_branch_types, title="test_tree")
    #outfile["test_tree"].show()
    outfile["test_tree"].extend(data)
    #outfile["test_tree"].show()

