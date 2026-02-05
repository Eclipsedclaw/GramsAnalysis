import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

root_file_path = "/NAS/LAr_TPC_runs/Run64/LArCombo5cmDrift_Run64_UPS_30ch_TPCHV2500_acq1_20260122/" \
                 "LArCombo5cmDrift_Run64_UPS_30ch_TPCHV2500_acq1_20260122_dig2-usb51054.root"

with uproot.open(root_file_path) as root_file:
    tree = root_file["test_tree"]

for event in tqdm(uproot.iterate(tree, step_size=1, library="np"), total=tree.num_entries):
    event_num = event["event_num"][0]
    n_chans = event["num_of_channels"][0]
    n_samps = event["num_of_samples"][0]
    waveforms_flat = event["waveform_data"][0]
    waveforms = waveforms_flat.reshape(n_chans, n_samps)
    active_chans = event["active_channels"][0]
    res = event["resolution"][0]*1e-3 # convert to us
    t_vals = res*np.arange(n_samps)

    plt.figure()
    for i, ch in enumerate(active_chans):
        wf = waveforms[i,:]
        plt.plot(t_vals, wf)
    plt.savefig(f"/home/drew/GramsAnalysis/other/drew/plot_test/test_{event_num}.png")
    plt.close()
