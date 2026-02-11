import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

root_file_path = "/NAS/LAr_TPC_runs/Run64/LArCombo5cmDrift_Run64_UPS_30ch_TPCHV2500_acq1_20260122/" \
                 "LArCombo5cmDrift_Run64_UPS_30ch_TPCHV2500_acq1_20260122_dig2-usb51054.root"

channel_mapping = {
    "sipm_vuv" : [32, 33],
    "sipm_vis" : [30, 31],
    "csp_x"    : [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    "csp_y"    : [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
}


class Event:
    def __init__(self, root_entry, channel_mapping):
        self.event_num = root_entry["event_num"]
        self.timestamp = root_entry["timestamp"]
        self.num_channels = root_entry["num_of_channels"]
        self.num_samples = root_entry["num_of_samples"]
        self.resolution = root_entry["resolution"]*1e-3 # convert to us
        self.channel_mapping = channel_mapping

        # reshape waveform data into a dict
        raw_data = root_entry["waveform_data"].reshape(self.num_channels, self.num_samples)
        self.waveform_data = { ch : raw_data[i,:] for i, ch in
                               enumerate(root_entry["active_channels"]) }
 
        # placeholder for baseline subtracted waveform data
        self.corrected_data = {}

    def baseline_subtract(self, pre_trigger_samples=1500):
        # subtract mean of pre-trigger data from each waveform
        # default 1500 samples for 8 ns resolution, trigger @ t=16us (2000 samples)
        self.corrected_data = { ch : raw_data - np.mean(raw_data[:pre_trigger_samples]) 
                                for ch, raw_data in self.waveform_data.items() }

    def plot_event(self):
        # Create a 2x2 grid
        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1])
        
        # Define subplots
        ax_sipm_vis = plt.subplot(gs[0, 0])  # Top-left (SiPM_VIS)
        ax_sipm_vuv = plt.subplot(gs[0, 1])  # Top-right (SiPM_VUV)
        ax_csp_x = plt.subplot(gs[1, 0])     # Bottom-left (CSP_x)
        ax_csp_y = plt.subplot(gs[1, 1])     # Bottom-right (CSP_y)
        ax_csp_y.sharex(ax_csp_x)

        # Move SIPM_VUV and CSP_y y-axis to the right
        ax_sipm_vuv.yaxis.tick_right()
        ax_sipm_vuv.yaxis.set_label_position("right")
        ax_csp_y.yaxis.tick_right()
        ax_csp_y.yaxis.set_label_position("right")

        time_vals = self.resolution*np.arange(self.num_samples)

        # perform baseline subtraction, if not already done
        if not self.corrected_data:
            self.baseline_subtract()

        csp_offset = 25 # vertical offset between each csp channel

        for ch, data in self.corrected_data:
            if ch in self.channel_mapping["sipm_vuv"]:
                axis = ax_sipm_vuv
                offset = 0
            elif ch in self.channel_mapping["sipm_vis"]:
                axis = ax_sipm_vis
                offset = 0
            elif ch in self.channel_mapping["csp_x"]:
                axis = ax_csp_x
                offset = csp_offset*self.channel_mapping["csp_x"].index(ch)
            elif ch in self.channel_mapping["csp_y"]:
                axis = ax_csp_y
                offset = csp_offset*self.channel_mapping["csp_y"].index(ch)

            axis.plot(time_vals, data + offset)

        # Configure SiPM plots
        for ax_sipm in [ax_sipm_vis, ax_sipm_vuv]:
            ax_sipm.set_ylabel('Output [mV]')
            ax_sipm.set_ylim(-150, 10)
        ax_sipm_vis.set_title('SiPM (VIS)')
        ax_sipm_vuv.set_title('SiPM (VUV)')

        # Configure CSP plot
        for ax_csp in [ax_csp_x, ax_csp_y]:
            ax_csp.set_xlabel('Time [μs]')
            ax_csp.set_ylabel('Output [mV]')
            ax_csp.set_ylim(-50, 550)
        ax_csp_x.set_title('CSP (X-axis)')
        ax_csp_y.set_title('CSP (Y-axis)')

        #fig.suptitle(f"{file_name}\nEvent {self.event_num:04d}")
        plt.tight_layout()
 

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
