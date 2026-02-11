import uproot
import numpy as np
import matplotlib as mpl
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path

root_file_path = "/NAS/GAr_TPC_Runs/Run10/GArCombo5cmDrift_Run10_UPS_33ch_TPCHV500_acq5_20260209/" \
                 "acq5.root"

channel_mapping = {
    "sipm_vuv" : [32, 33],
    "sipm_vis" : [30, 31],
    "csp_x"    : [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    "csp_y"    : [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
}

class WaveformPlotter:
    def __init__(self, csp_offset=25):
        self.is_initialized = False
        self.fig = None
        self.lines = {}
        self.axes = {}
        self.time_vals = None
        self.csp_offset = csp_offset

    def plot_first_event(self, event, title, filename):
        if not event.corrected_data:
            event.baseline_subtract()

        # Create a 2x2 grid
        self.fig = plt.figure(figsize=(12, 10), layout="constrained")
        gs = plt.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1], figure=self.fig)

        # Define subplots
        self.axes = {
            "sipm_vis" : plt.subplot(gs[0, 0]),  # Top-left (SiPM_VIS)
            "sipm_vuv" : plt.subplot(gs[0, 1]),  # Top-right (SiPM_VUV)
            "csp_x"    : plt.subplot(gs[1, 0]),  # Bottom-left (CSP_x)
            "csp_y"    : plt.subplot(gs[1, 1])   # Bottom-right (CSP_y)
        }

        # Configure SiPM plots
        for sl in ["sipm_vis", "sipm_vuv"]:
            self.axes[sl].set_ylabel('Output [mV]')
            self.axes[sl].set_ylim(-150, 10)
        self.axes["sipm_vis"].set_title('SiPM (VIS)')
        self.axes["sipm_vuv"].set_title('SiPM (VUV)')
        self.axes["sipm_vuv"].yaxis.tick_right()
        self.axes["sipm_vuv"].yaxis.set_label_position("right")
        self.axes["sipm_vuv"].sharex(self.axes["sipm_vis"])

        # Configure CSP plot
        for cl in ["csp_x", "csp_y"]:
            self.axes[cl].set_xlabel('Time [μs]')
            self.axes[cl].set_ylabel('Output [mV]')
            self.axes[cl].set_ylim(-50, 550)
        self.axes["csp_x"].set_title('CSP (X-axis)')
        self.axes["csp_y"].set_title('CSP (Y-axis)')
        self.axes["csp_y"].yaxis.tick_right()
        self.axes["csp_y"].yaxis.set_label_position("right")
        self.axes["csp_y"].sharex(self.axes["csp_x"])

        time_vals = event.resolution*np.arange(event.num_samples)

        csp_offset = 25 # vertical offset between each csp channel

        for ch in event.actives:
            for l, ax in self.axes.items():
                if ch in event.channel_mapping[l]:
                    if l in ["csp_x", "csp_y"]:
                        offset = csp_offset*event.channel_mapping[l].index(ch)
                    else:
                        offset = 0
                    self.lines[ch], = ax.plot(time_vals, event.corrected_data[ch] + offset)
                    break

        self.fig.suptitle(title)
        plt.savefig(filename)

        self.is_initialized = True

    def plot_event(self, event, title, filename):
        for ch in event.actives:
            for l, ax in self.axes.items():
                if ch in event.channel_mapping[l]:
                    if l in ["csp_x", "csp_y"]:
                        offset = self.csp_offset*event.channel_mapping[l].index(ch)
                    else:
                        offset = 0
                    self.lines[ch].set_ydata(event.corrected_data[ch] + offset)
                    break
        self.fig.suptitle(title)
        plt.savefig(filename)


class Event:
    def __init__(self, root_entry, channel_mapping):
        self.event_num = root_entry["event_num"][0]
        self.timestamp = root_entry["timestamp"][0]
        self.num_channels = root_entry["num_of_channels"][0]
        self.num_samples = root_entry["num_of_samples"][0]
        self.actives = root_entry["active_channels"][0]
        self.resolution = root_entry["resolution"][0]*1e-3 # convert to us
        self.channel_mapping = channel_mapping

        # reshape waveform data into a dict
        raw_data = root_entry["waveform_data"][0].reshape(self.num_channels, self.num_samples)
        self.waveform_data = { ch : raw_data[i,:] for i, ch in enumerate(self.actives) }
 
        # placeholder for baseline subtracted waveform data
        self.corrected_data = {}

    def baseline_subtract(self, pre_trigger_samples=1500):
        # subtract mean of pre-trigger data from each waveform
        # default 1500 samples for 8 ns resolution, trigger @ t=16us (2000 samples)
        self.corrected_data = { ch : raw_data - np.mean(raw_data[:pre_trigger_samples]) 
                                for ch, raw_data in self.waveform_data.items() }

    def plot_raw_waveforms(self, title, filename, plotter):
        """Plot unfiltered CSP and SiPM waveforms. Performs baseline subtraction if not done
        already. Checking for hits not implemented yet"""

        # perform baseline subtraction, if not already done
        if not self.corrected_data:
            self.baseline_subtract()

        if not plotter.is_initialized:
            plotter.plot_first_event(self, title, filename)
        else:
            plotter.plot_event(self, title, filename)


start = time.time()

mplstyle.use('fast')

with uproot.open(root_file_path) as root_file:
    tree = root_file["test_tree"]

plotter = WaveformPlotter()
for event_arr in tqdm(uproot.iterate(tree, step_size=1, library="np"), total=tree.num_entries):
    event = Event(event_arr, channel_mapping)
    file_name = f"/home/drew/GramsAnalysis/other/drew/plot_test/test_{event.event_num}.png"
    title = f"test plot\nEvent {event.event_num}"
    event.plot_raw_waveforms(title, file_name, plotter)
    #event.plot_waveforms_test(title, file_name)

end = time.time()
print(f"total time: {end - start:0.1f} seconds")
