import uproot
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from enum import Enum
from tqdm import tqdm
from pathlib import Path


class ChType(Enum):
    SIPM_VIS = "sipm_vis"
    SIPM_VUV = "sipm_vuv"
    CSP_X = "csp_x"
    CSP_Y = "csp_y"

    @classmethod
    def sipm_channels(cls):
        return cls.SIPM_VIS, cls.SIPM_VUV
    
    @classmethod
    def csp_channels(cls):
        return cls.CSP_X, cls.CSP_Y


class Acquisition:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise Exception(f"Data directory {base_dir} not found")
        self.name = self.base_dir.stem # acquisition name from base directory
        self.channel_mapping = {}
        self.root_tree = None

    def load_channel_mapping(self, file_path=""):
        """Read channel mapping spec from yaml file and create mapping dictionary
        
        Mapping dict is rearranged from { ch_type : [list_of_chans] } syntax (as in yaml file) to 
        { ch_num : ch_type } to improve lookup speed."""

        # check file
        if file_path:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_dir / path
        else:
            path = self.base_dir / "channel_mapping.yaml"
        if not path.exists():
            raise Exception(f"{path} not found")

        # load yaml config
        channel_mapping = {}
        with open(path) as f:
            yaml_mapping = yaml.safe_load(f)
        
        for ch_type_str, val in yaml_mapping.items():
            # validate channel type
            try: 
                ch_type = ChType(ch_type_str)
            except ValueError as e:
                msg = f"Channel mapping file {path} contains invalid channel type: {ch_type_str}"
                raise Exception(msg) from e
            
            # parse channel list / range
            if isinstance(val, list): # list of channels
                chan_list = val
            elif isinstance(val, dict): # range of channels
                try:
                    chan_list = range(val["range_start"], val["range_end"]+1) # inclusive start and end
                except KeyError as e:
                    msg = f"Invalid channel specification for {ch_type_str} in {path}"
                    raise Exception(msg) from e
            elif isinstance(val, int):
                chan_list = [val]
            else:
                msg = f"Invalid channel specification for {ch_type_str} in {path}: {val}"
                raise Exception(msg)
            
            # validate channel numbers
            for ch in chan_list:
                try:
                    channel_mapping[int(ch)] = ch_type
                except TypeError as e:
                    msg = f"Invalid {ch_type_str} channel in {path}: {ch}"
                    raise Exception(msg) from e
    
        self.channel_mapping = channel_mapping

    def load_root_tree(self, file_path=""):
        if file_path: # if file path is specified
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_dir / path
        else: # if not specified, look for a single root file in base directory
            root_files = list(self.base_dir.glob("*.root"))
            if len(root_files) == 0:
                raise Exception(f"No root files found in {self.base_dir}. Specify filename")
            if len(root_files) > 1:
                raise Exception(f"Multiple root files found in {self.base_dir}. Specify filename")
            path = root_files[0]
        if not path.exists():
            raise Exception(f"{path} not found")

        with uproot.open(path) as f:
            self.root_tree = f["test_tree"]


    def plot_all_events(self, plot_dir="", csp_offset=25):
        if plot_dir: # if file path is specified
            path = Path(plot_dir)
            if not path.is_absolute():
                path = self.base_dir / path
        else: # otherwise, create a directory for plots
            path = self.base_dir / "raw_waveforms"
            path.mkdir(exist_ok=True)
        
        if not self.root_tree:
            raise Exception("Load root file before plotting event waveforms")
        if not self.channel_mapping:
            raise Exception("Load channel mapping before plotting event waveforms")

        mplstyle.use('fast')

        # Create a 2x2 grid
        fig = plt.figure(figsize=(12, 10), layout="constrained")
        gs = plt.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1], figure=fig)

        # Define subplots
        axes = {
            ChType.SIPM_VIS : plt.subplot(gs[0, 0]),  # Top-left (SiPM_VIS)
            ChType.SIPM_VUV : plt.subplot(gs[0, 1]),  # Top-right (SiPM_VUV)
            ChType.CSP_X    : plt.subplot(gs[1, 0]),  # Bottom-left (CSP_x)
            ChType.CSP_Y    : plt.subplot(gs[1, 1])   # Bottom-right (CSP_y)
        }

        # Configure SiPM plots
        for sl in ChType.sipm_channels():
            axes[sl].set_ylabel('Output [mV]')
            axes[sl].set_ylim(-150, 10)
        axes[ChType.SIPM_VIS].set_title('SiPM (VIS)')
        axes[ChType.SIPM_VUV].set_title('SiPM (VUV)')
        axes[ChType.SIPM_VUV].yaxis.tick_right()
        axes[ChType.SIPM_VUV].yaxis.set_label_position("right")
        axes[ChType.SIPM_VUV].sharex(axes[ChType.SIPM_VIS])

        # Configure CSP plots
        for cl in ChType.csp_channels():
            axes[cl].set_xlabel('Time [μs]')
            axes[cl].set_ylabel('Output [mV]')
            axes[cl].set_ylim(-50, 550)
        axes[ChType.CSP_X].set_title('CSP (X-axis)')
        axes[ChType.CSP_Y].set_title('CSP (Y-axis)')
        axes[ChType.CSP_Y].yaxis.tick_right()
        axes[ChType.CSP_Y].yaxis.set_label_position("right")
        axes[ChType.CSP_Y].sharex(axes[ChType.CSP_X])

        # Iterate over events
        lines = {} # container for plot artists
        for i, ev_arr in enumerate(tqdm(uproot.iterate(self.root_tree, step_size=1, library="np"), 
                                      total=self.root_tree.num_entries)):
            event = Event(ev_arr)
            data = event.baseline_subtract()

            if i == 0: # first event
                time_vals = event.resolution*np.arange(event.num_samples)
            
            offset_csp_x = 0
            offset_csp_y = 0
            for ch in sorted(self.channel_mapping.keys()):
                if ch not in event.actives:
                    msg = f"No waveform data for channel {ch} specified in channel mapping"
                    raise Exception(msg)
                ch_type = self.channel_mapping[ch]
                if ch_type == ChType.CSP_X:
                    offset = offset_csp_x
                    offset_csp_x += csp_offset
                elif ch_type == ChType.CSP_Y:
                    offset = offset_csp_y
                    offset_csp_y += csp_offset
                else:
                    offset = 0

                if ch in lines: # already plotted this channel before - just replace y data
                    lines[ch].set_ydata(data[ch] + offset)
                else: # first event, need to initialize plot artist
                    lines[ch], = axes[ch_type].plot(time_vals, data[ch] + offset)

            event_id = f"{i:04d}_{event.event_num:04d}"
            fig.suptitle(f"{self.name}\nEvent {event_id}")
            plt.savefig(path / f"{event_id}.png")
        
        plt.close(fig)


class Event:
    def __init__(self, root_entry):
        self.event_num = root_entry["event_num"][0]
        self.timestamp = root_entry["timestamp"][0]
        self.num_channels = root_entry["num_of_channels"][0]
        self.num_samples = root_entry["num_of_samples"][0]
        self.actives = root_entry["active_channels"][0]
        self.resolution = root_entry["resolution"][0]*1e-3 # convert to us

        # reshape waveform data into a dict
        raw_data = root_entry["waveform_data"][0].reshape(self.num_channels, self.num_samples)
        self.waveform_data = { ch : raw_data[i,:] for i, ch in enumerate(self.actives) }
 
    def baseline_subtract(self, pre_trigger_samples=1500):
        """Subtract mean of pre-trigger data from each waveform
        Default to first 1500 samples for 8 ns resolution, trigger @ t=16us (2000 samples)"""
        return { ch : raw_data - np.mean(raw_data[:pre_trigger_samples]) for ch, raw_data in 
                 self.waveform_data.items() }


if __name__ == "__main__":

    start = time.time()

    data_path = "/NAS/GAr_TPC_Runs/Run10/GArCombo5cmDrift_Run10_UPS_33ch_TPCHV500_acq5_20260209/"
    channel_mapping_path = "/home/drew/GramsAnalysis/other/drew/channel_mapping.yaml"

    acq = Acquisition(data_path)
    acq.load_channel_mapping(channel_mapping_path)
    acq.load_root_tree()
    acq.plot_all_events("/home/drew/GramsAnalysis/other/drew/plot_test")
