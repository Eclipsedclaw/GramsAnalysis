"""
33_Mass_Plotter_Coinc_Text.py
LArTPC Mass PDF Compiler (Coincidence Filtered)
- Thinner trigger lines on SiPM panels.
- Bounding box text showing number of active channels per panel.
- UPDATED FOR RUN 7: Board 85 is Slave 1, Board 75 is Slave 2.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse
import time
import gc
from multiprocessing import Pool
from tqdm import tqdm
import detector_config as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_event_to_pdf(f, event_idx, mapping, filename, pdf, charge_thresh, light_thresh):
    vuv_chans = mapping["VUV"]
    vis_chans = mapping["VIS"]
    x_chans = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"]
    y_chans = mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]

    master_dt = f['Board_72'].attrs['sampling_period_ns']
    slave_dt = f['Board_85'].attrs['sampling_period_ns']
    charge_max_t = (f['Board_85'].attrs['n_samples'] * slave_dt) / 1000.0
    pre_trig_us = f['Board_72'].attrs.get('baseline_pre_trigger_us', 16.0)

    trigger_ratio = pre_trig_us / charge_max_t
    sipm_window_width = 15.0
    sipm_xmin = pre_trig_us - (sipm_window_width * trigger_ratio)
    sipm_xmax = sipm_xmin + sipm_window_width

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    
    def get_trace(global_ch):
        if global_ch < 32: grp = f['Board_72']; local_ch = global_ch
        elif global_ch < 96: grp = f['Board_85']; local_ch = global_ch - 32
        else: grp = f['Board_75']; local_ch = global_ch - 96
        
        dt_ns = grp.attrs['sampling_period_ns']
        trace = grp['waveforms'][event_idx, local_ch, :].copy() 
        if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_idx, local_ch]):
            trace -= grp['baseline_mean'][event_idx, local_ch]
        return np.arange(len(trace)) * dt_ns / 1000.0, trace

    def count_active(channels, thresh):
        active = 0
        for ch in channels:
            _, v = get_trace(ch)
            if not np.all(np.isnan(v)) and np.nanmax(np.abs(v)) > thresh:
                active += 1
        return active

    act_vuv = count_active(vuv_chans, light_thresh)
    act_vis = count_active(vis_chans, light_thresh)
    act_x = count_active(x_chans, charge_thresh)
    act_y = count_active(y_chans, charge_thresh)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')

    # --- Top Row: SiPMs ---
    for ch in vuv_chans:
        t, v = get_trace(ch)
        axes[0,0].plot(t, v, alpha=0.5, linewidth=0.8, rasterized=True)
    axes[0,0].set_title(f"VUV SiPMs", fontweight='bold')
    axes[0,0].text(0.02, 0.95, f"Active: {act_vuv}/{len(vuv_chans)}\n(>{light_thresh}mV)", transform=axes[0,0].transAxes, va='top', bbox=props)
    
    for ch in vis_chans:
        t, v = get_trace(ch)
        axes[0,1].plot(t, v, alpha=0.5, linewidth=0.8, rasterized=True)
    axes[0,1].set_title(f"VIS SiPMs", fontweight='bold')
    axes[0,1].text(0.02, 0.95, f"Active: {act_vis}/{len(vis_chans)}\n(>{light_thresh}mV)", transform=axes[0,1].transAxes, va='top', bbox=props)
    
    axes[0,0].set_xlim(sipm_xmin, sipm_xmax); axes[0,1].set_xlim(sipm_xmin, sipm_xmax)
    vuv_y, vis_y = axes[0,0].get_ylim(), axes[0,1].get_ylim()
    sipm_min, sipm_max = min(vuv_y[0], vis_y[0]), max(vuv_y[1], vis_y[1])
    axes[0,0].set_ylim(sipm_min, sipm_max); axes[0,1].set_ylim(sipm_min, sipm_max)
    
    # --- Bottom Row: Charge ---
    x_min, x_max = np.inf, -np.inf
    for ch in x_chans:
        t, v = get_trace(ch)
        axes[1,0].plot(t, v, alpha=0.4, linewidth=0.6, rasterized=True)
        if not np.all(np.isnan(v)): x_min, x_max = min(x_min, np.nanmin(v)), max(x_max, np.nanmax(v))
    axes[1,0].set_title(f"Charge X-Direction", fontweight='bold')
    axes[1,0].text(0.02, 0.95, f"Active: {act_x}/{len(x_chans)}\n(>{charge_thresh}mV)", transform=axes[1,0].transAxes, va='top', bbox=props)
    
    y_min, y_max = np.inf, -np.inf
    for ch in y_chans:
        t, v = get_trace(ch)
        axes[1,1].plot(t, v, alpha=0.4, linewidth=0.6, rasterized=True)
        if not np.all(np.isnan(v)): y_min, y_max = min(y_min, np.nanmin(v)), max(y_max, np.nanmax(v))
    axes[1,1].set_title(f"Charge Y-Direction", fontweight='bold')
    axes[1,1].text(0.02, 0.95, f"Active: {act_y}/{len(y_chans)}\n(>{charge_thresh}mV)", transform=axes[1,1].transAxes, va='top', bbox=props)

    axes[1,0].set_xlim(0, charge_max_t); axes[1,1].set_xlim(0, charge_max_t)
    c_min, c_max = min(x_min, y_min), max(x_max, y_max)
    axes[1,0].set_ylim(min(-10, c_min - 2), max(20, c_max + 5))
    axes[1,1].set_ylim(min(-10, c_min - 2), max(20, c_max + 5))

    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlabel("Time (us)"); ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, linestyle='--', alpha=0.5)
        if pre_trig_us > 0: 
            # THINNER lines for SiPMs (idx 0 and 1), normal for charge
            lw = 0.8 if idx < 2 else 1.5 
            ax.axvline(pre_trig_us, color='black', linestyle='--', linewidth=lw, alpha=0.8)

    fig.suptitle(f"Event: {event_idx} | File: {filename}\n(Coincidence Filtered & Aligned)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def process_acquisition(task):
    acq_dir, h5_files, limit, charge_thresh, light_thresh = task
    acq_name = os.path.basename(acq_dir)
    h5_files.sort()
    
    save_dir = os.path.join(acq_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"{acq_name}_AlignedCoincidencePlot_V2.pdf")
    mapping = dc.get_channel_map("MINIG")

    events_plotted = 0
    try:
        with PdfPages(pdf_path) as pdf, tqdm(total=limit, desc=acq_name, unit="evt") as pbar:
            for f_path in h5_files:
                if events_plotted >= limit: break
                filename = os.path.basename(f_path)
                try:
                    with h5py.File(f_path, 'r') as f:
                        if 'baseline_mean' not in f['Board_72']: continue
                        n_in_file = f['Board_72']['waveforms'].shape[0]
                        w75, b75 = f['Board_75']['waveforms'], f['Board_75'].get('baseline_mean', None)
                        w85, b85 = f['Board_85']['waveforms'], f['Board_85'].get('baseline_mean', None)
                        
                        for i in range(n_in_file):
                            if events_plotted >= limit: break
                            if np.isnan(w75[i, 0, 0]) or np.isnan(w85[i, 0, 0]): continue 
                            
                            has_charge = False
                            for ch in range(64):
                                trace = w85[i, ch] if b85 is None else w85[i, ch] - b85[i, ch]
                                if np.nanmax(np.abs(trace)) > charge_thresh: has_charge = True; break
                            if not has_charge:
                                for ch in range(60):
                                    trace = w75[i, ch] if b75 is None else w75[i, ch] - b75[i, ch]
                                    if np.nanmax(np.abs(trace)) > charge_thresh: has_charge = True; break
                            
                            if not has_charge: continue 
                            
                            plot_event_to_pdf(f, i, mapping, filename, pdf, charge_thresh, light_thresh)
                            events_plotted += 1
                            pbar.update(1)
                            if events_plotted % 10 == 0: gc.collect()
                except Exception as e: print(f"\n[ERR] {filename}: {e}"); continue
    except Exception as e: print(f"\n[FAIL] Generating PDF: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_target')
    parser.add_argument('--events', '-e', type=int, default=100)
    parser.add_argument('--charge_thresh', '-c', type=float, default=10.0)
    parser.add_argument('--light_thresh', '-l', type=float, default=5.0)
    parser.add_argument('--workers', '-w', type=int, default=4)
    args = parser.parse_args()
    
    acq_groups = {}
    if os.path.isfile(args.input_target): acq_groups[os.path.dirname(args.input_target)] = [args.input_target]
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5s = [os.path.join(root, f) for f in files if f.endswith('.h5')]
            if h5s: acq_groups[root] = h5s
    
    tasks = [(acq_dir, files, args.events, args.charge_thresh, args.light_thresh) for acq_dir, files in acq_groups.items()]
    if tasks:
        with Pool(min(args.workers, len(tasks))) as p: p.map(process_acquisition, tasks)

if __name__ == "__main__": main()