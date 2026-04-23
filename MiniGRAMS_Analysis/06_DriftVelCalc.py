import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import re
import gc
import detector_config as dc

# --- STYLE ---
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.dpi'] = 300 

# --- PHYSICS CONSTANTS (THE "ROLLING HILLS" TUNE) ---
TARGET_SG_WINDOW_NS = 3000.0  # Keep SG window at 3us to preserve the step
SG_POLY_ORDER = 2             
POST_DERIV_SIGMA_NS = 2000.0  # INCREASED: 2000ns smoothing to kill "grass" and leave "hills"

def extract_voltage(filename):
    match = re.search(r'TPCHV(\d+)', filename, re.IGNORECASE)
    if match: return f"{match.group(1)} V"
    return "Unknown Voltage"

def get_total_response_mV_us(waves, baselines, ch_map, res_ns):
    signal_sum = 0
    dt_us = res_ns / 1000.0
    for ch in ch_map['Charge']:
        if ch < waves.shape[0]:
            trace = waves[ch] - baselines[ch]
            signal_sum += np.sum(trace) * dt_us
    return signal_sum

def process_waveform_joy(trace, res_ns):
    # 1. SG Filter (Clean the Step)
    win_samples = int(TARGET_SG_WINDOW_NS / res_ns)
    if win_samples % 2 == 0: win_samples += 1 
    if win_samples < 3: win_samples = 3
    smoothed = savgol_filter(trace, window_length=win_samples, polyorder=SG_POLY_ORDER)
    
    # 2. Derivative (Current)
    deriv = np.gradient(smoothed)
    
    # 3. HEAVY Post-Derivative Smoothing
    sigma_samples = POST_DERIV_SIGMA_NS / res_ns
    joy_trace = gaussian_filter1d(deriv, sigma=sigma_samples)
    
    return joy_trace

def find_crest_time_constrained(trace_deriv, t_axis, pre_trig_us=16.0):
    """Finds Global Max within physical window."""
    start_mask = t_axis > pre_trig_us
    end_mask = t_axis < (pre_trig_us + 100.0) 
    window_mask = start_mask & end_mask
    
    if np.sum(window_mask) == 0: return 0
    
    window_deriv = trace_deriv[window_mask]
    window_t = t_axis[window_mask]
    
    max_idx = np.argmax(window_deriv)
    peak_val = window_deriv[max_idx]
    
    # Lower sensitivity threshold because heavy smoothing reduces peak amplitude
    if peak_val < 0.01: 
        return 0
        
    return window_t[max_idx]

def generate_joy_plot(pdf, event_id, waves, baselines, ch_map, res_ns, t_axis, gain):
    fig, (ax_sipm, ax_x, ax_y) = plt.subplots(3, 1, figsize=(11, 14), sharex=True,
                                              gridspec_kw={'height_ratios': [1, 4, 4]})
    
    plt.subplots_adjust(hspace=0.1) 

    # --- 1. SiPMs ---
    ax_sipm.set_title(f"Event {event_id} - SiPM Channels", fontsize=12, fontweight='bold')
    ax_sipm.axvline(16.0, color='#00C853', linestyle='--', alpha=0.8, lw=1.5, zorder=0)
    
    light_chs = ch_map['VUV'] + ch_map['VIS']
    colors = ['#E65100', '#1B5E20', '#01579B', '#4A148C'] 
    
    for i, ch in enumerate(light_chs):
        trace = waves[ch] - baselines[ch]
        c = colors[i % len(colors)]
        lbl_type = "VUV" if ch in ch_map['VUV'] else "VIS"
        label_str = f"Ch{ch} ({lbl_type})"
        ax_sipm.plot(t_axis, trace, lw=1.0, alpha=0.9, label=label_str, color=c, rasterized=True, zorder=10)
    
    ax_sipm.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', frameon=False)
    ax_sipm.set_ylabel("Amplitude (mV)")
    ax_sipm.grid(True, linestyle=':', alpha=0.5) 

    # --- HELPER FOR RIDGE PLOTS ---
    def plot_ridge(ax, channels, title):
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_facecolor('white')
        
        processed_traces = []
        for ch in channels:
            raw = waves[ch] - baselines[ch]
            proc = process_waveform_joy(raw, res_ns)
            processed_traces.append(proc)
        
        if not processed_traces: return 0
        
        # Scaling: Use 98th percentile of the HEAVILY SMOOTHED derivative
        vals = np.abs(np.array(processed_traces))
        lane_height = np.percentile(vals, 98) 
        if lane_height == 0: lane_height = 1.0

        spacing = lane_height * 2.5 

        latest_crest_time = 0

        for i, ch in enumerate(channels):
            trace = processed_traces[i]
            
            # Apply Gain
            amplified_trace = trace * gain
            
            y_offset = i * spacing 
            y_data = amplified_trace + y_offset
            
            # Ridge Plot
            ax.fill_between(t_axis, y_data, y_offset, color='white', zorder=i*2, rasterized=True)
            ax.plot(t_axis, y_data, color='black', lw=0.8, zorder=i*2 + 1, rasterized=True)
            ax.text(t_axis[0], y_offset, f"Ch{ch}", color='black', fontsize=8, ha='right', va='center')

            t_crest = find_crest_time_constrained(trace, t_axis, pre_trig_us=16.0)
            if t_crest > latest_crest_time:
                latest_crest_time = t_crest
                    
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axvline(16.0, color='#00C853', linestyle='--', alpha=0.4, lw=1.5, zorder=0)
        
        return latest_crest_time

    t_max_x = plot_ridge(ax_x, ch_map['Charge_X'], f"X-Anode (Current x {gain})")
    t_max_y = plot_ridge(ax_y, ch_map['Charge_Y'], f"Y-Anode (Current x {gain})")
    
    # Drift Time Line
    global_max_t = max(t_max_x, t_max_y)
    if global_max_t > 16.5: 
        for ax in [ax_x, ax_y]:
            ax.axvline(global_max_t, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.9)
            ax.text(global_max_t, ax.get_ylim()[1], f"  t={global_max_t:.1f}us", 
                    color='#D32F2F', fontsize=10, rotation=0, va='bottom', fontweight='bold')

    ax_y.set_xlabel(r"Time ($\mu s$)") 
    ax_x.set_xlim(0, 300) 

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def process_directory(target_dir, mode, threshold, gain):
    print(f"\n--- Processing: {target_dir} ---")
    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.h5')]
    files.sort()
    if not files: return

    pdf = None
    if mode == 'plot':
        out_name = os.path.join(target_dir, "JoyDivision_Waveforms_V13.pdf")
        try:
            pdf = PdfPages(out_name)
            print(f"  > Output PDF: {out_name}")
            print(f"  > Visual Gain: {gain}x")

            count_plotted = 0
            MAX_PLOTS = 50
            
            for filepath in files:
                if count_plotted >= MAX_PLOTS: break
                try:
                    with h5py.File(filepath, 'r') as f:
                        run_config = f.attrs.get('run_config', 'UPS')
                        ch_map = dc.get_channel_map(run_config)
                        res_ns = f.attrs.get('resolution_ns', 8.0) 
                        waves = f['waveforms_mV']
                        
                        if 'baseline_mean_mV' in f:
                            baselines_arr = f['baseline_mean_mV']
                        else:
                            baselines_arr = np.mean(waves[:, :, :20], axis=2)

                        n_events = waves.shape[0]
                        
                        for i in range(n_events):
                            resp = get_total_response_mV_us(waves[i], baselines_arr[i], ch_map, res_ns)
                            
                            if resp > threshold:
                                t_axis = np.arange(waves.shape[2]) * (res_ns / 1000.0) 
                                generate_joy_plot(pdf, i, waves[i], baselines_arr[i], ch_map, res_ns, t_axis, gain)
                                count_plotted += 1
                                if count_plotted % 10 == 0:
                                    print(f"    > Plotted {count_plotted} events...", end='\r')
                                    gc.collect()
                                if count_plotted >= MAX_PLOTS: break
                except Exception as e:
                    print(f"  > Error: {e}")
        finally:
            if pdf: pdf.close()
    
    if mode == 'scan':
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mother_dir')
    parser.add_argument('--mode', '-m', choices=['scan', 'plot'], required=True)
    parser.add_argument('--threshold', '-t', type=float, default=0)
    # INCREASED DEFAULT GAIN to 20.0
    parser.add_argument('--gain', '-g', type=float, default=20.0, help="Visual Boost (Default 20.0)")
    args = parser.parse_args()
    
    target_dirs = []
    for root, dirs, files in os.walk(args.mother_dir):
        if any(f.endswith('.h5') for f in files):
            target_dirs.append(root)
    target_dirs.sort()
    
    for d in target_dirs:
        process_directory(d, args.mode, args.threshold, args.gain)

if __name__ == "__main__":
    main()