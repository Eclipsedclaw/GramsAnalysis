import h5py
import numpy as np
import argparse
import os
import detector_config as dc
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import PolyCollection

# --- PHYSICS TUNING ---
TARGET_SG_WINDOW_NS = 3000.0  
SG_POLY_ORDER = 2             
POST_DERIV_SIGMA_NS = 2000.0  
DRIFT_VELOCITY_MM_US = 0.9 # Approx for 250V/5cm

def process_waveform_joy(trace, res_ns):
    win_samples = int(TARGET_SG_WINDOW_NS / res_ns)
    if win_samples % 2 == 0: win_samples += 1 
    if win_samples < 3: win_samples = 3
    smoothed = savgol_filter(trace, window_length=win_samples, polyorder=SG_POLY_ORDER)
    deriv = np.gradient(smoothed)
    joy_trace = gaussian_filter1d(deriv, sigma=POST_DERIV_SIGMA_NS / res_ns)
    return joy_trace

def polygon_under_graph(x, y):
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def render_joy_3d(fig, event_id, waves, baselines, ch_map, res_ns, gain, show_mm):
    ax = fig.add_subplot(111, projection='3d')
    
    t_axis = np.arange(waves.shape[1]) * (res_ns / 1000.0)
    mask = (t_axis >= 0) & (t_axis <= 80.0)
    t_crop = t_axis[mask]
    
    # Optional: Convert to mm
    if show_mm:
        x_axis = t_crop * DRIFT_VELOCITY_MM_US
        x_label = "Drift Distance (mm)"
    else:
        x_axis = t_crop
        x_label = "Time (µs)"

    # Theme Colors
    fill_color = 'black'
    line_color = 'white'
    
    # We only plot the ACTIVE plane (find which one has more signal)
    # Or plot both? Let's stick to X-Anode for the "Hero Shot" or let user choose.
    # For now, defaults to X-Anode (Channels 4-18) as they often look cleanest.
    channels = ch_map['Charge_X']
    
    verts = []
    for ch in reversed(channels):
        raw = waves[ch] - baselines[ch]
        proc = process_waveform_joy(raw[mask], res_ns) * gain
        verts.append(polygon_under_graph(x_axis, proc))

    poly = PolyCollection(verts, facecolors=fill_color, edgecolors=line_color, linewidths=1.2, alpha=1.0)
    ax.add_collection3d(poly, zs=range(len(channels)), zdir='y')

    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(-1, len(channels))
    ax.set_zlim(0, 5)
    
    ax.set_xlabel(x_label, color='white', labelpad=10)
    ax.set_ylabel('Channel Index', color='white', labelpad=10)
    ax.set_zlabel('Current (arb)', color='white', labelpad=10)
    
    # Clean up axes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors='white')
    
    # Initial View
    ax.view_init(elev=25, azim=-80)
    ax.set_title(f"Event {event_id}: Joy Division Reconstruction", color='white', fontsize=16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--event', '-e', type=int, required=True)
    parser.add_argument('--gain', '-g', type=float, default=20.0)
    parser.add_argument('--interactive', '-i', action='store_true', help="Open interactive window")
    parser.add_argument('--mm', action='store_true', help="Show Drift Distance (mm) instead of Time")
    args = parser.parse_args()
    
    # Backend Selection
    import matplotlib
    if args.interactive:
        try:
            matplotlib.use('Qt5Agg')
        except:
            matplotlib.use('TkAgg') # Fallback
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')

    # Load Data
    with h5py.File(args.filename, 'r') as f:
        run_config = f.attrs.get('run_config', 'UPS')
        res_ns = f.attrs.get('resolution_ns', 8.0)
        ch_map = dc.get_channel_map(run_config)
        waves = f['waveforms_mV'][args.event]
        baselines = f['baseline_mean_mV'][args.event] if 'baseline_mean_mV' in f else np.mean(waves[:,:20], axis=1)

    fig = plt.figure(figsize=(14, 10))
    render_joy_3d(fig, args.event, waves, baselines, ch_map, res_ns, args.gain, args.mm)

    if args.interactive:
        print("Interactive Mode: Use mouse to rotate. Close window to exit.")
        plt.show()
    else:
        out_name = f"HeroShot_Evt{args.event}.pdf"
        plt.savefig(out_name, facecolor='black', bbox_inches='tight')
        print(f"Saved static plot to {out_name}")

if __name__ == "__main__":
    main()