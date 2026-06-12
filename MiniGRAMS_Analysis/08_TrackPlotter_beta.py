"""
36_TrackPlotter_Joy3D_Run7.py
LArTPC 3D Track Visualizer (Joy Division Style)
- Generates 3D PolyCollection ribbons for a specific event.
- DUAL-PLANE TUNING: Takes separate SG and Gaussian parameters for X and Y anodes.
- Labels channels physically (e.g., A0, B13) on the depth axis.
- UPDATED FOR RUN 7: Board 85 is Slave 1, Board 75 is Slave 2.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import detector_config as dc
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- THEME: DARK MODE (JOY DIVISION) ---
plt.style.use('dark_background')
LINE_COLOR = 'white'
FILL_COLOR = 'black' 
AXIS_COLOR = 'white'

def get_charge_trace(f, event_id, global_ch):
    """Safely extracts a baseline-corrected charge trace for Run 7."""
    if global_ch < 32: return None, None 
    # RUN 7 SWAP: Board 85 is now Slave 1
    elif global_ch < 96: grp, local_ch = f['Board_85'], global_ch - 32
    # RUN 7 SWAP: Board 75 is now Slave 2
    else: grp, local_ch = f['Board_75'], global_ch - 96

    dt_ns = grp.attrs['sampling_period_ns']
    trace = grp['waveforms'][event_id, local_ch].copy()
    if np.isnan(trace[0]): return None, None
    
    if 'baseline_mean' in grp and not np.isnan(grp['baseline_mean'][event_id, local_ch]):
        bl = grp['baseline_mean'][event_id, local_ch]
    else:
        bl = np.nanmedian(trace[:1000])
        
    return trace - bl, dt_ns

def process_waveform_joy(trace, res_ns, sg_win_ns, gauss_sigma_ns):
    win_samples = int(sg_win_ns / res_ns)
    if win_samples % 2 == 0: win_samples += 1 
    if win_samples < 3: win_samples = 3
    
    try:
        smoothed = savgol_filter(trace, window_length=win_samples, polyorder=2)
    except ValueError:
        return np.zeros_like(trace)
        
    deriv = np.gradient(smoothed)
    joy_trace = gaussian_filter1d(deriv, sigma=gauss_sigma_ns / res_ns)
    
    return joy_trace

def polygon_under_graph(x, y):
    """Constructs vertices for the filled ribbon."""
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def render_joy_3d(pdf, f, event_id, sg_x, gauss_x, sg_y, gauss_y, gain):
    print(f"Rendering Joy Division 3D for Event {event_id}...")
    
    # Grab timing from Slave 1 (Board 85)
    res_ns = f['Board_85'].attrs['sampling_period_ns']
    n_samples = f['Board_85'].attrs['n_samples']
    t_axis = np.arange(n_samples) * (res_ns / 1000.0)
    
    # Crop to the physical window to avoid plotting long flat dead-times
    mask = (t_axis >= 0) & (t_axis <= 220.0)
    t_crop = t_axis[mask]
    
    ch_map = dc.get_channel_map("MINIG")
    
    # Build Planes with physical labels (A0, A1... B0... C13)
    def build_plane(banks):
        chs, lbls = [], []
        for bank_name, bank_list in banks:
            for i, ch in enumerate(bank_list):
                chs.append(ch)
                lbls.append(f"{bank_name}{i}")
        return chs, lbls

    x_chs, x_lbls = build_plane([('A', ch_map['Charge_A']), ('B', ch_map['Charge_B']), ('C', ch_map['Charge_C'])])
    y_chs, y_lbls = build_plane([('D', ch_map['Charge_D']), ('E', ch_map['Charge_E']), ('F', ch_map['Charge_F'])])
    
    planes = [('X-Anode', x_chs, x_lbls), ('Y-Anode', y_chs, y_lbls)]
    
    for plane_name, channels, labels in planes:
        if not channels: continue
        
        # Route the correct optimal parameters
        if plane_name == 'X-Anode':
            sg_win = sg_x
            gauss_sig = gauss_x
        else:
            sg_win = sg_y
            gauss_sig = gauss_y
            
        # Reverse order so Channel 0 is "closest" to camera
        channels_rev = list(reversed(channels))
        labels_rev = list(reversed(labels))
        
        verts = []
        for ch in channels_rev:
            raw, res = get_charge_trace(f, event_id, ch)
            
            if raw is None:
                # Insert flat line for dead/dropped channels to maintain spatial grid
                verts.append(polygon_under_graph(t_crop, np.zeros_like(t_crop)))
                continue
                
            raw_crop = raw[mask]
            proc = process_waveform_joy(raw_crop, res, sg_win, gauss_sig)
            proc = proc * gain
            
            verts.append(polygon_under_graph(t_crop, proc))

        # Setup 3D Plot
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        poly = PolyCollection(verts, facecolors=FILL_COLOR, edgecolors=LINE_COLOR, linewidths=1.0, alpha=1.0)
        ax.add_collection3d(poly, zs=range(len(channels_rev)), zdir='y')

        # Styling
        ax.set_xlim(t_crop[0], t_crop[-1])
        ax.set_ylim(-1, len(channels_rev))
        
        # Auto-scale Z height
        all_y_vals = [pt[1] for vert in verts for pt in vert]
        max_h = max(all_y_vals) if all_y_vals else 1.0
        ax.set_zlim(0, max_h * 1.1)
        
        ax.set_xlabel('Time (µs)', color=AXIS_COLOR, labelpad=15, fontsize=12)
        ax.set_ylabel('Channel', color=AXIS_COLOR, labelpad=20, fontsize=12)
        ax.set_zlabel('Current (arb)', color=AXIS_COLOR, labelpad=15, fontsize=12)
        
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(FILL_COLOR)
        ax.yaxis.pane.set_edgecolor(FILL_COLOR)
        ax.zaxis.pane.set_edgecolor(FILL_COLOR)
        
        ax.tick_params(axis='x', colors=AXIS_COLOR)
        ax.tick_params(axis='y', colors=AXIS_COLOR)
        ax.tick_params(axis='z', colors=AXIS_COLOR)
        
        ax.set_yticks(range(len(channels_rev)))
        ax.set_yticklabels(labels_rev, fontsize=7)

        title_str = f"Event {event_id}: {plane_name}\n(SG: {sg_win}ns, Gauss: {gauss_sig}ns)"
        
        # View 1: The Album Cover
        ax.set_title(title_str, color='white', fontsize=16, pad=20)
        ax.view_init(elev=30, azim=-85)
        pdf.savefig(fig, bbox_inches='tight', facecolor='black')
        
        # View 2: Isometric
        ax.set_title(title_str + " - Isometric", color='white', fontsize=16, pad=20)
        ax.view_init(elev=40, azim=-60)
        pdf.savefig(fig, bbox_inches='tight', facecolor='black')
        
        # View 3: Dramatic Angle
        ax.set_title(title_str + " - Angled", color='white', fontsize=16, pad=20)
        ax.view_init(elev=50, azim=-45)
        pdf.savefig(fig, bbox_inches='tight', facecolor='black')
        
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Joy Division 3D Plotter for Specific Events")
    parser.add_argument('filename', help="HDF5 File")
    parser.add_argument('--event', '-e', type=int, required=True, help="Target Event ID")
    parser.add_argument('--sg_x', type=float, required=True, help="Optimal SG Window for X-Anode (ns)")
    parser.add_argument('--gauss_x', type=float, required=True, help="Optimal Gaussian Sigma for X-Anode (ns)")
    parser.add_argument('--sg_y', type=float, required=True, help="Optimal SG Window for Y-Anode (ns)")
    parser.add_argument('--gauss_y', type=float, required=True, help="Optimal Gaussian Sigma for Y-Anode (ns)")
    parser.add_argument('--gain', type=float, default=20.0, help="Vertical amplitude multiplier (Default: 20)")
    args = parser.parse_args()
    
    out_dir = os.path.join(os.path.dirname(args.filename), "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"Joy3D_Event{args.event}.pdf")
    
    try:
        with h5py.File(args.filename, 'r') as f:
            with PdfPages(out_name) as pdf:
                render_joy_3d(pdf, f, args.event, args.sg_x, args.gauss_x, args.sg_y, args.gauss_y, args.gain)
                
        print(f"Saved Art Plot to: {out_name}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()