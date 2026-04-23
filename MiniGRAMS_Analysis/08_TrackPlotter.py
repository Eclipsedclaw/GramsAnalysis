import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import detector_config as dc

# --- THEME: DARK MODE (JOY DIVISION) ---
plt.style.use('dark_background')
LINE_COLOR = 'white'
FILL_COLOR = 'black' # Matches background to create occlusion
AXIS_COLOR = 'white'

# --- PHYSICS TUNING ---
#TARGET_SG_WINDOW_NS = 3000.0 
TARGET_SG_WINDOW_NS = 1510.0  
SG_POLY_ORDER = 2             
#POST_DERIV_SIGMA_NS = 2000.0 
POST_DERIV_SIGMA_NS = 2882.0  

def process_waveform_joy(trace, res_ns):
    win_samples = int(TARGET_SG_WINDOW_NS / res_ns)
    if win_samples % 2 == 0: win_samples += 1 
    if win_samples < 3: win_samples = 3
    
    smoothed = savgol_filter(trace, window_length=win_samples, polyorder=SG_POLY_ORDER)
    deriv = np.gradient(smoothed)
    # Heavy smoothing for "Rolling Hills" look
    joy_trace = gaussian_filter1d(deriv, sigma=POST_DERIV_SIGMA_NS / res_ns)
    
    return joy_trace

def polygon_under_graph(x, y):
    """Constructs vertices for the filled ribbon."""
    # Ensure y starts and ends at 0 for a closed polygon
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def render_joy_3d(pdf, event_id, waves, baselines, ch_map, res_ns, gain):
    print(f"Rendering Joy Division 3D for Event {event_id}...")
    
    t_axis = np.arange(waves.shape[1]) * (res_ns / 1000.0)
    
    # --- CROP TO PHYSICS WINDOW (0 - 80us) ---
    # This addresses your concern about the 300us window being too empty.
    mask = (t_axis >= 0) & (t_axis <= 80.0)
    t_crop = t_axis[mask]
    
    planes = [('X-Anode', ch_map['Charge_X']), ('Y-Anode', ch_map['Charge_Y'])]
    
    for plane_name, channels in planes:
        if not channels: continue
        
        # 1. Process Data
        verts = []
        # Reverse order so Channel 0 is "closest" to camera
        for ch in reversed(channels):
            raw = waves[ch] - baselines[ch]
            # Crop raw data first
            raw_crop = raw[mask]
            
            proc = process_waveform_joy(raw_crop, res_ns)
            
            # Apply Gain
            proc = proc * gain
            
            verts.append(polygon_under_graph(t_crop, proc))

        # 2. Setup 3D Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3. Create Ribbons (The "Unknown Pleasures" Effect)
        # facecolor=FILL_COLOR (Black) hides the lines behind it!
        # edgecolor=LINE_COLOR (White) draws the ridge.
        poly = PolyCollection(verts, 
                              facecolors=FILL_COLOR, 
                              edgecolors=LINE_COLOR, 
                              linewidths=1.2, 
                              alpha=1.0) # Opaque is key for occlusion
        
        ax.add_collection3d(poly, zs=range(len(channels)), zdir='y')

        # 4. Styling
        ax.set_xlim(t_crop[0], t_crop[-1])
        ax.set_ylim(-1, len(channels))
        ax.set_zlim(0, 5) # Arbitrary height limit, adjust based on gain
        
        # Labels
        ax.set_xlabel('Time (µs)', color=AXIS_COLOR, labelpad=10)
        ax.set_ylabel('Channel', color=AXIS_COLOR, labelpad=10)
        ax.set_zlabel('Current', color=AXIS_COLOR, labelpad=10)
        
        # Remove the "Box" and Grid for that floating look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(FILL_COLOR)
        ax.yaxis.pane.set_edgecolor(FILL_COLOR)
        ax.zaxis.pane.set_edgecolor(FILL_COLOR)
        
        # Ticks customization
        ax.tick_params(axis='x', colors=AXIS_COLOR)
        ax.tick_params(axis='y', colors=AXIS_COLOR)
        ax.tick_params(axis='z', colors=AXIS_COLOR)
        
        # Channel Ticks
        # ax.set_yticks(range(len(channels)))
        # ax.set_yticklabels([str(ch) for ch in reversed(channels)], fontsize=8)

        title_str = f"Event {event_id}: {plane_name} (Joy Division 3D)"
        ax.set_title(title_str, color='white', fontsize=16)

        # --- VIEW 1: THE ALBUM COVER (Front/Slight Angle) ---
        # Elev=20, Azim=-90 looks almost straight on but stacked
        ax.view_init(elev=25, azim=-85)
        pdf.savefig(fig, bbox_inches='tight', facecolor='black')
        
        # --- VIEW 2: ISOMETRIC (Diagnostic) ---
        ax.view_init(elev=35, azim=-60)
        ax.set_title(title_str + " - Iso View", color='white')
        pdf.savefig(fig, bbox_inches='tight', facecolor='black')
        
        # --- VIEW 3: DRAMATIC ANGLE ---
        ax.view_init(elev=45, azim=-45)
        ax.set_title(title_str + " - Angled", color='white')
        pdf.savefig(fig, bbox_inches='tight', facecolor='black')
        
        plt.close(fig)

def run_visualizer(filename, event_id, gain):
    out_name = f"Joy3D_Event{event_id}.pdf"
    
    try:
        with h5py.File(filename, 'r') as f:
            run_config = f.attrs.get('run_config', 'UPS')
            res_ns = f.attrs.get('resolution_ns', 8.0)
            ch_map = dc.get_channel_map(run_config)
            
            waves = f['waveforms_mV'][event_id]
            if 'baseline_mean_mV' in f:
                baselines = f['baseline_mean_mV'][event_id]
            else:
                baselines = np.mean(waves[:, :20], axis=1)
                
            with PdfPages(out_name) as pdf:
                render_joy_3d(pdf, event_id, waves, baselines, ch_map, res_ns, gain)
                
        print(f"Saved Art Plot to: {out_name}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--event', '-e', type=int, required=True)
    parser.add_argument('--gain', '-g', type=float, default=2.0, help="Height Multiplier")
    args = parser.parse_args()
    
    run_visualizer(args.filename, args.event, args.gain)