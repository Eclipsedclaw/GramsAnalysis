import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse
import io
import gc  
from multiprocessing import Pool
from PIL import Image 
import detector_config as dc
import time 

def render_single_event(payload):
    """
    Worker Unit: Generates plot based on config.
    """
    index, event_id, waveforms, baselines, res_ns, filename, run_config, trig_thresh = payload
    
    try:
        # 1. SETUP
        is_pedestal = "pedestal" in filename.lower() and "sig" not in filename.lower()
        draw_trigger = not is_pedestal
        
        time_axis = np.arange(waveforms.shape[1]) * (res_ns * 1e-3)
        ch_map = dc.get_channel_map(run_config)
        all_traces = waveforms - baselines[:, np.newaxis]
        
        # 2. LAYOUT
        has_charge = len(ch_map["Charge"]) > 0
        plot_groups = []
        
        if has_charge:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            csp_x = [c for c in ch_map["Charge"] if 24 <= c <= 27]
            csp_y = [c for c in ch_map["Charge"] if 28 <= c <= 31]
            plot_groups = [
                (axes[0,0], "CSP: X-Anodes", csp_x, "Set1"),
                (axes[0,1], "CSP: Y-Anodes", csp_y, "Set1"),
                (axes[1,0], "SiPM: VUV (Even)", ch_map["VUV"], "tab20"),
                (axes[1,1], "SiPM: VIS (Odd)",  ch_map["VIS"], "tab20")
            ]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
            plt.subplots_adjust(wspace=0.1)
            plot_groups = [
                (axes[0], "SiPM: VUV (Even)", ch_map["VUV"], "tab20"),
                (axes[1], "SiPM: VIS (Odd)",  ch_map["VIS"], "tab20")
            ]

        # 3. LIMITS
        lims_light = [-10, 10]
        lims_chg   = [-10, 10]

        if not is_pedestal:
            light_chs = ch_map["VUV"] + ch_map["VIS"]
            if light_chs:
                l_min, l_max = np.min(all_traces[light_chs]), np.max(all_traces[light_chs])
                span = max(l_max - l_min, 1.0)
                lims_light = [l_min - 0.15*span, l_max + 0.15*span]
                if draw_trigger: lims_light[0] = min(lims_light[0], trig_thresh - 5)
            
            if has_charge:
                c_min, c_max = np.min(all_traces[ch_map["Charge"]]), np.max(all_traces[ch_map["Charge"]])
                span = max(c_max - c_min, 1.0)
                lims_chg = [c_min - 0.15*span, c_max + 0.15*span]

        # 4. RENDER
        for ax, title, channels, cmap_name in plot_groups:
            if len(channels) > 0:
                cmap = matplotlib.colormaps[cmap_name]
                colors = cmap(np.linspace(0, 1, len(channels))) if cmap_name == "tab20" else cmap(np.linspace(0, 1, 9))
            
            ax.set_ylim(lims_chg if "CSP" in title else lims_light)

            for i, ch in enumerate(channels):
                if ch >= waveforms.shape[0]: continue
                c = colors[i % len(colors)]
                ax.plot(time_axis, all_traces[ch], color=c, lw=1.0, alpha=0.7, 
                        label=f"Ch{ch}", rasterized=True)
            
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.axhline(0, c='k', ls='-', lw=0.5, alpha=0.5)
            ax.grid(True, alpha=0.2)
            
            if draw_trigger and "SiPM" in title:
                ax.axhline(trig_thresh, color='red', linestyle='--', linewidth=1.5, label=f"Trig {trig_thresh}mV")

            if channels:
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize='xx-small', ncol=1)
                
            ax.set_xlabel("Time (µs)")

        fig.suptitle(f"Event: {index} \nFile: {filename}", fontsize=13, y=0.98)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        print(f"Plot Error {event_id}: {e}")
        return None

def process_acquisition(task):
    acq_dir, h5_files, requested_events, trig_thresh, force_mode = task
    acq_name = os.path.basename(acq_dir)
    h5_files.sort()
    
    print(f"--> Scanning Acquisition: {acq_name} ({len(h5_files)} files)")

    save_dir = os.path.join(acq_dir, "Plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # SUFFIX LOGIC
    first_file = os.path.basename(h5_files[0])
    fname_lower = first_file.lower()
    if "combo" in fname_lower:
        suffix = "_Combo_Pedestal_QC.pdf" if "pedestal" in fname_lower else "_Combo_QC.pdf"
    elif "pedestal" in fname_lower:
        suffix = "_Pedestal_QC.pdf"
    else:
        suffix = "_Scint_QC.pdf"
        
    pdf_path = os.path.join(save_dir, acq_name + suffix)

    # LIMIT LOGIC
    MAX_SAFE_LIMIT = 3000
    if requested_events == 'all':
        limit = float('inf')
    else:
        limit = int(requested_events)

    events_plotted = 0
    start_time = time.time()
    
    try:
        with PdfPages(pdf_path) as pdf:
            for f_path in h5_files:
                if events_plotted >= limit: break
                    
                filename = os.path.basename(f_path)
                try:
                    with h5py.File(f_path, 'r') as f:
                        if 'baseline_mean_mV' not in f:
                            print(f"    [SKIP] {filename}: No baselines.")
                            continue
                        
                        n_in_file = f['event_ids'].shape[0]
                        needed = limit - events_plotted
                        to_grab = min(n_in_file, needed) if limit != float('inf') else n_in_file
                        
                        # --- SAFETY CHECK: Force Mode ---
                        if (events_plotted + to_grab) > MAX_SAFE_LIMIT and not force_mode:
                            print(f"    [WARN] Event count ({events_plotted + to_grab}) exceeds safety limit ({MAX_SAFE_LIMIT}).")
                            print(f"           Capping at {MAX_SAFE_LIMIT}. Use --force to override.")
                            to_grab = max(0, MAX_SAFE_LIMIT - events_plotted)
                            limit = MAX_SAFE_LIMIT 
                            if to_grab == 0: break

                        # --- CHUNKED PLOTTING LOOP ---
                        # Process in small batches (e.g., 50 events) to prevent RAM runaway
                        CHUNK_SIZE = 50 
                        
                        # If to_grab is huge, this loop handles it safely with chunking procedure
                        for start_idx in range(0, to_grab, CHUNK_SIZE):
                            end_idx = min(start_idx + CHUNK_SIZE, to_grab)
                            current_batch_size = end_idx - start_idx
                            
                            # Load ONLY this chunk into RAM
                            waveforms = f['waveforms_mV'][start_idx:end_idx]
                            baselines = f['baseline_mean_mV'][start_idx:end_idx]
                            ids = f['event_ids'][start_idx:end_idx]
                            res_ns = f.attrs.get('resolution_ns', 2)
                            run_config = f.attrs.get('run_config', 'light_only') 

                            # Render Batch
                            payloads = []
                            for k in range(current_batch_size):
                                # Global index for the title
                                global_idx = events_plotted + k 
                                payloads.append((global_idx, ids[k], waveforms[k], baselines[k], res_ns, filename, run_config, trig_thresh))
                            
                            for p in payloads:
                                img_bytes = render_single_event(p)
                                if img_bytes:
                                    img = Image.open(io.BytesIO(img_bytes))
                                    fig_temp = plt.figure(figsize=(11.69, 8.27))
                                    ax_temp = plt.Axes(fig_temp, [0., 0., 1., 1.])
                                    ax_temp.set_axis_off()
                                    fig_temp.add_axes(ax_temp)
                                    ax_temp.imshow(img)
                                    pdf.savefig(fig_temp)
                                    plt.close(fig_temp)
                            
                            # Update Counter and Cleanup
                            events_plotted += current_batch_size
                            
                            # Explicitly clear batch memory
                            del waveforms, baselines, ids, payloads
                            gc.collect()

                        print(f"    [+ADDED] {to_grab} events from {filename}")

                except Exception as e:
                    print(f"    [ERR] Reading {filename}: {e}")
                    continue
            
            total_time = time.time() - start_time
            rate = total_time / events_plotted if events_plotted > 0 else 0
            
            print(f"    [DONE] Saved: {pdf_path}")
            print(f"    [STATS] Total Events: {events_plotted} | Time: {total_time:.2f}s ({rate:.4f} s/evt)")

    except Exception as e:
        print(f"    [FAIL] Generating PDF for {acq_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="LArTPC Mass Plotter (Chunked & Safe)")
    parser.add_argument('input_target', help="Input file or directory")
    parser.add_argument('--threshold', '-t', type=float, required=True, help="Trigger Threshold (mV)")
    parser.add_argument('--events', '-e', default='100', help="Events to plot: '100' or 'all'")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Max 8 workers")
    parser.add_argument('--force', action='store_true', help="Override 3000 event limit")
    args = parser.parse_args()
    
    active_workers = min(args.workers, 8)

    print(f"=== MASS PLOTTER ===")
    print(f"Target:    {args.input_target}")
    print(f"Events:    {args.events}")
    print(f"Force:     {args.force}")
    print("-" * 30)
    
    acq_groups = {}
    
    if os.path.isfile(args.input_target):
        d = os.path.dirname(args.input_target)
        acq_groups[d] = [args.input_target]
    else:
        for root, dirs, files in os.walk(args.input_target):
            h5s = [os.path.join(root, f) for f in files if f.endswith('.h5')]
            if h5s:
                acq_groups[root] = h5s
    
    tasks = []
    for acq_dir, file_list in acq_groups.items():
        tasks.append((acq_dir, file_list, args.events, args.threshold, args.force))
        
    print(f"Found {len(tasks)} Acquisitions to process.")
    
    if len(tasks) > 0:
        with Pool(active_workers) as p:
            p.map(process_acquisition, tasks)
    else:
        print("No acquisitions found.")

if __name__ == "__main__":
    main()