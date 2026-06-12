"""
29_VIS_VUV_Correlation.py
Per-Event VIS vs VUV Super-Pulse Correlation (2D Density)

For each event:
  1. Sum all 16 VIS channel waveforms -> VIS super-pulse
  2. Sum all 16 VUV channel waveforms -> VUV super-pulse
  3. Riemann-sum each over the full trace (mV * ns), sign-flipped positive
  4. Plot (VUV, VIS) 2D density as a 1x6 panel row of time bins

Uses BASELINE-CORRECTED waveforms (waveform - baseline_mean per channel).
Randomly samples files from each bin until the user-defined event target
is reached; opens the last file partially if needed.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import glob
import re
import time
from multiprocessing import Pool

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm


CAEN_CLOCK_HZ = 125_000_000.0
N_CAEN_CH = 32

# VUV = even CAEN channels, VIS = odd CAEN channels (Board 72 convention)
VUV_CHS = [c for c in range(N_CAEN_CH) if c % 2 == 0]
VIS_CHS = [c for c in range(N_CAEN_CH) if c % 2 == 1]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def read_file_timestamps(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            ts = f['Board_72']['timestamps']
            if ts.shape[0] == 0:
                return None
            return (int(ts[0]), int(ts[-1]))
    except Exception:
        return None


def process_file(task):
    """
    Return (vuv_integrals, vis_integrals) as two 1D arrays.
    If max_events is set, return at most that many events from the file.
    Both arrays sign-flipped positive (pulses are negative-going).
    """
    filepath, max_events = task
    try:
        with h5py.File(filepath, 'r') as f:
            grp = f['Board_72']
            dt_ns = float(grp.attrs.get('sampling_period_ns', 2.0))
            n_ev_file = grp['waveforms'].shape[0]
            n_read = n_ev_file if max_events is None else min(max_events, n_ev_file)
            if n_read == 0:
                return np.array([]), np.array([])
            waveforms = grp['waveforms'][:n_read]     # (n_read, 32, n_samp)
            if 'baseline_mean' not in grp:
                tqdm.write(f"  [skip] {os.path.basename(filepath)}: "
                           f"no baseline_mean — run 02_calc_baselines first")
                return np.array([]), np.array([])
            baselines = grp['baseline_mean'][:n_read] # (n_read, 32)
    except Exception as e:
        tqdm.write(f"  [skip] {os.path.basename(filepath)}: {e}")
        return np.array([]), np.array([])

    # Baseline-subtract: broadcast (n_read, 32, 1) against (n_read, 32, n_samp)
    corrected = waveforms - baselines[:, :, None]

    # Drop events with any NaN in VUV or VIS channels (dropped-event padding)
    vuv_stack = corrected[:, VUV_CHS, :]              # (n_read, 16, n_samp)
    vis_stack = corrected[:, VIS_CHS, :]
    bad = np.any(np.isnan(vuv_stack), axis=(1, 2)) | \
          np.any(np.isnan(vis_stack), axis=(1, 2))
    good = ~bad

    # Super-pulse: sum across channels -> (n_good, n_samp)
    vuv_super = vuv_stack[good].sum(axis=1)
    vis_super = vis_stack[good].sum(axis=1)

    # Riemann sum (mV * ns), sign-flipped positive since pulses are negative
    vuv_int = -vuv_super.sum(axis=1) * dt_ns
    vis_int = -vis_super.sum(axis=1) * dt_ns

    return vuv_int.astype(np.float32), vis_int.astype(np.float32)


def assign_files_to_bins(files, num_bins):
    print(f"Scanning headers of {len(files)} files...")
    headers = []
    for fp in tqdm(files, desc="Headers", unit="file"):
        h = read_file_timestamps(fp)
        if h is not None:
            headers.append((fp, h[0], h[1]))
    if not headers:
        return None, None

    t0 = min(h[1] for h in headers)
    t_final = max(h[2] for h in headers)
    total_sec = (t_final - t0) / CAEN_CLOCK_HZ
    bin_width = total_sec / num_bins

    bin_files = [[] for _ in range(num_bins)]
    for fp, first_ts, last_ts in headers:
        mid_sec = ((first_ts + last_ts) / 2 - t0) / CAEN_CLOCK_HZ
        b = int(min(num_bins - 1, max(0, mid_sec // bin_width)))
        bin_files[b].append(fp)
    return bin_files, total_sec


def peek_events_per_file(filepath):
    """Return the event count for a single file (cheap — just reads shape)."""
    try:
        with h5py.File(filepath, 'r') as f:
            return int(f['Board_72']['waveforms'].shape[0])
    except Exception:
        return 0


def sample_evenly(items, n):
    if n >= len(items) or n <= 0:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def plot_correlation(bin_data, bin_edges_sec, tag, out_path,
                     nbins_2d=80, log_z=True):
    """
    bin_data: list of (vuv_int_array, vis_int_array) per time bin
    Renders a 1xN row of panels (one per time bin) for side-by-side
    topology comparison.
    """
    num_bins = len(bin_data)

    # Common axis range: 0.5 / 99.5 percentiles across all bins combined
    all_vuv = np.concatenate([d[0] for d in bin_data if len(d[0])])
    all_vis = np.concatenate([d[1] for d in bin_data if len(d[1])])
    if len(all_vuv) == 0:
        print("[WARN] No events to plot")
        return
    vuv_lo, vuv_hi = np.percentile(all_vuv, [0.5, 99.5])
    vis_lo, vis_hi = np.percentile(all_vis, [0.5, 99.5])
    vuv_hi *= 1.05
    vis_hi *= 1.05

    # 1 x N row layout
    fig, axes = plt.subplots(1, num_bins,
                             figsize=(num_bins * 3.6, 4.2),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]  # unwrap the 1-row array

    norm = LogNorm(vmin=1) if log_z else None
    last_im = None
    for b in range(num_bins):
        ax = axes[b]
        vuv, vis = bin_data[b]
        if len(vuv) == 0:
            ax.text(0.5, 0.5, "No events", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f"Bin {b}\n{bin_edges_sec[b]/60:.0f}–"
                         f"{bin_edges_sec[b+1]/60:.0f} min", fontsize=10)
            continue

        h, xe, ye = np.histogram2d(vuv, vis, bins=nbins_2d,
                                   range=[[vuv_lo, vuv_hi], [vis_lo, vis_hi]])
        im = ax.imshow(h.T, origin='lower', aspect='auto',
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap='viridis', norm=norm)
        last_im = im
        ax.set_xlabel("VUV  ∫V dt  (mV·ns)", fontsize=9)
        if b == 0:
            ax.set_ylabel("VIS  ∫V dt  (mV·ns)", fontsize=10)
        ax.set_title(f"Bin {b}\n{bin_edges_sec[b]/60:.0f}–"
                     f"{bin_edges_sec[b+1]/60:.0f} min\nN={len(vuv)}",
                     fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=8)

    if last_im is not None:
        fig.subplots_adjust(right=0.93, wspace=0.15, top=0.82, bottom=0.15)
        cax = fig.add_axes([0.94, 0.15, 0.010, 0.67])
        fig.colorbar(last_im, cax=cax).set_label('Events / 2D bin', fontsize=9)

    fig.suptitle(f"VIS vs VUV Super-Pulse Correlation — {tag}",
                 fontsize=13, y=0.99)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir', help="Directory of HDF5 files")
    p.add_argument('--num_bins', '-n', type=int, default=6)
    p.add_argument('--events_per_bin', '-e', type=int, default=5000,
                   help="Target events per bin (default 5000)")
    p.add_argument('--workers', '-w', type=int, default=4)
    p.add_argument('--nbins_2d', type=int, default=80,
                   help="2D histogram resolution per axis")
    p.add_argument('--seed', type=int, default=42,
                   help="Random seed for reproducible file selection")
    p.add_argument('--out_dir', '-o', type=str, default=None)
    p.add_argument('--tag', type=str, default=None)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")), key=natural_key)
    files = [f for f in files if "SPE_Master_Integrals" not in os.path.basename(f)]
    if not files:
        print(f"[FAIL] No HDF5 files in {args.input_dir}")
        return

    tag = args.tag or os.path.basename(os.path.normpath(args.input_dir))
    out_dir = args.out_dir or os.path.join(args.input_dir, "Plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  VIS vs VUV CORRELATION — {tag}")
    print(f"{'='*60}")
    print(f"Files discovered : {len(files)}")
    print(f"Bins / target    : {args.num_bins} bins, "
          f"{args.events_per_bin} events/bin")
    print(f"Workers          : {args.workers}")
    print(f"Seed             : {args.seed}\n")

    bin_files, total_sec = assign_files_to_bins(files, args.num_bins)
    if bin_files is None:
        print("[FAIL] Could not read any timestamps")
        return
    bin_edges = np.linspace(0, total_sec, args.num_bins + 1)

    print(f"\nRun duration     : {total_sec/60:.1f} min")
    for b in range(args.num_bins):
        print(f"  Bin {b}: {len(bin_files[b])} files in window")

    # Probe one file to learn typical events-per-file
    events_per_file = 500  # fallback
    for fp in files:
        n = peek_events_per_file(fp)
        if n > 0:
            events_per_file = n
            break
    print(f"\nEvents per file  : ~{events_per_file}")

    # For each bin: shuffle files, pick enough to meet target, truncate last
    rng = np.random.default_rng(args.seed)
    tasks = []        # list of (bin_idx, filepath, max_events)
    for b in range(args.num_bins):
        bin_fps = list(bin_files[b])
        if not bin_fps:
            print(f"  [WARN] Bin {b} has no files!")
            continue
        rng.shuffle(bin_fps)

        # Estimate how many full files we need, + one partial
        remaining = args.events_per_bin
        for fp in bin_fps:
            if remaining <= 0:
                break
            take = min(remaining, events_per_file)
            # If we need at least a full file's worth, don't pass a cap
            # (avoids accidentally capping a slightly-undersized file)
            cap = None if take >= events_per_file else take
            tasks.append((b, fp, cap))
            remaining -= events_per_file

    print(f"\nQueued {len(tasks)} file-reads to hit "
          f"~{args.events_per_bin * args.num_bins} total events\n")

    worker_tasks = [(fp, cap) for (_, fp, cap) in tasks]
    bin_data = [([], []) for _ in range(args.num_bins)]

    t_start = time.time()
    with Pool(args.workers) as pool:
        for i, (vuv_int, vis_int) in enumerate(
                tqdm(pool.imap(process_file, worker_tasks),
                     total=len(worker_tasks), desc="Files", unit="file")):
            b = tasks[i][0]
            if len(vuv_int) > 0:
                bin_data[b][0].append(vuv_int)
                bin_data[b][1].append(vis_int)
    elapsed = time.time() - t_start

    # Concatenate per-bin and trim to target event count (defensive)
    bin_arrays = []
    for b in range(args.num_bins):
        if bin_data[b][0]:
            vuv = np.concatenate(bin_data[b][0])
            vis = np.concatenate(bin_data[b][1])
            if len(vuv) > args.events_per_bin:
                vuv = vuv[:args.events_per_bin]
                vis = vis[:args.events_per_bin]
        else:
            vuv = np.array([])
            vis = np.array([])
        bin_arrays.append((vuv, vis))

    plot_path = os.path.join(out_dir,
        f"{tag}_VIS_VUV_Correlation_{args.num_bins}bin.pdf")
    plot_correlation(bin_arrays, bin_edges, tag, plot_path,
                     nbins_2d=args.nbins_2d)

    # Summary
    print(f"\n{'='*82}")
    print(f"{'Bin':<4}{'Window (min)':<16}{'Events':>9}"
          f"{'VUV med':>12}{'VIS med':>12}{'VIS/VUV med':>14}"
          f"{'VIS>>VUV frac':>15}")
    print('-' * 82)
    for b in range(args.num_bins):
        vuv, vis = bin_arrays[b]
        if len(vuv) == 0:
            continue
        med_vuv = np.median(vuv)
        med_vis = np.median(vis)
        ratio = med_vis / med_vuv if med_vuv != 0 else float('inf')
        # Anomaly indicator: fraction of events where VIS dominates VUV by 2x
        # (nominal events have VUV > VIS almost always)
        if med_vuv > 0:
            frac_vis_dom = float(np.mean(vis > 2 * vuv))
        else:
            frac_vis_dom = 0.0
        print(f"{b:<4}{bin_edges[b]/60:>5.1f}–{bin_edges[b+1]/60:<10.1f}"
              f"{len(vuv):>9d}{med_vuv:>12.1f}{med_vis:>12.1f}{ratio:>14.3f}"
              f"{frac_vis_dom*100:>13.1f}%")
    print('=' * 82)
    print(f"\nProcessed in {elapsed:.1f} s")
    print(f"Plot: {plot_path}\n")


if __name__ == "__main__":
    main()