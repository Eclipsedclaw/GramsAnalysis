"""
28_TriggerRate_PhysicalHeatmap.py
Per-Channel, Per-TimeBin Trigger Rate Heatmap (Physical Layout)

Samples N files per time bin (default 10) rather than processing the entire
acquisition. For on/off anomalies this gives the same physics answer in a
small fraction of the runtime.

Uses RAW (non-baseline-corrected) waveforms for first-crossing attribution:
baseline correction can promote DC-offset channels into false "crossings".
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

# Physical map: CAEN ch -> (MB, light_ch_1indexed, 'VUV'|'VIS')
# MB1=CAEN 0-11, MB2=CAEN 12-23, MB3=CAEN 24-31 (only 8 chans; 9-12 go to slaves)
# Odd light_ch = VUV, even = VIS (matches detector_config.py)
def physical_info(caen_ch):
    if caen_ch < 12:   mb, lc = 1, caen_ch + 1
    elif caen_ch < 24: mb, lc = 2, caen_ch - 11
    else:              mb, lc = 3, caen_ch - 23
    return mb, lc, 'VUV' if lc % 2 == 1 else 'VIS'

def find_caen(mb, light_ch):
    for c in range(N_CAEN_CH):
        m, l, _ = physical_info(c)
        if m == mb and l == light_ch:
            return c
    return None


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def read_file_timestamps(filepath):
    """Return (first_ts, last_ts) for a file. Cheap — reads only timestamps dataset."""
    try:
        with h5py.File(filepath, 'r') as f:
            ts = f['Board_72']['timestamps']
            if ts.shape[0] == 0:
                return None
            return (int(ts[0]), int(ts[-1]))
    except Exception:
        return None


def process_file(task):
    """Read one file's waveforms, return (counts_per_ch, n_events_used, duration_sec)."""
    filepath, threshold_mv, search_start_us, search_end_us = task
    try:
        with h5py.File(filepath, 'r') as f:
            grp = f['Board_72']
            dt_ns = float(grp.attrs.get('sampling_period_ns', 2.0))
            waveforms = grp['waveforms'][:]       # raw, (n_ev, 32, n_samp)
            timestamps = grp['timestamps'][:]
        n_events = waveforms.shape[0]
        if n_events == 0:
            return np.zeros(N_CAEN_CH, dtype=np.int64), 0, 0.0

        idx_s = int(search_start_us * 1000.0 / dt_ns)
        idx_e = int(search_end_us * 1000.0 / dt_ns)
        chunk = waveforms[:, :, idx_s:idx_e]       # (n_ev, 32, w)

        valid = ~np.any(np.isnan(chunk), axis=(1, 2))

        sentinel = np.iinfo(np.int32).max
        crossed = chunk < threshold_mv
        any_cross = crossed.any(axis=2)
        first_idx = np.where(any_cross, np.argmax(crossed, axis=2), sentinel)
        winners = first_idx.argmin(axis=1)
        event_crossed = first_idx.min(axis=1) < sentinel

        mask = valid & event_crossed
        counts = np.bincount(winners[mask], minlength=N_CAEN_CH).astype(np.int64)
        dur_sec = (int(timestamps[-1]) - int(timestamps[0])) / CAEN_CLOCK_HZ
        return counts, int(mask.sum()), dur_sec

    except Exception as e:
        tqdm.write(f"  [skip] {os.path.basename(filepath)}: {e}")
        return np.zeros(N_CAEN_CH, dtype=np.int64), 0, 0.0


def assign_files_to_bins(files, num_bins):
    """Bucket each file into a time bin based on midpoint of its timestamp range."""
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


def sample_evenly(items, n):
    if n >= len(items) or n <= 0:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def plot_heatmap(rates_hz, bin_edges_sec, acq_tag, threshold_mv,
                 search_window, duration_sec, out_path):
    """Draw MB3/MB1/MB2 physical-layout heatmaps, one per bin."""
    num_bins = rates_hz.shape[0]
    nrows, ncols = (2, 3) if num_bins == 6 else (1, num_bins)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 5.8),
                             squeeze=False)

    pos = rates_hz[rates_hz > 0]
    vmin = max(pos.min() * 0.8 if len(pos) else 1e-3, 1e-3)
    vmax = rates_hz.max() * 1.2 if len(pos) else 1.0

    cmap = plt.cm.viridis
    cmap.set_bad(color='#E8E8E8')
    mb_order = [3, 1, 2]     # top -> bottom

    last_im = None
    for b in range(num_bins):
        ax = axes[b // ncols, b % ncols]
        grid = np.full((6, 6), np.nan)
        labels = [['' for _ in range(6)] for _ in range(6)]
        for mb_row, mb in enumerate(mb_order):
            for col in range(6):
                for is_vis in (False, True):
                    r = 2 * mb_row + (1 if is_vis else 0)
                    lc = 2 * col + (2 if is_vis else 1)
                    c = find_caen(mb, lc)
                    if c is None:
                        labels[r][col] = f"MB{mb}-CH{lc}\nN/C"
                    else:
                        grid[r, col] = rates_hz[b, c]
                        labels[r][col] = f"MB{mb}-CH{lc}\n{rates_hz[b, c]:.2f} Hz"

        disp = np.where(grid > 0, grid, vmin)
        disp = np.ma.masked_where(np.isnan(grid), disp)
        im = ax.imshow(disp, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                       aspect='equal', origin='upper')
        last_im = im

        for r in range(6):
            for c in range(6):
                if np.isnan(grid[r, c]):
                    ax.text(c, r, labels[r][c], ha='center', va='center',
                            fontsize=7.5, color='#888')
                else:
                    val = max(grid[r, c], vmin)
                    frac = (np.log10(val) - np.log10(vmin)) / \
                           (np.log10(vmax) - np.log10(vmin))
                    color = 'black' if frac > 0.55 else 'white'
                    ax.text(c, r, labels[r][c], ha='center', va='center',
                            fontsize=7.5, color=color, fontweight='bold')

        for y in (1.5, 3.5): ax.axhline(y, color='black', linewidth=2.5)
        for x in (1.5, 3.5): ax.axvline(x, color='black', linewidth=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Bin {b}: {bin_edges_sec[b]/60:.1f}–"
                     f"{bin_edges_sec[b+1]/60:.1f} min", fontsize=11)

    for b in range(num_bins, nrows * ncols):
        axes[b // ncols, b % ncols].axis('off')

    if last_im is not None:
        fig.subplots_adjust(right=0.91)
        cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
        fig.colorbar(last_im, cax=cax).set_label('Trigger Rate (Hz)', fontsize=11)

    fig.suptitle(f"Per-Channel Trigger Rate — {acq_tag}\n"
                 f"Threshold: {threshold_mv:.1f} mV (raw) | Search: {search_window} "
                 f"| Run duration: {duration_sec/60:.1f} min",
                 fontsize=13, y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir', help="Directory of HDF5 files")
    p.add_argument('--num_bins', '-n', type=int, default=6)
    p.add_argument('--files_per_bin', '-f', type=int, default=10,
                   help="Files to sample per bin (default 10)")
    p.add_argument('--threshold_mv', '-t', type=float, default=-60.0)
    p.add_argument('--search_start_us', '-s', type=float, default=3.8)
    p.add_argument('--search_end_us', '-e', type=float, default=4.5)
    p.add_argument('--workers', '-w', type=int, default=4)
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
    print(f"  TRIGGER-RATE PHYSICAL HEATMAP — {tag}")
    print(f"{'='*60}")
    print(f"Files discovered : {len(files)}")
    print(f"Bins / files     : {args.num_bins} / {args.files_per_bin} per bin")
    print(f"Threshold        : {args.threshold_mv} mV (raw)")
    print(f"Search window    : {args.search_start_us}–{args.search_end_us} µs")
    print(f"Workers          : {args.workers}\n")

    bin_files, total_sec = assign_files_to_bins(files, args.num_bins)
    if bin_files is None:
        print("[FAIL] Could not read any timestamps")
        return
    bin_edges = np.linspace(0, total_sec, args.num_bins + 1)

    print(f"\nRun duration     : {total_sec/60:.1f} min ({total_sec:.0f} s)")
    print(f"Bin width        : {total_sec/args.num_bins/60:.1f} min")
    for b in range(args.num_bins):
        print(f"  Bin {b}: {len(bin_files[b])} files in window")

    # Evenly subsample each bin
    sampled = []
    for b in range(args.num_bins):
        for fp in sample_evenly(bin_files[b], args.files_per_bin):
            sampled.append((b, fp))
    print(f"\nSampling {len(sampled)} files (~{len(sampled) * 500} events)\n")

    tasks = [(fp, args.threshold_mv, args.search_start_us, args.search_end_us)
             for (_, fp) in sampled]

    counts = np.zeros((args.num_bins, N_CAEN_CH), dtype=np.int64)
    sampled_sec = np.zeros(args.num_bins)
    sampled_ev  = np.zeros(args.num_bins, dtype=int)

    t_start = time.time()
    with Pool(args.workers) as pool:
        # imap preserves order so we can align results back to bin indices
        for i, (ch_counts, n_used, dur_sec) in enumerate(
                tqdm(pool.imap(process_file, tasks),
                     total=len(tasks), desc="Files", unit="file")):
            b = sampled[i][0]
            counts[b] += ch_counts
            sampled_sec[b] += dur_sec
            sampled_ev[b] += n_used
    elapsed = time.time() - t_start

    # Rate = counts within a bin / actual seconds of data sampled in that bin
    rates_hz = np.zeros_like(counts, dtype=np.float64)
    for b in range(args.num_bins):
        if sampled_sec[b] > 0:
            rates_hz[b] = counts[b] / sampled_sec[b]

    plot_path = os.path.join(out_dir,
        f"{tag}_TriggerRate_PhysicalHeatmap_{args.num_bins}bin.pdf")
    plot_heatmap(rates_hz, bin_edges, tag, args.threshold_mv,
                 f"{args.search_start_us}–{args.search_end_us} µs",
                 total_sec, plot_path)

    vuv = [c for c in range(N_CAEN_CH) if physical_info(c)[2] == 'VUV']
    vis = [c for c in range(N_CAEN_CH) if physical_info(c)[2] == 'VIS']
    print(f"\n{'='*76}")
    print(f"{'Bin':<4}{'Window (min)':<16}{'Events':>9}{'Samp s':>9}"
          f"{'VUV Hz':>10}{'VIS Hz':>10}{'VIS/VUV':>10}")
    print('-' * 76)
    for b in range(args.num_bins):
        v_vuv = rates_hz[b, vuv].mean()
        v_vis = rates_hz[b, vis].mean()
        ratio = v_vis / v_vuv if v_vuv > 0 else float('inf')
        print(f"{b:<4}{bin_edges[b]/60:>5.1f}–{bin_edges[b+1]/60:<10.1f}"
              f"{sampled_ev[b]:>9d}{sampled_sec[b]:>9.1f}"
              f"{v_vuv:>10.3f}{v_vis:>10.3f}{ratio:>10.2f}")
    print('=' * 76)
    print(f"\nProcessed in {elapsed:.1f} s")
    print(f"Plot: {plot_path}\n")


if __name__ == "__main__":
    main()