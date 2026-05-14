#!/usr/bin/env python3
"""
plot_audit.py
=============
Standalone audit visualiser for ephys alignment.
Loads a feature cache (.npz) + behavior CSV and produces one figure per feature
showing:
  - The full feature trace (downsampled for display)
  - All reward times as vertical red lines
  - All candidate peaks (from the audit) as blue dots
  - The largest ISI gap highlighted in yellow — this is your best landmark

Usage (PowerShell):
    python plot_audit.py `
      --cache "D:\...\ephys_alignment_out\15082023_Milka_StrCer_S1_g0_t0.imec0.ap_features_cache.npz" `
      --bhv   "D:\...\GlobalLogInt2023-08-15T11_14_10.csv" `
      --out   "D:\...\ephys_alignment_out\audit_plots"

All plots are saved as PNG.  No bin files needed — works purely from cache.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_cache(cache_path):
    d = np.load(str(cache_path), allow_pickle=False)
    traces, times = {}, {}
    for key in d.files:
        if key.startswith('trace_'):
            traces[key[6:]] = d[key]
        elif key.startswith('time_'):
            times[key[5:]]  = d[key]
    sync_edges = d['sync_edge_times'] if 'sync_edge_times' in d.files else None
    return traces, times, sync_edges


def load_bhv(bhv_path):
    df = pd.read_csv(bhv_path, header=None,
                     names=['timestamp', 'code'],
                     skipinitialspace=True)
    # Handle files that already have a header row
    try:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['code']      = pd.to_numeric(df['code'],      errors='coerce')
        df = df.dropna(subset=['timestamp', 'code'])
    except Exception:
        pass
        
    # --- THE FIX: Zero the behavior clock ---
    sr_mask = df['code'] == 1000
    if sr_mask.any():
        t0 = float(df['timestamp'][sr_mask].iloc[0])
        df['timestamp'] = df['timestamp'] - t0
    else:
        # Fallback if code 1000 is missing: subtract the very first timestamp
        t0 = float(df['timestamp'].iloc[0])
        df['timestamp'] = df['timestamp'] - t0
    # ----------------------------------------

    reward_times = df['timestamp'][df['code'] == 3].values.astype(float)
    return df, reward_times


def get_peaks(trace, times, min_height=1.5, min_distance_s=0.15):
    sr = 1.0 / np.median(np.diff(times[:10000]))
    dist_samp = max(1, int(min_distance_s * sr))
    peaks, props = find_peaks(trace, height=min_height, distance=dist_samp,
                               prominence=0.5)
    return times[peaks], props.get('peak_heights', trace[peaks])


def find_candidate_in_window(trace, times, rwd_t,
                              window=(-0.1, 0.5), min_height=1.5):
    """Return (best_time, best_height) or (None, None) if nothing found."""
    mask = (times >= rwd_t + window[0]) & (times <= rwd_t + window[1])
    if not np.any(mask):
        return None, None
    sub = trace[mask]
    sub_t = times[mask]
    idx = np.argmax(sub)
    if sub[idx] >= min_height:
        return float(sub_t[idx]), float(sub[idx])
    return None, None


def downsample_for_display(trace, times, max_pts=500_000):
    """Decimate trace/times arrays for plotting — keeps shape visible."""
    if len(trace) <= max_pts:
        return trace, times
    step = int(np.ceil(len(trace) / max_pts))
    return trace[::step], times[::step]


# ── Main plot function ────────────────────────────────────────────────────────

def plot_feature(fname, trace, times, reward_times, out_folder,
                 search_window=(-0.1, 0.5),
                 min_height=1.5, min_distance_s=0.15,
                 context_sec=60.0):
    """
    Produce two plots for one feature:
      1. Full session overview (small dots for peaks + reward lines)
      2. Zoomed view around the LARGEST ISI gap (your best landmark)
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    tr_ds, t_ds = downsample_for_display(trace, times)
    peak_times, peak_heights = get_peaks(trace, times,
                                          min_height=min_height,
                                          min_distance_s=min_distance_s)

    # ── Compute candidate hits and misses ─────────────────────────────────
    hit_times, hit_heights = [], []
    miss_times = []
    errors_ms  = []
    for rt in reward_times:
        bt, bh = find_candidate_in_window(trace, times, rt,
                                           window=search_window,
                                           min_height=min_height)
        if bt is not None:
            hit_times.append(bt)
            hit_heights.append(bh)
            errors_ms.append((bt - rt) * 1000.0)
        else:
            miss_times.append(rt)

    hit_times    = np.array(hit_times)
    hit_heights  = np.array(hit_heights)
    miss_times   = np.array(miss_times)
    errors_ms    = np.array(errors_ms)

    hit_rate  = len(hit_times) / len(reward_times) if len(reward_times) else 0
    false_pos = len([p for p in peak_times
                     if not np.any((p >= reward_times + search_window[0]) &
                                   (p <= reward_times + search_window[1]))])
    false_rate = false_pos / len(peak_times) if len(peak_times) else 0

    # ── Find largest ISI gap ──────────────────────────────────────────────
    isis = np.diff(reward_times)
    largest_gap_idx = int(np.argmax(isis))
    gap_start = reward_times[largest_gap_idx]
    gap_end   = reward_times[largest_gap_idx + 1]
    gap_dur   = isis[largest_gap_idx]

    # ── Figure 1: Full session ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(20, 10),
                              gridspec_kw={'height_ratios': [3, 1, 1]},
                              sharex=True)
    fig.suptitle(f"Feature audit: {fname}\n"
                 f"hit_rate={hit_rate:.2f}  "
                 f"false_peak_rate={false_rate:.2f}  "
                 f"n_peaks={len(peak_times)}  "
                 f"med_err={np.median(np.abs(errors_ms)):.0f}ms  "
                 f"n_rewards={len(reward_times)}",
                 fontsize=11)

    # Trace
    ax = axes[0]
    ax.plot(t_ds, tr_ds, lw=0.4, color='#444', alpha=0.7, label='trace')
    ax.axhline(min_height, color='gray', lw=0.8, ls='--', label=f'threshold={min_height}')

    # Reward lines
    for rt in reward_times:
        ax.axvline(rt, color='red', lw=0.4, alpha=0.3)

    # Highlight largest ISI gap
    ax.axvspan(gap_start, gap_end, color='gold', alpha=0.25,
               label=f'Largest gap ({gap_dur:.0f}s)')

    # Hits (blue) and misses (red X)
    if len(hit_times):
        ax.scatter(hit_times, hit_heights, s=20, color='royalblue',
                   zorder=5, label=f'hits ({len(hit_times)})')
    if len(miss_times):
        ax.scatter(miss_times,
                   [min_height * 1.1] * len(miss_times),
                   marker='x', s=40, color='crimson', zorder=5,
                   label=f'misses ({len(miss_times)})')

    ax.set_ylabel('z-score')
    ax.legend(loc='upper right', fontsize=7, ncol=3)

    # ISI plot
    ax2 = axes[1]
    isi_centers = (reward_times[:-1] + reward_times[1:]) / 2.0
    ax2.bar(isi_centers, isis, width=isis * 0.8, color='steelblue',
            alpha=0.6, align='center')
    ax2.axvspan(gap_start, gap_end, color='gold', alpha=0.3)
    ax2.set_ylabel('ISI (s)')

    # Error per trial
    ax3 = axes[2]
    if len(hit_times):
        ax3.scatter(hit_times, errors_ms, s=15, color='royalblue', alpha=0.7)
        ax3.axhline(0, color='black', lw=0.8)
        ax3.axhline(np.median(errors_ms), color='orange', lw=1.2, ls='--',
                    label=f'median={np.median(errors_ms):.0f}ms')
        ax3.set_ylim(-600, 600)
        ax3.legend(fontsize=7)
    ax3.set_ylabel('error (ms)')
    ax3.set_xlabel('behavior time (s)')

    plt.tight_layout()
    p1 = out_folder / f"{fname}_full_session.png"
    fig.savefig(str(p1), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p1.name}")

    # ── Figure 2: Zoomed into largest ISI gap ─────────────────────────────
    ctx = context_sec
    zoom_start = max(0, gap_start - ctx)
    zoom_end   = min(times[-1], gap_end + ctx)

    zm = (times >= zoom_start) & (times <= zoom_end)
    fig2, ax_z = plt.subplots(figsize=(18, 5))
    ax_z.set_title(f"{fname} — zoom around largest ISI gap "
                   f"({gap_dur:.0f}s between rewards at "
                   f"{gap_start:.0f}s and {gap_end:.0f}s)",
                   fontsize=10)

    tr_z = trace[zm]
    t_z  = times[zm]
    tr_z_ds, t_z_ds = downsample_for_display(tr_z, t_z, max_pts=100_000)
    ax_z.plot(t_z_ds, tr_z_ds, lw=0.6, color='#333')
    ax_z.axhline(min_height, color='gray', lw=0.8, ls='--')
    ax_z.axvspan(gap_start, gap_end, color='gold', alpha=0.25,
                  label=f'ISI gap = {gap_dur:.0f}s')

    for rt in reward_times:
        if zoom_start <= rt <= zoom_end:
            ax_z.axvline(rt, color='red', lw=1.0, alpha=0.6)

    in_zoom_hit = hit_times[(hit_times >= zoom_start) & (hit_times <= zoom_end)]
    in_zoom_hit_h = hit_heights[(hit_times >= zoom_start) & (hit_times <= zoom_end)]
    in_zoom_miss = miss_times[(miss_times >= zoom_start) & (miss_times <= zoom_end)]
    if len(in_zoom_hit):
        ax_z.scatter(in_zoom_hit, in_zoom_hit_h, s=50, color='royalblue',
                     zorder=5, label='hits')
    if len(in_zoom_miss):
        ax_z.scatter(in_zoom_miss, [min_height * 1.1] * len(in_zoom_miss),
                     marker='x', s=80, color='crimson', zorder=5,
                     label='misses')

    ax_z.set_xlabel('behavior time (s)')
    ax_z.set_ylabel('z-score')
    ax_z.legend(fontsize=8)

    plt.tight_layout()
    p2 = out_folder / f"{fname}_zoom_largest_gap.png"
    fig2.savefig(str(p2), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {p2.name}")

    # ── Figure 3: Error distribution histogram ────────────────────────────
    if len(errors_ms) >= 3:
        fig3, ax_h = plt.subplots(figsize=(8, 4))
        ax_h.hist(errors_ms, bins=30, color='steelblue', edgecolor='white')
        ax_h.axvline(np.median(errors_ms), color='orange', lw=2,
                     label=f'median={np.median(errors_ms):.0f}ms')
        ax_h.axvline(0, color='black', lw=1, ls='--')
        ax_h.set_xlabel('detection error (ms)')
        ax_h.set_ylabel('count')
        ax_h.set_title(f'{fname} — detection error distribution')
        ax_h.legend()
        plt.tight_layout()
        p3 = out_folder / f"{fname}_error_histogram.png"
        fig3.savefig(str(p3), dpi=120, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved: {p3.name}")

    return {
        'feature': fname,
        'hit_rate': hit_rate,
        'false_peak_rate': false_rate,
        'median_error_ms': float(np.median(np.abs(errors_ms))) if len(errors_ms) else float('nan'),
        'n_hits': len(hit_times),
        'n_misses': len(miss_times),
        'n_peaks': len(peak_times),
        'largest_isi_s': float(gap_dur),
        'largest_isi_start_s': float(gap_start),
        'largest_isi_end_s': float(gap_end),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Audit plots for ephys alignment features. '
                    'Works from cache — no bin files needed.')
    parser.add_argument('--cache', required=True,
                        help='Feature cache .npz file')
    parser.add_argument('--bhv', required=True,
                        help='Behavior CSV (GlobalLog*.csv)')
    parser.add_argument('--out', default=None,
                        help='Output folder for plots (default: next to cache)')
    parser.add_argument('--features', default='all',
                        help='Comma-separated feature names, or "all"')
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Peak detection threshold in z-scores (default 1.5)')
    parser.add_argument('--min-distance', type=float, default=0.15,
                        help='Min distance between peaks in seconds (default 0.15)')
    parser.add_argument('--context-sec', type=float, default=60.0,
                        help='Seconds of context around largest ISI gap zoom (default 60)')
    args = parser.parse_args()

    cache_path = Path(args.cache)
    out_folder = Path(args.out) if args.out else cache_path.parent / 'audit_plots'

    print(f"Loading cache: {cache_path.name}")
    traces, times, sync_edges = load_cache(cache_path)
    print(f"  Features in cache: {list(traces.keys())}")

    print(f"Loading behavior: {Path(args.bhv).name}")
    _, reward_times = load_bhv(args.bhv)
    print(f"  {len(reward_times)} reward events  "
          f"(session: {reward_times[0]:.0f}s – {reward_times[-1]:.0f}s)")

    isis = np.diff(reward_times)
    largest_gap_idx = int(np.argmax(isis))
    print(f"\nLargest ISI gap: {isis[largest_gap_idx]:.0f}s  "
          f"between reward {largest_gap_idx} ({reward_times[largest_gap_idx]:.0f}s) "
          f"and reward {largest_gap_idx+1} ({reward_times[largest_gap_idx+1]:.0f}s)")
    print(f"  → Use this gap as your visual landmark when inspecting plots.\n")

    if args.features == 'all':
        feat_names = list(traces.keys())
    else:
        feat_names = [f.strip() for f in args.features.split(',')]

    summaries = []
    for fname in feat_names:
        if fname not in traces:
            print(f"[WARN] {fname} not in cache, skipping.")
            continue
        print(f"\n── {fname} ──")
        s = plot_feature(
            fname,
            traces[fname],
            times[fname],
            reward_times,
            out_folder,
            min_height=args.threshold,
            min_distance_s=args.min_distance,
            context_sec=args.context_sec,
        )
        summaries.append(s)

    # Print summary table
    print("\n\n══ AUDIT SUMMARY ══")
    print(f"{'feature':<25} {'hit_rate':>9} {'false_rate':>10} "
          f"{'med_err_ms':>11} {'n_hits':>7} {'n_peaks':>8}")
    print('─' * 75)
    for s in sorted(summaries, key=lambda x: x['median_error_ms']):
        print(f"{s['feature']:<25} "
              f"{s['hit_rate']:>9.3f} "
              f"{s['false_peak_rate']:>10.3f} "
              f"{s['median_error_ms']:>11.1f} "
              f"{s['n_hits']:>7} "
              f"{s['n_peaks']:>8}")

    print(f"\nAll plots saved to: {out_folder}")


if __name__ == '__main__':
    main()
