#!/usr/bin/env python3
"""
feature_explorer.py
===================
Comprehensive ERP-style analysis on a 500-second chunk of ephys features.

For every feature in the cache, generates:
  1. A 2×2 ERP grid (codes 2, 31, 11, -11) — individual trials + mean ± SEM
  2. A zoomed continuous trace (full 500s window) with all event types marked
  3. A summary heatmap: which features respond to which event codes

Usage (PowerShell):
    python feature_explorer.py `
      --cache "...\\features_cache.npz" `
      --bhv   "...\\GlobalLogInt*.csv" `
      --out   "...\\explorer_out" `
      --tmin 1500 --tmax 2000 `
      --pre 1.0 --post 2.0 `
      --features all

The behavior file timestamps are normalised automatically (subtract first timestamp)
to match the ephys-relative time base stored in the cache.

What is NOT yet built here (tell the next Claude):
  - PC2 / PC3 extraction (requires re-reading the bin file;
    add --n-components 3 to extract_lfp_deflection and store pc1/pc2/pc3 separately)
  - Different filter bands on LF (delta 1-4, theta 4-8, beta 8-30, gamma 30-80)
    before PCA (requires bin access or a second pass on the raw LF cache if stored)
  - AP blocks with different bands (MUA 300-3000, HFO 80-200, gamma 30-80)
  - Solenoid by depth blocks (median per 20-channel block instead of global)
  - Template matching on AP and LF
  All of those need bin-level access and should be added to a new
  extract_extra_features.py that saves its output to the same .npz cache.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


# ── Constants ─────────────────────────────────────────────────────────────────

EVENT_CODES = {
    2:   ('Trial init / water',  'red'),
    3:   ('Reward delivered',    'blue'),
    31:  ('Reach',               'cyan'),
    11:  ('Pull',                'orange'),
    -11: ('Inverse pull',        'magenta'),
}

WATER_DELAY_S = 0.221   # solenoid opens ~221 ms after code 2 in this rig


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cache(path):
    """Load feature cache; return (traces_dict, times_dict)."""
    d = np.load(str(path), allow_pickle=False)
    traces, times = {}, {}
    for k in d.files:
        if k.startswith('trace_'):
            traces[k[6:]] = d[k]
        elif k.startswith('time_'):
            times[k[5:]] = d[k]
    # Legacy caches store features without prefix
    for k in d.files:
        if k not in {'sync_edge_times', 'sr_ap', 'sr_lf'} \
                and not k.startswith('trace_') \
                and not k.startswith('time_'):
            if k not in traces:
                traces[k] = d[k]
    return traces, times


def load_bhv(path):
    """Load GlobalLogInt CSV; normalise timestamps to ephys-relative seconds."""
    df = pd.read_csv(str(path), header=None, names=['timestamp', 'code'])
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['code']      = pd.to_numeric(df['code'],      errors='coerce')
    df = df.dropna()
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    return df


def infer_sr(times_arr):
    """Estimate sample rate from a time axis array."""
    if times_arr is None or len(times_arr) < 10:
        return 1000.0
    return float(1.0 / np.median(np.diff(times_arr[:min(10000, len(times_arr))])))


def get_times_for_feature(fname, traces, times):
    """Return the time axis array for a feature, inferring SR if needed."""
    if fname in times:
        return times[fname]
    # Reconstruct from trace length assuming common rates
    n = len(traces[fname])
    # Guess: solenoid = 30kHz, LFP = 2500Hz, MUA = 1000Hz
    if 'solenoid' in fname:
        sr = 30000.0
    elif 'lfp' in fname:
        sr = 2500.0
    else:
        sr = 1000.0
    return np.arange(n) / sr


def slice_window(trace, time_arr, tmin, tmax):
    """Return (trace_slice, time_slice) for [tmin, tmax]."""
    mask = (time_arr >= tmin) & (time_arr <= tmax)
    return trace[mask], time_arr[mask]


def extract_erp_windows(trace, time_arr, event_times, pre_s=1.0, post_s=2.0):
    """
    Cut trace into windows around each event.
    Returns (windows array [n_trials × n_samples], time_axis relative to event).
    """
    sr = infer_sr(time_arr)
    pre_n  = int(pre_s * sr)
    post_n = int(post_s * sr)
    n_pts  = pre_n + post_n

    windows = []
    for t in event_times:
        # Find nearest sample
        idx = np.searchsorted(time_arr, t)
        s = idx - pre_n
        e = idx + post_n
        if s >= 0 and e <= len(trace):
            windows.append(trace[s:e])

    if not windows:
        return np.empty((0, n_pts)), np.linspace(-pre_s, post_s, n_pts)

    t_axis = np.linspace(-pre_s, post_s, n_pts)
    return np.vstack(windows), t_axis


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_erp_grid(fname, trace, time_arr, bhv_df, tmin, tmax,
                  pre_s, post_s, out_folder):
    """
    2×2 grid: one panel per event code (2, 31, 11, -11).
    Each panel: individual trial traces (thin, semi-transparent) +
                mean ± SEM (thick black) + event line at t=0.
    Code 2 panel gets an extra dashed line at +221ms for water.
    """
    codes_to_plot = [2, 31, 11, -11]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Feature: {fname}   |   Window: {tmin}–{tmax}s',
                 fontsize=13, fontweight='bold')

    trace_win, time_win = slice_window(trace, time_arr, tmin - pre_s,
                                        tmax + post_s)

    for ax, code in zip(axes.flat, codes_to_plot):
        label, color = EVENT_CODES.get(code, (f'Code {code}', 'gray'))

        # Event times within window
        ev_t = bhv_df.loc[bhv_df['code'] == code, 'timestamp'].values
        ev_t = ev_t[(ev_t >= tmin) & (ev_t <= tmax)]

        if len(ev_t) == 0:
            ax.text(0.5, 0.5, f'No events\n(code {code})',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label}  (n=0)')
            continue

        windows, t_axis = extract_erp_windows(trace_win, time_win,
                                               ev_t, pre_s, post_s)
        n = windows.shape[0]
        if n == 0:
            ax.text(0.5, 0.5, 'No complete windows',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label}  (n=0)')
            continue

        # Individual trials
        alpha_t = max(0.05, min(0.3, 3.0 / n))
        for row in windows:
            ax.plot(t_axis, row, color='gray', lw=0.4, alpha=alpha_t)

        # Mean ± SEM
        mean  = windows.mean(axis=0)
        sem   = windows.std(axis=0) / np.sqrt(n)
        ax.plot(t_axis, mean, color='black', lw=2.0, label='mean')
        ax.fill_between(t_axis, mean - sem, mean + sem,
                         color='black', alpha=0.2)

        # Event line
        ax.axvline(0, color=color, lw=1.5, ls='-', label='event')

        # Water delay for code 2
        if code == 2:
            ax.axvline(WATER_DELAY_S, color='blue', lw=1.5, ls='--',
                        label=f'+{int(WATER_DELAY_S*1000)}ms water')

        ax.axhline(0, color='silver', lw=0.5)
        ax.set_xlim(-pre_s, post_s)
        ax.set_title(f'{label}  (n={n})', fontsize=10)
        ax.set_xlabel('Time from event (s)', fontsize=8)
        ax.set_ylabel('Z-score', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fname_safe = fname.replace('/', '_')
    out_path = Path(out_folder) / f'erp_{fname_safe}.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def plot_continuous_overview(fname, trace, time_arr, bhv_df, tmin, tmax,
                              out_folder):
    """
    Continuous trace for the 500s window with all event types as vertical lines.
    Useful for visual inspection: do peaks align with any events?
    """
    trace_win, time_win = slice_window(trace, time_arr, tmin, tmax)
    if len(trace_win) == 0:
        return

    # Downsample for display (max 100k points)
    step = max(1, len(trace_win) // 100_000)
    t_ds = time_win[::step]
    tr_ds = trace_win[::step]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(t_ds, tr_ds, lw=0.5, color='#333', alpha=0.8)
    ax.axhline(0, color='silver', lw=0.5)

    for code, (label, color) in EVENT_CODES.items():
        ev_t = bhv_df.loc[bhv_df['code'] == code, 'timestamp'].values
        ev_t = ev_t[(ev_t >= tmin) & (ev_t <= tmax)]
        for i, t in enumerate(ev_t):
            ax.axvline(t, color=color, lw=0.8, alpha=0.6,
                        label=label if i == 0 else '')

    ax.set_title(f'{fname}  |  {tmin}–{tmax}s', fontsize=11)
    ax.set_xlabel('Session time (s)')
    ax.set_ylabel('Z-score')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
               loc='upper right', fontsize=7, ncol=3)

    plt.tight_layout()
    fname_safe = fname.replace('/', '_')
    out_path = Path(out_folder) / f'overview_{fname_safe}.png'
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary_heatmap(summary, out_folder):
    """
    Heatmap: rows = features, cols = event codes.
    Cell value = mean absolute ERP amplitude (rough consistency score).
    Higher = more consistent response = better feature candidate.
    """
    features = list(summary.keys())
    codes    = [2, 31, 11, -11]
    code_labels = [EVENT_CODES[c][0] for c in codes]

    mat = np.zeros((len(features), len(codes)))
    for fi, fname in enumerate(features):
        for ci, code in enumerate(codes):
            mat[fi, ci] = summary[fname].get(code, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(codes) * 1.8),
                                     max(4, len(features) * 0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(code_labels, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    plt.colorbar(im, ax=ax, label='Mean |ERP| amplitude (z-score)')
    ax.set_title('Feature × Event Code response summary\n'
                 '(brighter = stronger / more consistent ERP)',
                 fontsize=11)
    plt.tight_layout()
    out_path = Path(out_folder) / 'summary_heatmap.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[SUMMARY] Heatmap saved → {out_path.name}')


def compute_erp_score(windows):
    """
    Simple consistency score: mean absolute amplitude of the mean ERP,
    normalised by the per-trial std (measures SNR of the average).
    Returns a scalar >= 0. Higher is better.
    """
    if windows.shape[0] < 3:
        return 0.0
    mean_erp   = windows.mean(axis=0)
    trial_noise = windows.std(axis=0).mean() + 1e-9
    return float(np.abs(mean_erp).mean() / trial_noise)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ERP-style feature explorer for ephys alignment caches.')
    parser.add_argument('--cache',    required=True,
                        help='Feature cache .npz file')
    parser.add_argument('--bhv',      required=True,
                        help='GlobalLogInt*.csv behaviour file')
    parser.add_argument('--out',      default=None,
                        help='Output folder (default: cache_dir/explorer_out)')
    parser.add_argument('--tmin',     type=float, default=1500.0,
                        help='Start of analysis window in session seconds (default 1500)')
    parser.add_argument('--tmax',     type=float, default=2000.0,
                        help='End of analysis window (default 2000)')
    parser.add_argument('--pre',      type=float, default=1.0,
                        help='Seconds before event for ERP window (default 1.0)')
    parser.add_argument('--post',     type=float, default=2.0,
                        help='Seconds after event for ERP window (default 2.0)')
    parser.add_argument('--features', default='all',
                        help='Comma-separated feature names or "all"')
    parser.add_argument('--skip-overview', action='store_true',
                        help='Skip the continuous overview plots (faster)')
    args = parser.parse_args()

    cache_path = Path(args.cache)
    out_folder = Path(args.out) if args.out else cache_path.parent / 'explorer_out'
    out_folder.mkdir(parents=True, exist_ok=True)

    # ── Load cache ─────────────────────────────────────────────────────────
    print(f'Loading cache: {cache_path.name}')
    traces, times = load_cache(cache_path)
    print(f'  Features: {sorted(traces.keys())}')

    # ── Load behaviour ─────────────────────────────────────────────────────
    print(f'Loading behaviour: {Path(args.bhv).name}')
    bhv_df = load_bhv(args.bhv)

    # Print event counts in window
    for code, (label, _) in EVENT_CODES.items():
        n = ((bhv_df['code'] == code) &
             (bhv_df['timestamp'] >= args.tmin) &
             (bhv_df['timestamp'] <= args.tmax)).sum()
        print(f'  Code {code:>4} ({label}): {n} events in {args.tmin}–{args.tmax}s')

    # ── Select features ────────────────────────────────────────────────────
    if args.features == 'all':
        feat_names = sorted(traces.keys())
    else:
        feat_names = [f.strip() for f in args.features.split(',')]
        feat_names = [f for f in feat_names if f in traces]

    print(f'\nAnalysing {len(feat_names)} features...\n')

    # ── Main loop ──────────────────────────────────────────────────────────
    summary = {}   # fname → {code: score}

    for fname in feat_names:
        print(f'  ── {fname}')
        trace    = traces[fname]
        time_arr = get_times_for_feature(fname, traces, times)

        sr = infer_sr(time_arr)
        n_in_window = int((args.tmax - args.tmin) * sr)
        print(f'     sr≈{sr:.0f}Hz  trace_len={len(trace)}  '
              f'window_samples≈{n_in_window}')

        # ERP grid (2×2)
        p = plot_erp_grid(fname, trace, time_arr, bhv_df,
                           args.tmin, args.tmax,
                           args.pre, args.post, out_folder)
        print(f'     ERP grid → {Path(p).name}')

        # Continuous overview
        if not args.skip_overview:
            plot_continuous_overview(fname, trace, time_arr, bhv_df,
                                      args.tmin, args.tmax, out_folder)

        # Compute summary scores
        summary[fname] = {}
        trace_win, time_win = slice_window(trace, time_arr,
                                            args.tmin - args.pre,
                                            args.tmax + args.post)
        for code in [2, 31, 11, -11]:
            ev_t = bhv_df.loc[bhv_df['code'] == code, 'timestamp'].values
            ev_t = ev_t[(ev_t >= args.tmin) & (ev_t <= args.tmax)]
            if len(ev_t) == 0:
                summary[fname][code] = 0.0
                continue
            windows, _ = extract_erp_windows(trace_win, time_win,
                                               ev_t, args.pre, args.post)
            summary[fname][code] = compute_erp_score(windows)

    # ── Summary heatmap ────────────────────────────────────────────────────
    plot_summary_heatmap(summary, out_folder)

    # ── Print top features per code ────────────────────────────────────────
    print('\n══ TOP FEATURES PER EVENT CODE ══')
    for code in [2, 31, 11, -11]:
        label = EVENT_CODES[code][0]
        ranked = sorted(summary.items(),
                        key=lambda x: x[1].get(code, 0.0), reverse=True)
        top5 = [(f, f'{s[code]:.3f}') for f, s in ranked[:5] if s.get(code, 0) > 0]
        print(f'\n  Code {code} ({label}):')
        for rank, (f, score) in enumerate(top5, 1):
            print(f'    {rank}. {f:<30} score={score}')

    print(f'\nAll outputs → {out_folder}')


if __name__ == '__main__':
    main()
