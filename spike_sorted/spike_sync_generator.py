#!/usr/bin/env python3
"""
spike_sync_generator.py
=======================
Generates a high-quality sync.json from a single well-isolated anchor unit.

Strategy
--------
For each behavioral reward event (code 2):
  1. Predict the spike time in ephys: t_ephys_pred = t_bhv + peak_latency
  2. Search a narrow window (±search_win) around that prediction
  3. Take the nearest spike as the matched ephys anchor
  4. If no spike found → trial is unmatched (skipped)

Result: N matched (bhv_time, ephys_time) pairs → fit linear warp → sync.json

This replaces the feature-based warp (RMSE ~167ms) with spike-anchored
alignment (expected RMSE ~5-20ms depending on unit jitter).

Usage
-----
python spike_sync_generator.py \
    --ks-dir  "path/to/kilosort4" \
    --bhv     "path/to/GlobalLogInt*.csv" \
    --unit    358 \
    --latency 0.215 \
    --out     "path/to/output_sync.json"

The --unit and --latency values come from anchor_unit_hunter.py output.
If --unit is not specified, the script auto-selects the best unit by
re-running the PSTH scoring (slower but fully automatic).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AP_SR = 30_000.0


# ── I/O (shared with anchor_unit_hunter) ──────────────────────────────────────

def load_good_units(ks_dir):
    path = Path(ks_dir) / "cluster_group.tsv"
    if not path.exists():
        sys.exit(f"[ERROR] cluster_group.tsv not found in {ks_dir}")
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip().str.lower()
    
    # --- PATCHED LOGIC ---
    group_col = "group" if "group" in df.columns else "kslabel"
    good = set(df.loc[df[group_col].astype(str).str.strip().str.lower() == "good", "cluster_id"].astype(int).values)
    # ---------------------
    
    print(f"  {len(good)} good units")
    return good


def load_spikes(ks_dir, good_units):
    st = np.load(str(Path(ks_dir) / "spike_times.npy")).ravel().astype(np.int64)
    sc = np.load(str(Path(ks_dir) / "spike_clusters.npy")).ravel().astype(np.int64)
    mask = np.isin(sc, list(good_units))
    return st[mask] / AP_SR, sc[mask]


def load_behavior(bhv_path, target_code=2):
    df = pd.read_csv(str(bhv_path), header=None, names=["ts", "code"])
    df["ts"]   = pd.to_numeric(df["ts"],   errors="coerce")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df = df.dropna()

    # Zero-anchor to ephys start (code 1000) or first timestamp
    sr_mask = df["code"] == 1000
    t0 = float(df["ts"][sr_mask].iloc[0]) if sr_mask.any() \
         else float(df["ts"].iloc[0])
    df["ts"] -= t0

    events = df.loc[df["code"] == target_code, "ts"].values.astype(np.float64)
    print(f"  {len(events)} events with code={target_code} "
          f"(t={events.min():.1f}–{events.max():.1f}s)")
    return events


# ── Auto-select best unit if --unit not given ──────────────────────────────────

def score_unit(spikes_s, event_times, window=0.5, bin_size=0.010):
    """Return (score, peak_latency_s) for one unit."""
    bins = np.arange(-window, window + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    for et in event_times:
        offsets = spikes_s[np.searchsorted(spikes_s, et - window):
                           np.searchsorted(spikes_s, et + window)] - et
        c, _ = np.histogram(offsets, bins=bins)
        counts += c
    fr = counts / (len(event_times) * bin_size)
    centers = 0.5 * (bins[:-1] + bins[1:])
    base = float(np.mean(fr[centers < -0.1]))
    post_mask = centers >= -0.1
    peak_fr   = float(np.max(fr[post_mask]))
    peak_lat  = float(centers[post_mask][np.argmax(fr[post_mask])])
    return peak_fr - base, peak_lat


def auto_select_unit(all_spikes_s, all_clusters, good_units, event_times):
    print("  Auto-selecting best anchor unit (scoring all good units)…")
    best_score, best_uid, best_lat = -np.inf, None, None
    for uid in sorted(good_units):
        mask = all_clusters == uid
        spk  = np.sort(all_spikes_s[mask])
        if len(spk) < 500:
            continue
        score, lat = score_unit(spk, event_times)
        if score > best_score:
            best_score, best_uid, best_lat = score, uid, lat
    print(f"  Best unit: {best_uid}  score={best_score:.1f} Hz  "
          f"latency={best_lat*1000:.0f} ms")
    return best_uid, best_lat


# ── Core matching ──────────────────────────────────────────────────────────────

def match_spikes_to_events(unit_spikes_s, event_times_bhv,
                            peak_latency_s, search_win_s=0.075):
    """
    For each behavioral event, find the nearest spike within
    [t_bhv + peak_lat - search_win, t_bhv + peak_lat + search_win].

    Returns
    -------
    bhv_matched   : float array, behavior times of matched events (s)
    ephys_matched : float array, corresponding ephys spike times (s)
    errors_ms     : float array, |spike_time - predicted| in ms
    n_unmatched   : int
    """
    bhv_matched   = []
    ephys_matched = []
    errors_ms     = []
    n_unmatched   = 0

    for t_bhv in event_times_bhv:
        t_pred = t_bhv + peak_latency_s
        lo     = t_pred - search_win_s
        hi     = t_pred + search_win_s

        i0 = np.searchsorted(unit_spikes_s, lo)
        i1 = np.searchsorted(unit_spikes_s, hi)

        if i0 >= i1:
            n_unmatched += 1
            continue

        # Nearest spike to prediction
        candidates = unit_spikes_s[i0:i1]
        nearest    = candidates[np.argmin(np.abs(candidates - t_pred))]

        # The true behavior time for this anchor is:
        # t_bhv_anchor = t_spike_ephys - peak_latency
        # We store: bhv_time → ephys_spike_time
        bhv_matched.append(t_bhv)
        ephys_matched.append(float(nearest))
        errors_ms.append(abs(nearest - t_pred) * 1000.0)

    return (np.array(bhv_matched),
            np.array(ephys_matched),
            np.array(errors_ms),
            n_unmatched)


# ── Linear warp fit ────────────────────────────────────────────────────────────

def fit_warp(bhv_times, ephys_times):
    """
    Fit ephys_time = stretch * bhv_time + offset via least squares.
    Returns (stretch, offset, rmse_s).
    """
    A = np.column_stack([bhv_times, np.ones_like(bhv_times)])
    result = np.linalg.lstsq(A, ephys_times, rcond=None)
    stretch, offset = result[0]
    predicted = stretch * bhv_times + offset
    rmse = float(np.sqrt(np.mean((ephys_times - predicted) ** 2)))
    return float(stretch), float(offset), rmse


# ── Outlier rejection ──────────────────────────────────────────────────────────

def reject_outliers(bhv_times, ephys_times, n_iter=3, sigma=3.0):
    """
    Iterative sigma-clipping: fit warp, remove residuals > sigma*std, repeat.
    """
    mask = np.ones(len(bhv_times), dtype=bool)
    for _ in range(n_iter):
        if mask.sum() < 10:
            break
        stretch, offset, _ = fit_warp(bhv_times[mask], ephys_times[mask])
        residuals = ephys_times - (stretch * bhv_times + offset)
        std = np.std(residuals[mask])
        mask = np.abs(residuals) < sigma * std
    return mask


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def make_diagnostic_plot(bhv_matched, ephys_matched, stretch, offset, rmse,
                          errors_ms, unit_id, peak_lat_ms, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Spike-sync diagnostic  |  Unit {unit_id}  |  '
                 f'latency={peak_lat_ms:.0f}ms  |  RMSE={rmse*1000:.1f}ms',
                 fontsize=12, fontweight='bold')

    # 1. Bhv vs ephys scatter + warp line
    ax = axes[0]
    ax.scatter(bhv_matched, ephys_matched, s=8, alpha=0.5, color='steelblue',
               label=f'n={len(bhv_matched)} matched')
    t_line = np.array([bhv_matched.min(), bhv_matched.max()])
    ax.plot(t_line, stretch * t_line + offset, 'r-', lw=1.5,
            label=f'stretch={stretch:.6f}\noffset={offset:.3f}s')
    ax.set_xlabel('Behavior time (s)')
    ax.set_ylabel('Ephys spike time (s)')
    ax.set_title('Clock alignment')
    ax.legend(fontsize=8)

    # 2. Residuals over time
    ax = axes[1]
    residuals_ms = (ephys_matched - (stretch * bhv_matched + offset)) * 1000
    ax.scatter(bhv_matched, residuals_ms, s=6, alpha=0.5, color='darkorange')
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline( rmse * 1000, color='r', lw=0.8, ls='--',
               label=f'±RMSE={rmse*1000:.1f}ms')
    ax.axhline(-rmse * 1000, color='r', lw=0.8, ls='--')
    ax.set_xlabel('Behavior time (s)')
    ax.set_ylabel('Residual (ms)')
    ax.set_title('Residuals over session')
    ax.legend(fontsize=8)

    # 3. Error histogram
    ax = axes[2]
    ax.hist(errors_ms, bins=30, color='mediumseagreen', edgecolor='white')
    ax.axvline(np.median(errors_ms), color='r', lw=1.5,
               label=f'median={np.median(errors_ms):.1f}ms')
    ax.set_xlabel('|spike − predicted| (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Matching error distribution')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Diagnostic plot → {Path(out_path).name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate spike-anchored sync.json from a single anchor unit.')
    parser.add_argument('--ks-dir',   required=True,
                        help='Kilosort4 / Phy output directory')
    parser.add_argument('--bhv',      required=True,
                        help='GlobalLogInt*.csv behavior file')
    parser.add_argument('--unit',     type=int, default=None,
                        help='Anchor unit ID (from anchor_unit_hunter). '
                             'If omitted, auto-selected.')
    parser.add_argument('--latency',  type=float, default=None,
                        help='Peak latency in SECONDS (e.g. 0.215 for 215ms). '
                             'If omitted, auto-estimated.')
    parser.add_argument('--search-win', type=float, default=0.075,
                        help='Search window ±s around predicted spike (default 0.075)')
    parser.add_argument('--target-code', type=int, default=2,
                        help='Behavior event code to anchor on (default 2)')
    parser.add_argument('--out',      required=True,
                        help='Output sync.json path')
    parser.add_argument('--min-matches', type=int, default=20,
                        help='Minimum matched pairs required (default 20)')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  spike_sync_generator.py')
    print(f'{"="*60}')

    # ── Load data ──────────────────────────────────────────────
    print('\nLoading Phy outputs…')
    good_units    = load_good_units(args.ks_dir)
    all_spikes_s, all_clusters = load_spikes(args.ks_dir, good_units)
    print(f'  {len(all_spikes_s):,} spikes loaded')

    print('\nLoading behavior…')
    event_times = load_behavior(args.bhv, args.target_code)

    # ── Select anchor unit ──────────────────────────────────────
    if args.unit is not None and args.latency is not None:
        uid      = args.unit
        peak_lat = args.latency
        print(f'\nUsing specified anchor: Unit {uid}  latency={peak_lat*1000:.0f}ms')
    elif args.unit is not None and args.latency is None:
        uid = args.unit
        print(f'\nUnit {uid} specified — estimating latency…')
        mask     = all_clusters == uid
        spk      = np.sort(all_spikes_s[mask])
        _, peak_lat = score_unit(spk, event_times)
        print(f'  Estimated latency: {peak_lat*1000:.0f}ms')
    else:
        print('\nNo unit specified — auto-selecting…')
        uid, peak_lat = auto_select_unit(
            all_spikes_s, all_clusters, good_units, event_times)

    # ── Extract unit spikes ────────────────────────────────────
    unit_mask   = all_clusters == uid
    unit_spikes = np.sort(all_spikes_s[unit_mask])
    print(f'\nUnit {uid}: {len(unit_spikes):,} spikes  '
          f'latency={peak_lat*1000:.0f}ms  search_win=±{args.search_win*1000:.0f}ms')

    # ── Match spikes to events ─────────────────────────────────
    print('\nMatching spikes to behavioral events…')
    bhv_matched, ephys_matched, errors_ms, n_unmatched = match_spikes_to_events(
        unit_spikes, event_times, peak_lat, args.search_win)

    n_matched = len(bhv_matched)
    print(f'  Matched:   {n_matched} / {len(event_times)} events '
          f'({100*n_matched/len(event_times):.0f}%)')
    print(f'  Unmatched: {n_unmatched}')
    print(f'  Median matching error: {np.median(errors_ms):.1f}ms')

    if n_matched < args.min_matches:
        print(f'\n[ERROR] Only {n_matched} matches — below minimum {args.min_matches}.')
        print('  Try: --search-win 0.1  or  --latency <value from hunter>')
        sys.exit(1)

    # ── Outlier rejection ──────────────────────────────────────
    print('\nRejecting outliers (3-sigma iterative)…')
    keep = reject_outliers(bhv_matched, ephys_matched)
    n_kept = keep.sum()
    print(f'  Kept {n_kept} / {n_matched} pairs after outlier rejection')

    bhv_clean   = bhv_matched[keep]
    ephys_clean = ephys_matched[keep]

    # ── Fit warp ───────────────────────────────────────────────
    stretch, offset, rmse = fit_warp(bhv_clean, ephys_clean)
    print(f'\nWarp fit:')
    print(f'  stretch = {stretch:.8f}')
    print(f'  offset  = {offset:.4f}s  ({offset*1000:.1f}ms)')
    print(f'  RMSE    = {rmse*1000:.2f}ms  ({n_kept} pairs)')

    if rmse > 0.050:
        print(f'  [WARN] RMSE > 50ms — anchor unit may be noisy. '
              f'Consider trying a different unit.')
    elif rmse > 0.020:
        print(f'  [OK] RMSE 20-50ms — usable but not ideal.')
    else:
        print(f'  [GOOD] RMSE < 20ms — high-quality alignment.')

    # ── Save sync.json ─────────────────────────────────────────
    # Format compatible with existing ephys_alignment_fusion.py / feature_explorer.py
    sync_data = {
        "warp": {
            "stretch": stretch,
            "offset":  offset,
            "rmse":    rmse,
        },
        "events": [
            {"bhv_time": float(b), "ephys_time": float(e)}
            for b, e in zip(bhv_clean, ephys_clean)
        ],
        "metadata": {
            "method":        "spike_anchor",
            "anchor_unit":   int(uid),
            "peak_lat_ms":   round(peak_lat * 1000, 1),
            "search_win_ms": round(args.search_win * 1000, 1),
            "n_matched":     int(n_matched),
            "n_kept":        int(n_kept),
            "n_events_total": int(len(event_times)),
            "target_code":   args.target_code,
            "ks_dir":        str(Path(args.ks_dir).resolve()),
        }
    }

    with open(str(out_path), 'w') as f:
        json.dump(sync_data, f, indent=2)
    print(f'\n  sync.json → {out_path}')

    # ── Diagnostic plot ────────────────────────────────────────
    plot_path = out_path.parent / (out_path.stem + '_diagnostic.png')
    make_diagnostic_plot(bhv_clean, ephys_clean, stretch, offset, rmse,
                         errors_ms[keep], uid, peak_lat * 1000, plot_path)

    # ── Summary ────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  DONE')
    print(f'  Anchor unit : {uid}  (latency {peak_lat*1000:.0f}ms)')
    print(f'  Pairs used  : {n_kept}')
    print(f'  RMSE        : {rmse*1000:.2f}ms')
    print(f'  vs before   : was 167ms  →  now {rmse*1000:.2f}ms')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
