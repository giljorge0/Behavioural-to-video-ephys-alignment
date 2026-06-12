#!/usr/bin/env python3
"""
multi_anchor_sync.py
====================
Fuses multiple anchor neurons into a single high-quality sync.json.

Why this helps
--------------
A single anchor unit matches ~70-95% of trials (some trials the neuron
just didn't fire in the search window). With N independent anchor units
you cover different trials: if each unit hits 80% of trials, three units
can cover ~99.2% of trials (1 - 0.2^3). More pairs → lower RMSE.

Approach
--------
For each behavioral reward event:
  1. Try every anchor unit in the list.
  2. For each unit, predict t_ephys = t_bhv + unit_latency and search
     a tight window (±search_win) for the nearest spike.
  3. Collect all matched estimates for that event.
  4. Take the WEIGHTED MEDIAN across anchor estimates, weighted by
     1 / (jitter_ms + 1) — units with tighter latency get more weight.

The fused (bhv_time, ephys_time) pairs are then fitted with the same
linear warp as before.

Inputs
------
--ks-dir          Kilosort output directory
--bhv             GlobalLogInt*.csv
--anchors         JSON file listing anchor units:
                  [
                    {"unit": 35,  "latency_s": 0.115, "search_win_s": 0.050},
                    {"unit": 173, "latency_s": 0.225, "search_win_s": 0.050},
                    ...
                  ]
                  OR pass --auto-n N to auto-select the top N anchor units
                  using the same scorer as anchor_unit_hunter.py.
--out             Output sync.json path

Output
------
  <out>/spike_sync_fused.json   — compatible with downstream pipeline
  <out>/fusion_diagnostic.png   — per-anchor coverage + combined RMSE

Usage examples
--------------
# From a JSON list of pre-identified anchors:
python multi_anchor_sync.py \
    --ks-dir  "path/to/kilosort4" \
    --bhv     "GlobalLogInt*.csv" \
    --anchors "anchors.json" \
    --out     "outputs/Ferrero_R1_imec0"

# Auto-select top 5 anchor units:
python multi_anchor_sync.py \
    --ks-dir  "path/to/kilosort4" \
    --bhv     "GlobalLogInt*.csv" \
    --auto-n  5 \
    --out     "outputs/Ferrero_R1_imec0"
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


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_good_units(ks_dir):
    path = Path(ks_dir) / "cluster_group.tsv"
    if not path.exists():
        sys.exit(f"[ERROR] cluster_group.tsv not found in {ks_dir}")
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip().str.lower()
    group_col = "group" if "group" in df.columns else "kslabel"
    good = set(df.loc[df[group_col].str.strip().str.lower() == "good",
                      "cluster_id"].astype(int).values)
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
    sr_mask = df["code"] == 1000
    t0 = float(df["ts"][sr_mask].iloc[0]) if sr_mask.any() \
         else float(df["ts"].iloc[0])
    df["ts"] -= t0
    events = df.loc[df["code"] == target_code, "ts"].values.astype(np.float64)
    print(f"  {len(events)} events with code={target_code}")
    return events


# ── Anchor scoring (copied from anchor_unit_hunter) ───────────────────────────

def score_unit(spikes_s, event_times, window=0.5, bin_size=0.010):
    bins   = np.arange(-window, window + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    for et in event_times:
        offsets = spikes_s[
            np.searchsorted(spikes_s, et - window):
            np.searchsorted(spikes_s, et + window)
        ] - et
        c, _ = np.histogram(offsets, bins=bins)
        counts += c
    fr       = counts / (len(event_times) * bin_size)
    centers  = 0.5 * (bins[:-1] + bins[1:])
    base     = float(np.mean(fr[centers < -0.1]))
    post     = centers >= -0.1
    peak_fr  = float(np.max(fr[post]))
    peak_lat = float(centers[post][np.argmax(fr[post])])
    return peak_fr - base, peak_lat


def auto_select_anchors(all_spikes_s, all_clusters, good_units,
                         event_times, n_top=5, min_spikes=500):
    """Return list of (uid, latency_s, score) sorted by score descending."""
    print(f"  Scoring all {len(good_units)} good units for top-{n_top} anchors…")
    results = []
    for uid in sorted(good_units):
        mask = all_clusters == uid
        spk  = np.sort(all_spikes_s[mask])
        if len(spk) < min_spikes:
            continue
        sc, lat = score_unit(spk, event_times)
        results.append((uid, lat, sc))
    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:n_top]
    print(f"\n  Top {len(top)} anchor candidates:")
    print(f"  {'Unit':>6}  {'Latency (ms)':>14}  {'Score (Hz above base)':>22}")
    print(f"  {'-'*50}")
    for uid, lat, sc in top:
        print(f"  {uid:>6}  {lat*1000:>14.1f}  {sc:>22.1f}")
    return [{"unit": uid, "latency_s": lat, "search_win_s": 0.050}
            for uid, lat, sc in top]


# ── Single-unit spike matching ─────────────────────────────────────────────────

def match_unit(unit_spikes_s, event_times, latency_s, search_win_s):
    """
    Returns arrays: matched_event_idx, matched_ephys_times, jitter_ms
    (only for events that had a spike in the search window)
    """
    ev_idx, eph_times, jitter = [], [], []
    for i, t_bhv in enumerate(event_times):
        t_pred = t_bhv + latency_s
        lo = t_pred - search_win_s
        hi = t_pred + search_win_s
        i0 = np.searchsorted(unit_spikes_s, lo)
        i1 = np.searchsorted(unit_spikes_s, hi)
        if i0 < i1:
            cands   = unit_spikes_s[i0:i1]
            nearest = cands[np.argmin(np.abs(cands - t_pred))]
            ev_idx.append(i)
            eph_times.append(float(nearest - latency_s))  # back-compute bhv-equivalent
            jitter.append(abs(nearest - t_pred) * 1000.0)
    return (np.array(ev_idx,   dtype=int),
            np.array(eph_times, dtype=np.float64),
            np.array(jitter,    dtype=np.float64))


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse_anchor_estimates(anchor_matches, event_times):
    """
    For each behavioral event, collect all anchor-unit estimates.
    The fused estimate = weighted median (weight = 1 / (jitter_ms + 1)).
    Returns (bhv_matched, ephys_matched) arrays — one row per successfully
    covered event.
    """
    # Build: event_idx → list of (ephys_estimate, weight)
    event_pool = {i: [] for i in range(len(event_times))}
    for uid, ev_idx, eph_times, jitter in anchor_matches:
        for i, et, j in zip(ev_idx, eph_times, jitter):
            w = 1.0 / (j + 1.0)   # weight = 1 / (jitter_ms + 1)
            event_pool[i].append((et, w))

    bhv_fused  = []
    eph_fused  = []
    n_single   = 0
    n_multi    = 0
    n_miss     = 0

    for ev_idx, estimates in event_pool.items():
        if len(estimates) == 0:
            n_miss += 1
            continue
        if len(estimates) == 1:
            n_single += 1
        else:
            n_multi += 1

        times   = np.array([e[0] for e in estimates])
        weights = np.array([e[1] for e in estimates])
        weights /= weights.sum()

        # Weighted median
        sort_idx   = np.argsort(times)
        times_s    = times[sort_idx]
        weights_s  = weights[sort_idx]
        cum_w      = np.cumsum(weights_s)
        med_idx    = np.searchsorted(cum_w, 0.5)
        fused_eph  = float(times_s[min(med_idx, len(times_s) - 1)])

        bhv_fused.append(float(event_times[ev_idx]))
        eph_fused.append(fused_eph)

    print(f"\n  Coverage: {len(bhv_fused)}/{len(event_times)} events fused")
    print(f"    Multi-anchor : {n_multi}")
    print(f"    Single-anchor: {n_single}")
    print(f"    Missed       : {n_miss}")

    return np.array(bhv_fused), np.array(eph_fused)


# ── Linear warp + outlier rejection ───────────────────────────────────────────

def fit_warp(bhv_t, eph_t):
    A  = np.column_stack([bhv_t, np.ones_like(bhv_t)])
    result = np.linalg.lstsq(A, eph_t, rcond=None)
    stretch, offset = result[0]
    pred = stretch * bhv_t + offset
    rmse = float(np.sqrt(np.mean((eph_t - pred) ** 2)))
    return float(stretch), float(offset), rmse


def reject_outliers(bhv_t, eph_t, n_iter=3, sigma=3.0):
    mask = np.ones(len(bhv_t), dtype=bool)
    for _ in range(n_iter):
        if mask.sum() < 10:
            break
        stretch, offset, _ = fit_warp(bhv_t[mask], eph_t[mask])
        resid = eph_t - (stretch * bhv_t + offset)
        std   = float(np.std(resid[mask]))
        mask  = np.abs(resid) < sigma * std
    return mask


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def plot_fusion_diagnostic(anchor_matches, bhv_fused, eph_fused,
                            stretch, offset, rmse, event_times,
                            anchor_specs, out_path):
    n_anchors = len(anchor_specs)
    fig = plt.figure(figsize=(16, 4 + 3 * n_anchors))
    gs  = plt.GridSpec(2 + n_anchors, 3, figure=fig,
                       hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"Multi-anchor Fusion Diagnostic  |  "
        f"{n_anchors} anchors  |  RMSE = {rmse*1000:.2f}ms  "
        f"|  n = {len(bhv_fused)} pairs",
        fontsize=12, fontweight="bold"
    )

    # ── Fused warp scatter ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(bhv_fused, eph_fused, s=8, alpha=0.6, color="steelblue",
               label=f"n={len(bhv_fused)}")
    t_line = np.array([bhv_fused.min(), bhv_fused.max()])
    ax.plot(t_line, stretch * t_line + offset, "r-", lw=1.5,
            label=f"stretch={stretch:.6f}\noffset={offset*1000:.1f}ms")
    ax.set_xlabel("Behavior time (s)", fontsize=9)
    ax.set_ylabel("Ephys time (s)", fontsize=9)
    ax.set_title("Fused warp (all anchors)", fontsize=10)
    ax.legend(fontsize=8)

    # ── Fused residuals ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    resid = (eph_fused - (stretch * bhv_fused + offset)) * 1000
    ax.scatter(bhv_fused, resid, s=8, alpha=0.5, color="darkorange")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline( rmse*1000, color="r", lw=1.0, ls="--",
               label=f"±RMSE={rmse*1000:.1f}ms")
    ax.axhline(-rmse*1000, color="r", lw=1.0, ls="--")
    ax.set_xlabel("Behavior time (s)", fontsize=9)
    ax.set_ylabel("Residual (ms)", fontsize=9)
    ax.set_title("Residuals", fontsize=10)
    ax.legend(fontsize=8)

    # ── Coverage bar chart ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    labels   = [f"Unit {a['unit']}" for a in anchor_specs]
    coverage = [len(m[1]) for m in anchor_matches]
    total    = len(event_times)
    bars = ax.barh(labels, [100 * c / total for c in coverage],
                   color="steelblue", alpha=0.75)
    ax.axvline(100 * len(bhv_fused) / total, color="red", lw=1.5, ls="--",
               label=f"Fused={100*len(bhv_fused)/total:.1f}%")
    ax.set_xlabel("Coverage (%)", fontsize=9)
    ax.set_title(f"Per-anchor coverage (n={total} events)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8)
    for bar, cov in zip(bars, coverage):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{100*cov/total:.0f}%", va="center", fontsize=8)

    # ── Per-anchor scatter panels ─────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    for row, (am, spec, color) in enumerate(
            zip(anchor_matches, anchor_specs, colors)):
        uid, ev_idx, eph_times, jitter = am

        # Back-compute bhv times for this anchor
        bhv_a = event_times[ev_idx]
        eph_a = eph_times   # already latency-corrected
        pred_a = stretch * bhv_a + offset
        resid_a = (eph_a - pred_a) * 1000

        ax_sc  = fig.add_subplot(gs[2 + row, 0])
        ax_res = fig.add_subplot(gs[2 + row, 1])
        ax_jit = fig.add_subplot(gs[2 + row, 2])

        ax_sc.scatter(bhv_a, eph_a, s=6, alpha=0.6, color=color)
        ax_sc.plot([bhv_a.min(), bhv_a.max()],
                   [stretch * bhv_a.min() + offset,
                    stretch * bhv_a.max() + offset], "r-", lw=1.2)
        ax_sc.set_title(f"Unit {uid}  lat={spec['latency_s']*1000:.0f}ms  "
                        f"n={len(ev_idx)}", fontsize=9)
        ax_sc.set_xlabel("Bhv (s)", fontsize=8)
        ax_sc.set_ylabel("Ephys (s)", fontsize=8)

        ax_res.scatter(bhv_a, resid_a, s=6, alpha=0.5, color=color)
        ax_res.axhline(0, color="k", lw=0.6)
        ax_res.set_ylabel("Resid (ms)", fontsize=8)
        ax_res.set_title(f"RMSE={np.sqrt(np.mean(resid_a**2)):.1f}ms", fontsize=9)

        ax_jit.hist(jitter, bins=20, color=color, alpha=0.75, edgecolor="none")
        ax_jit.axvline(np.median(jitter), color="red", lw=1.2,
                       label=f"med={np.median(jitter):.1f}ms")
        ax_jit.set_xlabel("Jitter (ms)", fontsize=8)
        ax_jit.set_title(f"Unit {uid} jitter", fontsize=9)
        ax_jit.legend(fontsize=7)

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fusion diagnostic → {Path(out_path).name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuse multiple anchor units into a single high-quality sync")
    parser.add_argument("--ks-dir",       required=True)
    parser.add_argument("--bhv",          required=True)
    parser.add_argument("--anchors",      default=None,
                        help="JSON list of anchor specs: "
                             '[{"unit":35,"latency_s":0.115,"search_win_s":0.05},...]')
    parser.add_argument("--auto-n",       type=int, default=None,
                        help="Auto-select top N anchor units (alternative to --anchors)")
    parser.add_argument("--target-code",  type=int, default=2,
                        help="Behavioral event code (default 2 = reward)")
    parser.add_argument("--search-win",   type=float, default=0.050,
                        help="Default search window ±s (default 0.050)")
    parser.add_argument("--min-spikes",   type=int, default=500)
    parser.add_argument("--out",          required=True)
    args = parser.parse_args()

    if args.anchors is None and args.auto_n is None:
        sys.exit("[ERROR] Provide either --anchors <file.json> or --auto-n <N>")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  multi_anchor_sync.py')
    print(f'{"="*60}\n')

    print("Loading Phy outputs…")
    good_units              = load_good_units(args.ks_dir)
    all_spikes_s, all_cl    = load_spikes(args.ks_dir, good_units)
    print(f"  {len(all_spikes_s):,} spikes loaded")

    print("\nLoading behavior…")
    event_times = load_behavior(args.bhv, args.target_code)

    # ── Select anchors ──────────────────────────────────────────────────────
    if args.anchors:
        with open(args.anchors) as f:
            anchor_specs = json.load(f)
        print(f"\nLoaded {len(anchor_specs)} anchors from {args.anchors}")
    else:
        anchor_specs = auto_select_anchors(
            all_spikes_s, all_cl, good_units, event_times,
            n_top=args.auto_n, min_spikes=args.min_spikes
        )

    # Ensure search_win is set
    for a in anchor_specs:
        if "search_win_s" not in a:
            a["search_win_s"] = args.search_win

    # ── Match each anchor independently ────────────────────────────────────
    print(f"\nMatching {len(anchor_specs)} anchor units…")
    anchor_matches = []
    for spec in anchor_specs:
        uid    = int(spec["unit"])
        lat    = float(spec["latency_s"])
        sw     = float(spec["search_win_s"])
        mask   = all_cl == uid
        spks   = np.sort(all_spikes_s[mask])
        ev_idx, eph_times, jitter = match_unit(spks, event_times, lat, sw)
        pct = 100 * len(ev_idx) / len(event_times)
        print(f"  Unit {uid:>5}  lat={lat*1000:>6.1f}ms  "
              f"matched={len(ev_idx):>4} / {len(event_times)}  "
              f"({pct:.1f}%)  med_jitter={np.median(jitter):.1f}ms")
        anchor_matches.append((uid, ev_idx, eph_times, jitter))

    # ── Fuse ────────────────────────────────────────────────────────────────
    print("\nFusing anchor estimates…")
    bhv_fused, eph_fused = fuse_anchor_estimates(anchor_matches, event_times)

    if len(bhv_fused) < 20:
        print(f"[ERROR] Only {len(bhv_fused)} fused pairs — check anchor specs.")
        sys.exit(1)

    # ── Outlier rejection + warp fit ────────────────────────────────────────
    print("\nRejecting outliers (3-sigma)…")
    keep = reject_outliers(bhv_fused, eph_fused)
    print(f"  Kept {keep.sum()} / {len(bhv_fused)} pairs")

    bhv_clean = bhv_fused[keep]
    eph_clean = eph_fused[keep]
    stretch, offset, rmse = fit_warp(bhv_clean, eph_clean)

    print(f"\nFused warp fit:")
    print(f"  stretch = {stretch:.8f}")
    print(f"  offset  = {offset:.4f}s  ({offset*1000:.1f}ms)")
    print(f"  RMSE    = {rmse*1000:.2f}ms  ({keep.sum()} pairs)")
    print(f"  Coverage: {100*len(bhv_fused)/len(event_times):.1f}% of events")

    quality = "EXCELLENT" if rmse < 0.005 else \
              "GOOD"      if rmse < 0.020 else \
              "OK"        if rmse < 0.050 else "POOR"
    print(f"  Quality : {quality}")

    # ── Save sync.json ──────────────────────────────────────────────────────
    sync_data = {
        "warp": {
            "stretch": stretch,
            "offset":  offset,
            "rmse":    rmse,
        },
        "events": [
            {"bhv_time": float(b), "ephys_time": float(e)}
            for b, e in zip(bhv_clean, eph_clean)
        ],
        "metadata": {
            "method":         "multi_anchor_fusion",
            "n_anchors":      len(anchor_specs),
            "anchor_units":   [int(a["unit"]) for a in anchor_specs],
            "anchor_latencies_ms": [round(a["latency_s"]*1000, 1) for a in anchor_specs],
            "n_fused_pairs":  int(keep.sum()),
            "n_events_total": int(len(event_times)),
            "coverage_pct":   round(100 * len(bhv_fused) / len(event_times), 1),
            "target_code":    args.target_code,
            # For compatibility with downstream scripts that expect anchor_unit:
            "anchor_unit":    int(anchor_specs[0]["unit"]),
            "peak_lat_ms":    round(anchor_specs[0]["latency_s"] * 1000, 1),
        }
    }

    sync_path = out_dir / "spike_sync_fused.json"
    with open(str(sync_path), "w") as f:
        json.dump(sync_data, f, indent=2)
    print(f"\n  sync.json → {sync_path}")

    # ── Diagnostic plot ─────────────────────────────────────────────────────
    diag_path = out_dir / "fusion_diagnostic.png"
    plot_fusion_diagnostic(
        anchor_matches, bhv_clean, eph_clean,
        stretch, offset, rmse, event_times,
        anchor_specs, diag_path
    )

    # ── Comparison summary ──────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  MULTI-ANCHOR FUSION COMPLETE')
    print(f'  Anchors   : {len(anchor_specs)} units')
    print(f'  Pairs     : {keep.sum()} (from {len(bhv_fused)} fused events)')
    print(f'  Coverage  : {100*len(bhv_fused)/len(event_times):.1f}%  '
          f'({len(event_times) - len(bhv_fused)} events missed)')
    print(f'  RMSE      : {rmse*1000:.2f}ms   [{quality}]')
    print(f'  Output    : {sync_path}')
    print(f'{"="*60}\n')


if __name__ == "__main__":
    main()
