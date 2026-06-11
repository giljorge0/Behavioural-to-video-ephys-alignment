#!/usr/bin/env python3
"""
anchor_unit_hunter.py
=====================

Hunts for "anchor units" — single well-isolated neurons that fire a
razor-sharp burst at a consistent time relative to a behavioral event.
Such a unit can serve as a clock anchor even when hardware sync has failed.

Inputs
------
Kilosort4 / Phy directory:
  spike_times.npy        raw sample indices (int64)
  spike_clusters.npy     cluster ID per spike (int64)
  cluster_group.tsv      cluster_id | group  (keep 'good')
  cluster_Extragood.tsv  cluster_id | Extragood  (keep 'y')  [optional]

Behavior log (GlobalLogInt*.csv):
  col-0 = ts (seconds)
  col-1 = event code (int)

Usage
-----
python anchor_unit_hunter.py \
    --ks-dir  "path/to/kilosort_output" \
    --bhv     "path/to/GlobalLogInt*.csv" \
    --target-code 11 \
    --window  0.5 \
    --bin-size 0.010 \
    --out     "anchor_unit_hunter_output"
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
AP_SR = 30_000.0  # Neuropixels AP sample rate


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def load_good_units(ks_dir):
    """Return set of cluster IDs labelled 'good' in cluster_group.tsv."""
    path = Path(ks_dir) / "cluster_group.tsv"
    if not path.exists():
        sys.exit(f"[ERROR] cluster_group.tsv not found in {ks_dir}")

    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip().str.lower()

    if "cluster_id" not in df.columns:
        sys.exit("[ERROR] cluster_group.tsv must contain column: cluster_id")

    group_col = "group" if "group" in df.columns else "kslabel"
    good = set(
    df.loc[df[group_col].astype(str).str.strip().str.lower() == "good", "cluster_id"]
    .astype(int)
    .values
)
    print(f"  cluster_group.tsv  → {len(good)} good units")
    return good


def load_extragood_units(ks_dir):
    """Return set of cluster IDs marked 'y' in cluster_Extragood.tsv (optional)."""
    path = Path(ks_dir) / "cluster_Extragood.tsv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = df.columns.str.strip().str.lower()

        if "cluster_id" not in df.columns:
            return None

        extra_col = [c for c in df.columns if "extragood" in c]
        if not extra_col:
            return None

        eg = set(
            df.loc[df[extra_col[0]].astype(str).str.strip().str.lower() == "y", "cluster_id"]
            .astype(int)
            .values
        )
        print(f"  cluster_Extragood.tsv → {len(eg)} extragood units")
        return eg
    except Exception as e:
        print(f"  [WARN] Could not parse cluster_Extragood.tsv: {e}")
        return None


def load_spikes(ks_dir, good_units):
    """Load spike_times / spike_clusters, filter to good units only."""
    st_path = Path(ks_dir) / "spike_times.npy"
    sc_path = Path(ks_dir) / "spike_clusters.npy"

    for p in (st_path, sc_path):
        if not p.exists():
            sys.exit(f"[ERROR] {p.name} not found in {ks_dir}")

    st = np.load(str(st_path)).ravel().astype(np.int64)
    sc = np.load(str(sc_path)).ravel().astype(np.int64)

    if len(st) != len(sc):
        sys.exit("[ERROR] spike_times.npy and spike_clusters.npy have different lengths")

    st_s = st / AP_SR  # convert to seconds
    mask = np.isin(sc, list(good_units))
    return st_s[mask], sc[mask]


def load_behavior(bhv_path, target_code):
    """Load CSV, zero-anchor timestamps, return event times for target_code."""
    df = pd.read_csv(str(bhv_path), header=None, names=["ts", "code"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        sys.exit("[ERROR] Behavior file is empty or could not be parsed.")

    df["ts"] -= df["ts"].iloc[0]  # zero-anchor
    events = df.loc[df["code"] == target_code, "ts"].values.astype(np.float64)

    if len(events) == 0:
        sys.exit(f"[ERROR] No events found with code={target_code}")

    print(
        f"  behavior log       → {len(events)} events with code={target_code} "
        f"(t={events.min():.1f}–{events.max():.1f}s)"
    )
    return events


# -----------------------------------------------------------------------------
# PSTH engine
# -----------------------------------------------------------------------------
def compute_psth(unit_spikes_s, event_times, window, bin_size):
    """
    Parameters
    ----------
    unit_spikes_s : sorted float64 array of spike times (seconds)
    event_times   : float64 array of event times (seconds)
    window        : float — half-window size (seconds), produces [-w, +w]
    bin_size      : float — bin width (seconds)

    Returns
    -------
    fr        : float array, firing rate per bin (Hz)
    bin_edges : float array (len(fr)+1)
    raster    : list of 1-D float arrays, one per trial (spike offsets)
    """
    bins = np.arange(-window, window + bin_size, bin_size)
    n_trials = len(event_times)
    counts = np.zeros(len(bins) - 1, dtype=np.float64)
    raster = []

    for et in event_times:
        lo = et - window
        hi = et + window
        i0 = np.searchsorted(unit_spikes_s, lo)
        i1 = np.searchsorted(unit_spikes_s, hi)
        offsets = unit_spikes_s[i0:i1] - et
        raster.append(offsets)
        c, _ = np.histogram(offsets, bins=bins)
        counts += c

    fr = counts / (n_trials * bin_size)  # spikes / s
    return fr, bins, raster


def sharpness_score(fr, bins, baseline_end=-0.1):
    """
    Peak firing rate in the peri-event window minus mean baseline.

    baseline_end : upper bound (s) of the pre-event baseline period.

    Returns (score, peak_fr, baseline_fr, peak_latency_s)
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    base_mask = bin_centers < baseline_end
    post_mask = bin_centers >= baseline_end

    baseline_fr = float(np.mean(fr[base_mask])) if base_mask.any() else 0.0
    peak_fr = float(np.max(fr[post_mask])) if post_mask.any() else 0.0
    peak_lat = (
        float(bin_centers[post_mask][np.argmax(fr[post_mask])])
        if post_mask.any()
        else 0.0
    )
    score = peak_fr - baseline_fr
    return score, peak_fr, baseline_fr, peak_lat


def trial_reliability(raster, peak_lat, half_width=0.05):
    """Fraction of trials that have ≥1 spike within ±half_width of peak_lat."""
    n_trials = len(raster)
    if n_trials == 0:
        return 0.0
    hits = sum(
        1
        for spk in raster
        if np.any(np.abs(spk - peak_lat) <= half_width)
    )
    return hits / n_trials


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_top_units(top_records, event_times, all_spikes_s, all_clusters,
                   window, bin_size, out_path, target_code):
    """
    For each top unit: raster (top) + PSTH (bottom) in a 2-row panel.
    Produces a clean multi-panel figure.
    """
    n = len(top_records)
    fig = plt.figure(figsize=(14, 3.2 * n), constrained_layout=True)
    fig.suptitle(
        f"Top {n} Anchor Units  |  Event code={target_code}  "
        f"|  window ±{window:.2f}s  |  bin={bin_size*1000:.0f}ms",
        fontsize=13,
        fontweight="bold",
    )

    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)
    bin_centers = None

    for row, rec in enumerate(top_records):
        uid = rec["uid"]
        fr = rec["fr"]
        bins = rec["bins"]
        raster = rec["raster"]
        score = rec["score"]
        pfr = rec["peak_fr"]
        bfr = rec["baseline_fr"]
        plat = rec["peak_lat"]
        rel = rec["reliability"]
        is_eg = rec.get("extragood", False)

        if bin_centers is None:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row],
            height_ratios=[1.6, 1], hspace=0.08
        )
        ax_rast = fig.add_subplot(inner[0])
        ax_psth = fig.add_subplot(inner[1], sharex=ax_rast)

        # Raster
        for trial_i, spk in enumerate(raster):
            ax_rast.vlines(
                spk, trial_i + 0.5, trial_i + 1.5,
                color="#2b6cb0", linewidth=0.6, alpha=0.7
            )
        ax_rast.axvline(0, color="crimson", ls="--", lw=1.5)
        ax_rast.set_xlim(-window, window)
        ax_rast.set_ylim(0.5, len(raster) + 0.5)
        ax_rast.set_yticks([1, len(raster)])
        ax_rast.set_yticklabels(["1", str(len(raster))], fontsize=8)
        ax_rast.set_ylabel("Trial", fontsize=8)
        ax_rast.tick_params(labelbottom=False)

        eg_tag = "  ★ ExtRaGood" if is_eg else ""
        ax_rast.set_title(
            f"Unit {uid}{eg_tag}   "
            f"score={score:.1f} Hz   "
            f"peak={pfr:.1f} Hz @ {plat*1000:.0f}ms   "
            f"base={bfr:.1f} Hz   "
            f"reliability={rel:.0%}",
            fontsize=9,
            loc="left",
            pad=3,
        )

        # PSTH
        ax_psth.bar(
            bin_centers,
            fr,
            width=bin_size * 0.9,
            color="#4a90d9",
            edgecolor="none",
            alpha=0.85,
        )
        ax_psth.axvline(0, color="crimson", ls="--", lw=1.5)
        ax_psth.axhline(bfr, color="#888", ls=":", lw=1, label=f"baseline {bfr:.1f} Hz")
        ax_psth.set_ylabel("FR (Hz)", fontsize=8)
        ax_psth.set_xlabel("Time from event (s)", fontsize=8)
        ax_psth.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax_psth.set_xlim(-window, window)
        ax_psth.tick_params(labelsize=8)

    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Hunt for anchor units in Kilosort4/Phy spike data."
    )
    ap.add_argument("--ks-dir", required=True, help="Kilosort / Phy output directory")
    ap.add_argument("--bhv", required=True, help="Behavior CSV (GlobalLogInt*.csv)")
    ap.add_argument(
        "--target-code",
        type=int,
        default=11,
        help="Behavioral event code to align to (default 11 = Pull)",
    )
    ap.add_argument(
        "--window",
        type=float,
        default=0.5,
        help="Half-window in seconds (default 0.5)",
    )
    ap.add_argument(
        "--bin-size",
        type=float,
        default=0.010,
        help="PSTH bin size in seconds (default 0.010 = 10 ms)",
    )
    ap.add_argument(
        "--min-spikes",
        type=int,
        default=50,
        help="Minimum total spikes to include a unit (default 50)",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Units in leaderboard (default 20)",
    )
    ap.add_argument(
        "--plot-n",
        type=int,
        default=5,
        help="Units in saved figure (default 5)",
    )
    ap.add_argument(
        "--out",
        default="anchor_unit_hunter_output",
        help="Output directory for plots",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f'\n{"="*65}')
    print(" anchor_unit_hunter.py")
    print(f" code={args.target_code}  window=±{args.window}s  bin={args.bin_size*1000:.0f}ms")
    print(f'{"="*65}\n')

    # Load
    print("Loading Phy outputs...")
    good_units = load_good_units(args.ks_dir)
    extragood = load_extragood_units(args.ks_dir)  # may be None
    all_spikes_s, all_clusters = load_spikes(args.ks_dir, good_units)
    print(
        f"  spike_times.npy    → {len(all_spikes_s):,} spikes "
        f"({len(good_units)} good units)"
    )

    print("\nLoading behavior...")
    event_times = load_behavior(args.bhv, args.target_code)
    if len(event_times) < 3:
        sys.exit("[ERROR] Fewer than 3 events found — check --target-code.")

    session_dur = float(all_spikes_s.max()) if len(all_spikes_s) else 1.0

    # Per-unit PSTH & scoring
    print(f"\nComputing PSTHs for {len(good_units)} good units...")
    records = []
    unit_list = sorted(good_units)

    for i, uid in enumerate(unit_list):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(unit_list)}...", flush=True)

        mask = all_clusters == uid
        n_spikes = int(mask.sum())
        if n_spikes < args.min_spikes:
            continue

        unit_spikes = np.sort(all_spikes_s[mask])
        mean_fr_global = n_spikes / session_dur  # global mean FR (Hz)

        fr, bins, raster = compute_psth(
            unit_spikes, event_times, args.window, args.bin_size
        )
        score, peak_fr, baseline_fr, peak_lat = sharpness_score(fr, bins)
        rel = trial_reliability(raster, peak_lat, half_width=args.bin_size * 3)

        records.append(
            dict(
                uid=uid,
                n_spikes=n_spikes,
                mean_fr=round(mean_fr_global, 2),
                baseline_fr=round(baseline_fr, 2),
                peak_fr=round(peak_fr, 2),
                score=round(score, 2),
                peak_lat=round(peak_lat, 4),
                reliability=round(rel, 3),
                extragood=(uid in extragood) if extragood else False,
                fr=fr,
                bins=bins,
                raster=raster,
            )
        )

    if not records:
        sys.exit("[ERROR] No units passed the minimum-spike filter.")

    records.sort(key=lambda r: r["score"], reverse=True)
    print(f'\n  {len(records)} units scored  (best score={records[0]["score"]:.1f} Hz)\n')

    # Console leaderboard
    top_n = min(args.top_n, len(records))
    hdr_w = 80
    print("=" * hdr_w)
    print(f" TOP {top_n} ANCHOR UNITS  (sorted by Sharpness Score = Peak − Baseline)")
    print("=" * hdr_w)
    print(
        f"{'Rk':<4} {'Unit':>6} {'Spikes':>7} "
        f"{'Base(Hz)':>9} {'Peak(Hz)':>9} {'Score(Hz)':>10} "
        f"{'PeakLat(ms)':>12} {'Reliability':>12} {'★':>3}"
    )
    print("-" * hdr_w)

    for i, r in enumerate(records[:top_n], 1):
        eg = "★" if r["extragood"] else ""
        print(
            f"{i:<4} {r['uid']:>6} {r['n_spikes']:>7,} "
            f"{r['baseline_fr']:>9.1f} {r['peak_fr']:>9.1f} "
            f"{r['score']:>10.1f} "
            f"{r['peak_lat']*1000:>11.1f}  "
            f"{r['reliability']:>11.1%}  {eg:>3}"
        )

    print("=" * hdr_w)
    print("\n  Score = Peak FR (Hz) − Baseline FR (Hz)")
    print("  Reliability = fraction of trials with ≥1 spike near peak latency")
    print("  ★ = marked ExtRaGood in Phy\n")

    # Save figure
    plot_n = min(args.plot_n, len(records))
    out_fig = Path(args.out) / "top_anchor_units.png"
    print(f"Generating figure for top {plot_n} units...")
    plot_top_units(
        records[:plot_n],
        event_times,
        all_spikes_s,
        all_clusters,
        args.window,
        args.bin_size,
        out_fig,
        args.target_code,
    )

    # Per-unit CSV summary
    csv_path = Path(args.out) / "anchor_unit_scores.csv"
    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("fr", "bins", "raster")} for r in records]
    )
    summary.to_csv(str(csv_path), index=False)
    print(f"  Saved: {csv_path}")

    # Best unit summary
    best = records[0]
    print(f'\n{"─"*65}')
    print(f' BEST ANCHOR CANDIDATE: Unit {best["uid"]}')
    print(f'  Sharpness  : {best["score"]:.1f} Hz above baseline')
    print(f'  Peak FR    : {best["peak_fr"]:.1f} Hz  at  {best["peak_lat"]*1000:.1f} ms')
    print(f'  Baseline FR: {best["baseline_fr"]:.1f} Hz')
    print(f'  Reliability: {best["reliability"]:.0%} of trials have a spike at peak')
    print(f'  ExtRaGood  : {"yes ★" if best["extragood"] else "no"}')

    if best["reliability"] > 0.7 and best["score"] > 10:
        print("\n  ✓ HIGH CONFIDENCE — this unit looks like a usable anchor.")
    elif best["reliability"] > 0.5 and best["score"] > 5:
        print("\n  ~ MODERATE — may work as an anchor; inspect the raster plot.")
    else:
        print("\n  ✗ LOW — no strong anchor unit found for this event code.")
        print("    Try --target-code with a different event (e.g. 2, 31, -11).")

    print(f'{"─"*65}\n')
    print(f"All outputs → {Path(args.out).resolve()}\n")


if __name__ == "__main__":
    main()