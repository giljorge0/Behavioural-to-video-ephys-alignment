#!/usr/bin/env python3
"""
plot_neural_circuits.py  (v2 — raster fix + ISI + sort-by-latency)
===================================================================
Generates raster + Z-scored PSTH plots for all good units in a session,
aligned to a behavioral event using a spike_sync.json warp.

Each unit figure:
  - Top panel   : spike raster (one row per trial, tick marks via vlines)
  - Bottom panel: Z-scored PSTH with threshold lines

Units are auto-classified:
  MOTOR       — peak Z > thr AND peak latency < 0 ms (fires before event)
  REWARD      — peak Z > thr AND peak latency > 0 ms (fires after event)
  SUPPRESSED  — trough Z < −thr after event
  UNRESPONSIVE — nothing else fires reliably

Output
------
  <out>/unit_plots/Unit_<ID>_code<N>.png
  <out>/summary.csv
  <out>/population_heatmap.png
  <out>/classification_counts.txt
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
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d

AP_SR       = 30_000.0
PSTH_BIN_S  = 0.020    # 20 ms bins
SMOOTH_BINS = 1.5      # Gaussian sigma in bins (was 2 — slightly sharper)
Z_THRESHOLD = 2.5

CLASS_COLORS = {
    "MOTOR":        "#d62728",
    "REWARD":       "#1f77b4",
    "SUPPRESSED":   "#9467bd",
    "UNRESPONSIVE": "#7f7f7f",
}


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


def load_extragood_units(ks_dir):
    path = Path(ks_dir) / "cluster_Extragood.tsv"
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = df.columns.str.strip().str.lower()
        col = [c for c in df.columns if "extragood" in c]
        if not col:
            return set()
        return set(df.loc[df[col[0]].str.strip().str.lower() == "y",
                          "cluster_id"].astype(int).values)
    except Exception:
        return set()


def load_spikes(ks_dir, good_units):
    st = np.load(str(Path(ks_dir) / "spike_times.npy")).ravel().astype(np.int64)
    sc = np.load(str(Path(ks_dir) / "spike_clusters.npy")).ravel().astype(np.int64)
    mask = np.isin(sc, list(good_units))
    return st[mask] / AP_SR, sc[mask]


def load_behavior(bhv_path, target_code):
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


def load_sync(sync_path):
    with open(str(sync_path)) as f:
        s = json.load(f)
    return s["warp"]["stretch"], s["warp"]["offset"], s["warp"]["rmse"]


# ── PSTH engine ───────────────────────────────────────────────────────────────

def compute_psth_and_raster(unit_spikes_s, event_times_ephys,
                             window_s, bin_s=PSTH_BIN_S):
    bins     = np.arange(-window_s, window_s + bin_s, bin_s)
    n_bins   = len(bins) - 1
    n_trials = len(event_times_ephys)
    counts   = np.zeros(n_bins)
    raster   = []
    trial_counts = np.zeros((n_trials, n_bins))

    for i, et in enumerate(event_times_ephys):
        i0 = np.searchsorted(unit_spikes_s, et - window_s)
        i1 = np.searchsorted(unit_spikes_s, et + window_s)
        offsets = unit_spikes_s[i0:i1] - et
        raster.append(offsets)
        c, _ = np.histogram(offsets, bins=bins)
        counts += c
        trial_counts[i] = c

    fr  = counts / (n_trials * bin_s)

    centers   = 0.5 * (bins[:-1] + bins[1:])
    base_mask = centers < -0.1     # baseline: everything before −100ms

    baseline_fr = float(np.mean(fr[base_mask]))
    baseline_sd = float(np.std(fr[base_mask]))
    if baseline_sd < 0.1:
        baseline_sd = max(0.1, baseline_fr * 0.05 + 0.1)

    fr_z = (fr - baseline_fr) / baseline_sd

    return bins, fr, fr_z, baseline_fr, baseline_sd, raster


def classify_unit(fr_z, bins, threshold=Z_THRESHOLD):
    centers = 0.5 * (bins[:-1] + bins[1:])
    post = centers >= 0.0
    pre  = (centers >= -0.5) & (centers < 0.0)

    peak_z_post  = float(np.max(fr_z[post]))  if post.any()  else 0.0
    trough_z     = float(np.min(fr_z[post]))  if post.any()  else 0.0
    peak_z_pre   = float(np.max(fr_z[pre]))   if pre.any()   else 0.0
    peak_lat_post = float(centers[post][np.argmax(fr_z[post])]) if post.any() else 0.0
    peak_lat_pre  = float(centers[pre][np.argmax(fr_z[pre])])   if pre.any()  else 0.0

    if peak_z_pre >= threshold and peak_z_pre > peak_z_post:
        return "MOTOR",       peak_z_pre,  peak_lat_pre
    elif peak_z_post >= threshold:
        return "REWARD",      peak_z_post, peak_lat_post
    elif trough_z <= -threshold:
        return "SUPPRESSED",  trough_z,    peak_lat_post
    else:
        return "UNRESPONSIVE", max(peak_z_post, peak_z_pre), peak_lat_post


# ── Single unit plot ──────────────────────────────────────────────────────────

def plot_unit(uid, bins, fr, fr_z, baseline_fr, raster, event_times_ephys,
              unit_class, peak_z, peak_lat, extragood,
              window_s, target_code, out_path, sort_by_latency=False):

    n_trials = len(raster)
    centers  = 0.5 * (bins[:-1] + bins[1:])
    fr_z_sm  = gaussian_filter1d(fr_z.astype(float), sigma=SMOOTH_BINS)
    color    = CLASS_COLORS[unit_class]
    eg_str   = " ★" if extragood else ""

    # Sort raster rows by first post-event spike
    if sort_by_latency:
        first_spk = []
        for offsets in raster:
            post = offsets[(offsets >= 0) & (offsets < window_s)]
            first_spk.append(float(post.min()) if len(post) else np.nan)
        order = np.argsort(first_spk)
    else:
        order = np.arange(n_trials)

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.2], hspace=0.08)
    ax_raster = fig.add_subplot(gs[0])
    ax_psth   = fig.add_subplot(gs[1], sharex=ax_raster)

    # ── Raster  (FIX: use vlines instead of scatter with invisible markers) ──
    for row_i, trial_i in enumerate(order):
        spks = raster[trial_i]
        if len(spks) > 0:
            y_lo = row_i + 0.55
            y_hi = row_i + 1.45
            # vlines is reliable regardless of marker size / linewidth quirks
            ax_raster.vlines(spks, y_lo, y_hi,
                             linewidth=0.35, color="black", alpha=0.75)

    ax_raster.axvline(0, color="red", lw=1.0, ls="--", alpha=0.8)
    ax_raster.set_ylabel("Trial", fontsize=11)
    ax_raster.set_xlim(-window_s, window_s)
    ax_raster.set_ylim(0, n_trials + 1)
    ax_raster.tick_params(labelbottom=False)
    sort_str = " (sorted by 1st spike)" if sort_by_latency else ""
    ax_raster.set_title(
        f"Unit {uid}{eg_str}  |  code={target_code}  |  {unit_class}  "
        f"|  peak Z={peak_z:.1f} @ {peak_lat*1000:.0f}ms\n"
        f"baseline={baseline_fr:.1f} Hz  |  n={n_trials} trials{sort_str}",
        fontsize=10, color=color, fontweight="bold"
    )

    # ── Z-scored PSTH ──────────────────────────────────────────────────────
    ax_psth.fill_between(centers, 0, fr_z_sm, where=fr_z_sm >= 0,
                         color=color, alpha=0.45, interpolate=True)
    ax_psth.fill_between(centers, 0, fr_z_sm, where=fr_z_sm < 0,
                         color=color, alpha=0.25, interpolate=True)
    ax_psth.plot(centers, fr_z_sm, color=color, lw=1.4)
    ax_psth.axvline(0, color="red", lw=1.0, ls="--", alpha=0.8)
    ax_psth.axhline(0, color="black", lw=0.6)
    ax_psth.axhline( Z_THRESHOLD, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_psth.axhline(-Z_THRESHOLD, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_psth.set_xlabel("Time from event (s)", fontsize=11)
    ax_psth.set_ylabel("Z-score (SD)", fontsize=11)
    ax_psth.set_xlim(-window_s, window_s)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Population heatmap ────────────────────────────────────────────────────────

def plot_population_heatmap(summary_df, all_fr_z, bins, window_s,
                             out_path, target_code):
    df_sorted = summary_df.sort_values("peak_lat_s")
    uids = df_sorted["uid"].values

    mat = np.zeros((len(uids), len(bins) - 1))
    for i, uid in enumerate(uids):
        if uid in all_fr_z:
            mat[i] = gaussian_filter1d(all_fr_z[uid].astype(float),
                                       sigma=SMOOTH_BINS)

    centers = 0.5 * (bins[:-1] + bins[1:])
    vmax    = min(8.0, float(np.nanpercentile(np.abs(mat), 98)))

    fig, ax = plt.subplots(figsize=(12, max(4, len(uids) * 0.045 + 2)))
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[-window_s, window_s, 0, len(uids)],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axvline(0, color="white", lw=1.0, ls="--", alpha=0.8)

    prev_y = 0
    for cls in ["MOTOR", "REWARD", "SUPPRESSED", "UNRESPONSIVE"]:
        n = (df_sorted["unit_class"] == cls).sum()
        if n > 0:
            ax.axhline(prev_y + n, color="white", lw=0.5, alpha=0.4)
            ax.text(window_s * 0.98, prev_y + n / 2, cls,
                    va="center", ha="right", fontsize=7,
                    color=CLASS_COLORS[cls], fontweight="bold")
            prev_y += n

    plt.colorbar(im, ax=ax, label="Z-score (SD above baseline)", shrink=0.6)
    ax.set_xlabel("Time from event (s)", fontsize=12)
    ax.set_ylabel(f"Units (n={len(uids)}, sorted by peak latency)", fontsize=12)
    ax.set_title(f"Population activity  |  code={target_code}  |  "
                 f"sorted by peak latency", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Population heatmap → {Path(out_path).name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Raster + Z-PSTH for all good units, aligned via spike_sync.json")
    parser.add_argument("--ks-dir",          required=True)
    parser.add_argument("--bhv",             required=True)
    parser.add_argument("--sync",            required=True)
    parser.add_argument("--target-code",     type=int,   default=2)
    parser.add_argument("--window",          type=float, default=1.0)
    parser.add_argument("--bin-size",        type=float, default=PSTH_BIN_S)
    parser.add_argument("--z-threshold",     type=float, default=Z_THRESHOLD)
    parser.add_argument("--min-spikes",      type=int,   default=200)
    parser.add_argument("--sort-by-latency", action="store_true")
    parser.add_argument("--pdf",             action="store_true")
    parser.add_argument("--only-responsive", action="store_true")
    parser.add_argument("--out",             required=True)
    args = parser.parse_args()

    out_dir   = Path(args.out)
    plots_dir = out_dir / "unit_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  plot_neural_circuits.py  v2')
    print(f'  code={args.target_code}  window=±{args.window}s  '
          f'bin={args.bin_size*1000:.0f}ms  Z_thr={args.z_threshold}')
    print(f'{"="*60}\n')

    print("Loading Phy outputs…")
    good_units             = load_good_units(args.ks_dir)
    extragood              = load_extragood_units(args.ks_dir)
    all_spikes_s, all_cl   = load_spikes(args.ks_dir, good_units)
    print(f"  {len(all_spikes_s):,} spikes")

    print("\nLoading behavior…")
    event_times_bhv = load_behavior(args.bhv, args.target_code)

    print("\nLoading sync…")
    stretch, offset, rmse = load_sync(args.sync)
    print(f"  stretch={stretch:.8f}  offset={offset*1000:.1f}ms  "
          f"RMSE={rmse*1000:.1f}ms")
    event_times_ephys = stretch * event_times_bhv + offset
    print(f"  {len(event_times_ephys)} events warped to ephys time")

    print(f"\nComputing PSTHs for {len(good_units)} good units…")
    records  = []
    all_fr_z = {}
    bins_ref = None

    for i, uid in enumerate(sorted(good_units), 1):
        if i % 50 == 0:
            print(f"  {i}/{len(good_units)}…")

        mask = all_cl == uid
        spks = np.sort(all_spikes_s[mask])
        if len(spks) < args.min_spikes:
            continue

        bins, fr, fr_z, base_fr, base_sd, raster = compute_psth_and_raster(
            spks, event_times_ephys, args.window, args.bin_size)

        if bins_ref is None:
            bins_ref = bins

        unit_class, peak_z, peak_lat = classify_unit(fr_z, bins, args.z_threshold)
        all_fr_z[uid] = fr_z

        eg  = uid in extragood
        records.append(dict(uid=uid, n_spikes=len(spks),
                            baseline_fr=round(base_fr, 2),
                            baseline_sd=round(base_sd, 2),
                            peak_z=round(peak_z, 2),
                            peak_lat_s=round(peak_lat, 3),
                            unit_class=unit_class,
                            extragood=eg))

        if args.only_responsive and unit_class == "UNRESPONSIVE":
            continue

        out_png = plots_dir / f"Unit_{uid:04d}_code{args.target_code}.png"
        plot_unit(uid, bins, fr, fr_z, base_fr, raster, event_times_ephys,
                  unit_class=unit_class, peak_z=peak_z, peak_lat=peak_lat,
                  extragood=eg, window_s=args.window,
                  target_code=args.target_code, out_path=out_png,
                  sort_by_latency=args.sort_by_latency)

    print(f"\n  {len(records)} units processed")

    df = pd.DataFrame(records)
    df.to_csv(str(out_dir / "summary.csv"), index=False)
    print(f"  Summary → summary.csv")

    counts_path = out_dir / "classification_counts.txt"
    with open(str(counts_path), "w") as f:
        f.write(f"Classification summary  |  code={args.target_code}  "
                f"|  Z_threshold={args.z_threshold}\n")
        f.write(f"Total units processed: {len(records)}\n\n")
        for cls in ["MOTOR", "REWARD", "SUPPRESSED", "UNRESPONSIVE"]:
            n    = (df["unit_class"] == cls).sum()
            pct  = 100 * n / len(records) if records else 0
            eg_n = ((df["unit_class"] == cls) & df["extragood"]).sum()
            f.write(f"  {cls:<14}: {n:>4}  ({pct:4.1f}%)  [{eg_n} ExtRaGood]\n")
        f.write(f"\nSync RMSE used: {rmse*1000:.2f}ms\n")
    with open(str(counts_path)) as f:
        print("\n" + f.read())

    if bins_ref is not None and len(records) > 1:
        plot_population_heatmap(df, all_fr_z, bins_ref, args.window,
                                out_dir / "population_heatmap.png",
                                args.target_code)

    if args.pdf:
        pdf_path   = out_dir / f"all_units_code{args.target_code}.pdf"
        png_files  = sorted(plots_dir.glob(f"Unit_*_code{args.target_code}.png"))
        print(f"  Compiling PDF ({len(png_files)} pages)…")
        with PdfPages(str(pdf_path)) as pdf:
            for png in png_files:
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(plt.imread(str(png)))
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"  PDF → {pdf_path.name}")

    print(f'\n{"="*60}')
    print(f'  DONE  →  {out_dir}')
    print(f'{"="*60}\n')


if __name__ == "__main__":
    main()
