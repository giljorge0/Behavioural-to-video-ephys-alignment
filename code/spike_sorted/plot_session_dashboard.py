#!/usr/bin/env python3
"""
plot_session_dashboard.py
=========================
Generates a comprehensive multi-panel session report from:
  - spike_sync.json      (warp parameters + matched event pairs)
  - summary.csv          (per-unit classification from plot_neural_circuits.py)
  - GlobalLogInt*.csv    (behavioral event log)
  - Kilosort directory   (spike data for the anchor unit specifically)

Outputs (all in --out folder)
------------------------------
  1_warp_analysis.png       — scatter, residuals, error distribution
  2_classification.png      — pie chart + firing rate violin per class
  3_latency_map.png         — peak latency distribution + latency-sorted heatmap
  4_anchor_unit.png         — anchor unit raster + PSTH + ISI distribution
  5_session_timeline.png    — when trials occurred, reaction-time drift
  dashboard_summary.png     — single 2×3 overview combining key panels

Usage
-----
python plot_session_dashboard.py \
    --sync    "outputs/Ferrero_R1_imec0/spike_sync.json" \
    --summary "outputs/neural_circuits/Ferrero_R1_imec0_code11/summary.csv" \
    --bhv     "path/to/GlobalLogInt*.csv" \
    --ks-dir  "path/to/kilosort4" \
    --target-code 11 \
    --out     "outputs/neural_circuits/Ferrero_R1_imec0_code11/dashboard"
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
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

AP_SR = 30_000.0

CLASS_COLORS = {
    "MOTOR":        "#d62728",
    "REWARD":       "#1f77b4",
    "SUPPRESSED":   "#9467bd",
    "UNRESPONSIVE": "#7f7f7f",
}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_sync(sync_path):
    with open(str(sync_path)) as f:
        return json.load(f)


def load_behavior(bhv_path):
    df = pd.read_csv(str(bhv_path), header=None, names=["ts", "code"])
    df["ts"]   = pd.to_numeric(df["ts"],   errors="coerce")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df = df.dropna()
    sr_mask = df["code"] == 1000
    t0 = float(df["ts"][sr_mask].iloc[0]) if sr_mask.any() \
         else float(df["ts"].iloc[0])
    df["ts"] -= t0
    return df


def load_anchor_spikes(ks_dir, uid):
    st  = np.load(str(Path(ks_dir) / "spike_times.npy")).ravel().astype(np.int64)
    sc  = np.load(str(Path(ks_dir) / "spike_clusters.npy")).ravel().astype(np.int64)
    mask = sc == uid
    return np.sort(st[mask] / AP_SR)


# ── Panel 1: Warp analysis ────────────────────────────────────────────────────

def plot_warp_analysis(sync, out_path):
    """
    3-panel warp figure:
      Left  : behavior time vs ephys time scatter with fit line
      Middle: residuals over the session
      Right : error histogram
    """
    warp    = sync["warp"]
    stretch = warp["stretch"]
    offset  = warp["offset"]
    rmse    = warp["rmse"]
    meta    = sync.get("metadata", {})

    events   = sync.get("events", [])
    if not events:
        print("  [WARN] No events list in sync.json — skipping warp analysis")
        return

    bhv_t   = np.array([e["bhv_time"]   for e in events])
    eph_t   = np.array([e["ephys_time"] for e in events])
    pred_t  = stretch * bhv_t + offset
    resid   = (eph_t - pred_t) * 1000   # ms
    errors  = np.abs(resid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Warp Analysis  |  Unit {meta.get('anchor_unit','?')}  "
        f"|  latency {meta.get('peak_lat_ms','?')}ms  "
        f"|  n={meta.get('n_kept','?')} pairs  "
        f"|  RMSE = {rmse*1000:.2f}ms",
        fontsize=12, fontweight="bold"
    )

    # ── Left: scatter + fit ───────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(bhv_t, eph_t, s=14, alpha=0.55, color="steelblue",
               label=f"n = {len(bhv_t)} pairs", zorder=3)
    t_line = np.array([bhv_t.min(), bhv_t.max()])
    ax.plot(t_line, stretch * t_line + offset, "r-", lw=1.8,
            label=f"stretch = {stretch:.7f}\noffset = {offset*1000:.1f}ms")
    # Ideal 1:1 line (what it would look like with zero offset)
    mid_bhv = (bhv_t.min() + bhv_t.max()) / 2
    mid_eph = (eph_t.min() + eph_t.max()) / 2
    ax.set_xlabel("Behavior time (s)", fontsize=11)
    ax.set_ylabel("Ephys spike time (s)", fontsize=11)
    ax.set_title("Clock alignment", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── Middle: residuals over time ───────────────────────────────────────
    ax = axes[1]
    ax.scatter(bhv_t, resid, s=10, alpha=0.55, color="darkorange", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline( rmse * 1000, color="red", lw=1.0, ls="--",
               label=f"±RMSE = {rmse*1000:.1f}ms")
    ax.axhline(-rmse * 1000, color="red", lw=1.0, ls="--")

    # Running median to reveal slow drift
    if len(bhv_t) > 10:
        sort_idx = np.argsort(bhv_t)
        win = max(5, len(bhv_t) // 10)
        run_med = np.array([
            np.median(resid[sort_idx][max(0, j-win):j+win])
            for j in range(len(bhv_t))
        ])
        ax.plot(bhv_t[sort_idx], run_med, "b-", lw=1.5, alpha=0.7,
                label="Running median (drift)")

    ax.set_xlabel("Behavior time (s)", fontsize=11)
    ax.set_ylabel("Residual (ms)", fontsize=11)
    ax.set_title("Residuals over session", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Right: error histogram ────────────────────────────────────────────
    ax = axes[2]
    ax.hist(errors, bins=30, color="mediumseagreen", edgecolor="white", lw=0.5)
    ax.axvline(np.median(errors), color="red", lw=1.8,
               label=f"median = {np.median(errors):.1f}ms")
    ax.axvline(rmse * 1000, color="orange", lw=1.4, ls="--",
               label=f"RMSE = {rmse*1000:.1f}ms")
    ax.set_xlabel("|spike − predicted| (ms)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Matching error distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Warp analysis → {Path(out_path).name}")


# ── Panel 2: Classification summary ──────────────────────────────────────────

def plot_classification(summary_df, out_path):
    """
    Left : pie chart of unit classes (with ExtRaGood overlay)
    Right: violin plot of baseline firing rates per class
    """
    fig, (ax_pie, ax_vio) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Unit Classification Summary", fontsize=13, fontweight="bold")

    # ── Pie ───────────────────────────────────────────────────────────────
    classes = ["MOTOR", "REWARD", "SUPPRESSED", "UNRESPONSIVE"]
    counts  = [int((summary_df["unit_class"] == c).sum()) for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]
    total   = sum(counts)

    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=[f"{c}\n(n={n})" for c, n in zip(classes, counts)],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct*total/100))})",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for t in texts:
        t.set_fontsize(10)
    for at in autotexts:
        at.set_fontsize(8)

    # Show ExtRaGood breakdown in legend
    eg_counts = [int(((summary_df["unit_class"] == c) & summary_df["extragood"]).sum())
                 for c in classes]
    legend_patches = [
        Patch(color=colors[i], label=f"{classes[i]}: {eg_counts[i]} ExtRaGood")
        for i in range(len(classes))
    ]
    ax_pie.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), fontsize=9, ncol=2)
    ax_pie.set_title(f"Unit classification  (n={total} total)", fontsize=11)

    # ── Violin: baseline firing rate per class ────────────────────────────
    data_by_class = [
        summary_df.loc[summary_df["unit_class"] == c, "baseline_fr"].values
        for c in classes
    ]
    data_by_class = [d[~np.isnan(d)] for d in data_by_class]

    parts = ax_vio.violinplot(
        [d for d in data_by_class if len(d) > 0],
        positions=range(len(classes)),
        showmedians=True, showextrema=True
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
        pc.set_edgecolor(c)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")

    # Scatter jitter overlay
    for i, (d, c) in enumerate(zip(data_by_class, colors)):
        if len(d) > 0:
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(d))
            ax_vio.scatter(i + jitter, d, s=8, alpha=0.5, color=c, zorder=3)

    ax_vio.set_xticks(range(len(classes)))
    ax_vio.set_xticklabels(classes, fontsize=10)
    ax_vio.set_ylabel("Baseline firing rate (Hz)", fontsize=11)
    ax_vio.set_title("Baseline firing rate by class", fontsize=11)
    ax_vio.set_yscale("log")
    ax_vio.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Classification summary → {Path(out_path).name}")


# ── Panel 3: Latency map ──────────────────────────────────────────────────────

def plot_latency_map(summary_df, out_path, target_code):
    """
    Left : histogram of peak latencies per class (reveals population timing)
    Right: scatter of peak_Z vs peak_latency coloured by class
    """
    fig, (ax_hist, ax_scat) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Population Timing Map  |  code={target_code}",
                 fontsize=13, fontweight="bold")

    # ── Latency histogram stacked by class ───────────────────────────────
    bins = np.linspace(
        summary_df["peak_lat_s"].min() - 0.05,
        summary_df["peak_lat_s"].max() + 0.05,
        40
    )
    bottom = np.zeros(len(bins) - 1)
    for cls in ["MOTOR", "REWARD", "SUPPRESSED", "UNRESPONSIVE"]:
        sub = summary_df.loc[summary_df["unit_class"] == cls, "peak_lat_s"].values
        if len(sub) == 0:
            continue
        h, _ = np.histogram(sub, bins=bins)
        ax_hist.bar(0.5 * (bins[:-1] + bins[1:]), h, width=np.diff(bins),
                    bottom=bottom, color=CLASS_COLORS[cls], alpha=0.8,
                    label=cls, edgecolor="white", linewidth=0.3)
        bottom += h.astype(float)

    ax_hist.axvline(0, color="black", lw=1.5, ls="--", label="Event (t=0)")
    ax_hist.set_xlabel("Peak latency (s)", fontsize=11)
    ax_hist.set_ylabel("Unit count", fontsize=11)
    ax_hist.set_title("Peak latency distribution", fontsize=11)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.25)

    # ── Z vs latency scatter ──────────────────────────────────────────────
    for cls in ["MOTOR", "REWARD", "SUPPRESSED", "UNRESPONSIVE"]:
        sub = summary_df[summary_df["unit_class"] == cls]
        eg  = sub[sub["extragood"]]
        ax_scat.scatter(sub["peak_lat_s"], sub["peak_z"],
                        s=22, alpha=0.55, color=CLASS_COLORS[cls], label=cls)
        if len(eg) > 0:
            ax_scat.scatter(eg["peak_lat_s"], eg["peak_z"],
                            s=60, alpha=0.9, color=CLASS_COLORS[cls],
                            edgecolors="black", linewidths=1.5, marker="*",
                            label=f"{cls} ★ ExtRaGood")

    ax_scat.axvline(0, color="black", lw=1.0, ls="--")
    ax_scat.axhline(0, color="black", lw=0.5)
    ax_scat.set_xlabel("Peak latency (s)", fontsize=11)
    ax_scat.set_ylabel("Peak Z-score", fontsize=11)
    ax_scat.set_title("Z-score vs latency", fontsize=11)
    ax_scat.legend(fontsize=8, loc="upper right")
    ax_scat.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Latency map → {Path(out_path).name}")


# ── Panel 4: Anchor unit deep-dive ───────────────────────────────────────────

def plot_anchor_unit(sync, bhv_df, ks_dir, target_code, out_path):
    """
    Top    : spike raster aligned to reward events
    Middle : PSTH (firing rate and Z-score)
    Bottom : ISI distribution (shows isolation quality)
    """
    meta    = sync.get("metadata", {})
    uid     = meta.get("anchor_unit", None)
    lat_ms  = meta.get("peak_lat_ms", 0)
    stretch = sync["warp"]["stretch"]
    offset  = sync["warp"]["offset"]
    rmse    = sync["warp"]["rmse"]

    if uid is None or ks_dir is None:
        print("  [WARN] No anchor unit in sync.json — skipping anchor plot")
        return

    spikes_s = load_anchor_spikes(ks_dir, uid)
    print(f"  Anchor unit {uid}: {len(spikes_s):,} spikes  "
          f"latency={lat_ms:.0f}ms")

    events_bhv  = bhv_df.loc[bhv_df["code"] == target_code, "ts"].values
    events_eph  = stretch * events_bhv + offset
    window_s    = 1.0
    bin_s       = 0.010   # 10ms bins for anchor unit
    bins        = np.arange(-window_s, window_s + bin_s, bin_s)
    centers     = 0.5 * (bins[:-1] + bins[1:])
    n_trials    = len(events_eph)

    counts  = np.zeros(len(bins) - 1)
    raster  = []
    for et in events_eph:
        i0 = np.searchsorted(spikes_s, et - window_s)
        i1 = np.searchsorted(spikes_s, et + window_s)
        offsets = spikes_s[i0:i1] - et
        raster.append(offsets)
        c, _ = np.histogram(offsets, bins=bins)
        counts += c

    fr = counts / (n_trials * bin_s)
    base_mask   = centers < -0.1
    baseline_fr = float(np.mean(fr[base_mask]))
    baseline_sd = max(0.1, float(np.std(fr[base_mask])))
    fr_z        = (fr - baseline_fr) / baseline_sd
    fr_z_sm     = gaussian_filter1d(fr_z.astype(float), sigma=2.0)

    # ISI
    isi_ms = np.diff(spikes_s) * 1000.0
    isi_ms = isi_ms[isi_ms < 500]   # cap at 500ms for display

    fig = plt.figure(figsize=(12, 12))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.2, 1.0],
                             hspace=0.1, figure=fig)
    ax_rast = fig.add_subplot(gs[0])
    ax_psth = fig.add_subplot(gs[1], sharex=ax_rast)
    ax_isi  = fig.add_subplot(gs[2])

    fig.suptitle(
        f"Anchor Unit {uid}  |  code={target_code}  "
        f"|  latency={lat_ms:.0f}ms  |  RMSE={rmse*1000:.1f}ms\n"
        f"baseline={baseline_fr:.1f} Hz  |  n={n_trials} trials  "
        f"|  {len(spikes_s):,} total spikes",
        fontsize=11, fontweight="bold", color="#2ca02c"
    )

    # ── Raster ────────────────────────────────────────────────────────────
    for row_i, offsets in enumerate(raster):
        if len(offsets) > 0:
            ax_rast.vlines(offsets, row_i + 0.55, row_i + 1.45,
                           linewidth=0.35, color="black", alpha=0.75)

    ax_rast.axvline(0, color="red", lw=1.2, ls="--", alpha=0.9,
                    label=f"code={target_code}")
    ax_rast.axvline(lat_ms / 1000, color="#2ca02c", lw=1.0, ls="-",
                    alpha=0.8, label=f"latency {lat_ms:.0f}ms")
    ax_rast.set_ylabel("Trial", fontsize=11)
    ax_rast.set_ylim(0, n_trials + 1)
    ax_rast.set_xlim(-window_s, window_s)
    ax_rast.tick_params(labelbottom=False)
    ax_rast.legend(fontsize=9, loc="upper right")

    # ── PSTH ──────────────────────────────────────────────────────────────
    ax_psth.fill_between(centers, 0, fr_z_sm, where=fr_z_sm >= 0,
                         color="#2ca02c", alpha=0.45, interpolate=True)
    ax_psth.fill_between(centers, 0, fr_z_sm, where=fr_z_sm < 0,
                         color="#2ca02c", alpha=0.2, interpolate=True)
    ax_psth.plot(centers, fr_z_sm, color="#2ca02c", lw=1.5)
    ax_psth.axvline(0, color="red", lw=1.0, ls="--")
    ax_psth.axvline(lat_ms / 1000, color="#2ca02c", lw=1.0, ls="-", alpha=0.8)
    ax_psth.axhline(0, color="black", lw=0.6)
    ax_psth.axhline(2.0, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_psth.set_ylabel("Z-score (SD)", fontsize=11)
    ax_psth.set_xlim(-window_s, window_s)
    ax_psth.tick_params(labelbottom=False)

    # ── ISI ───────────────────────────────────────────────────────────────
    ax_isi.hist(isi_ms, bins=100, color="steelblue", edgecolor="none",
                alpha=0.8)
    # Refractory violation marker (< 2ms = contamination)
    n_refrac   = int(np.sum(isi_ms < 2.0))
    pct_refrac = 100 * n_refrac / len(isi_ms) if len(isi_ms) > 0 else 0
    ax_isi.axvline(2.0, color="red", lw=1.5, ls="--",
                   label=f"2ms refractory  ({n_refrac} violations = {pct_refrac:.2f}%)")
    ax_isi.set_xlabel("Inter-spike interval (ms)", fontsize=11)
    ax_isi.set_ylabel("Count", fontsize=11)
    ax_isi.set_title("ISI distribution (isolation quality)", fontsize=10)
    ax_isi.legend(fontsize=9)
    ax_isi.set_xlim(0, min(250, float(np.percentile(isi_ms, 99))))
    ax_isi.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Anchor unit → {Path(out_path).name}")


# ── Panel 5: Session timeline ─────────────────────────────────────────────────

def plot_session_timeline(sync, bhv_df, target_code, out_path):
    """
    Top   : all event types over the session (raster of codes)
    Bottom: inter-trial interval (time between consecutive target events)
    """
    events_bhv = bhv_df.loc[bhv_df["code"] == target_code, "ts"].values
    if len(events_bhv) < 2:
        print("  [WARN] Too few events for timeline — skipping")
        return

    warp    = sync["warp"]
    stretch = warp["stretch"]
    offset  = warp["offset"]

    fig, (ax_top, ax_iti) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    fig.suptitle(f"Session Timeline  |  code={target_code}",
                 fontsize=13, fontweight="bold")

    # ── Top: code raster over session ─────────────────────────────────────
    code_map = {2: ("Trial init", "#1f77b4", 0.6),
                3: ("Water",      "#17becf", 0.5),
               31: ("Reach",      "#ff7f0e", 0.6),
               11: ("Pull",       "#2ca02c", 0.8),
              -11: ("Inv. pull",  "#d62728", 0.8)}

    for row_i, (code, (label, color, alpha)) in enumerate(code_map.items()):
        times = bhv_df.loc[bhv_df["code"] == code, "ts"].values
        if len(times) > 0:
            ax_top.scatter(times, np.full_like(times, row_i),
                           s=6, color=color, alpha=alpha, label=label)

    ax_top.set_yticks(range(len(code_map)))
    ax_top.set_yticklabels([v[0] for v in code_map.values()], fontsize=9)
    ax_top.set_xlabel("Session time (s)", fontsize=11)
    ax_top.set_title("Event timeline", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper right", ncol=3)
    ax_top.grid(True, axis="x", alpha=0.25)

    # ── Bottom: inter-trial interval ───────────────────────────────────────
    iti = np.diff(events_bhv)
    trial_num = np.arange(len(iti)) + 1

    ax_iti.scatter(trial_num, iti, s=12, alpha=0.6, color="#2ca02c",
                   label="ITI (s)")
    # Running mean
    win = max(5, len(iti) // 20)
    run_mean = np.convolve(iti, np.ones(win) / win, mode="valid")
    ax_iti.plot(range(win // 2, len(iti) - win // 2 + 1), run_mean,
                color="black", lw=1.8, label=f"Running mean (w={win})")
    ax_iti.set_xlabel(f"Trial number (code={target_code})", fontsize=11)
    ax_iti.set_ylabel("Inter-trial interval (s)", fontsize=11)
    ax_iti.set_title("Trial spacing over session", fontsize=11)
    ax_iti.legend(fontsize=9)
    ax_iti.grid(True, alpha=0.25)

    # Annotation: total session stats
    total_s = float(events_bhv[-1] - events_bhv[0])
    ax_iti.text(0.02, 0.95,
                f"n={len(events_bhv)} events  |  "
                f"session={total_s/60:.1f}min  |  "
                f"mean ITI={iti.mean():.1f}s",
                transform=ax_iti.transAxes, fontsize=9,
                va="top", color="gray")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Session timeline → {Path(out_path).name}")


# ── Dashboard overview (single compact figure) ───────────────────────────────

def plot_dashboard_overview(sync, bhv_df, summary_df, target_code, out_path):
    """
    Single 2×3 figure summarising warp, classification, and timing at a glance.
    Suitable as a quick session quality overview.
    """
    warp    = sync["warp"]
    stretch = warp["stretch"]
    offset  = warp["offset"]
    rmse    = warp["rmse"]
    meta    = sync.get("metadata", {})

    events = sync.get("events", [])
    has_events = len(events) > 0
    bhv_t  = np.array([e["bhv_time"]   for e in events]) if has_events else np.array([])
    eph_t  = np.array([e["ephys_time"] for e in events]) if has_events else np.array([])
    resid  = (eph_t - (stretch * bhv_t + offset)) * 1000 if has_events else np.array([])

    events_bhv = bhv_df.loc[bhv_df["code"] == target_code, "ts"].values

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    fig.suptitle(
        f"Session Dashboard  |  code={target_code}  "
        f"|  Anchor unit {meta.get('anchor_unit','?')}  "
        f"|  RMSE = {rmse*1000:.2f}ms  "
        f"|  n_pairs = {meta.get('n_kept','?')}",
        fontsize=12, fontweight="bold"
    )

    # ── (0,0) Warp scatter ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    if has_events:
        ax.scatter(bhv_t, eph_t, s=8, alpha=0.5, color="steelblue")
        t_line = np.array([bhv_t.min(), bhv_t.max()])
        ax.plot(t_line, stretch * t_line + offset, "r-", lw=1.5,
                label=f"offset={offset*1000:.0f}ms")
        ax.legend(fontsize=8)
    ax.set_xlabel("Bhv time (s)", fontsize=9)
    ax.set_ylabel("Ephys time (s)", fontsize=9)
    ax.set_title("Clock alignment", fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── (0,1) Residuals ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if has_events:
        ax.scatter(bhv_t, resid, s=8, alpha=0.5, color="darkorange")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline( rmse*1000, color="r", lw=1.0, ls="--",
                   label=f"RMSE={rmse*1000:.1f}ms")
        ax.axhline(-rmse*1000, color="r", lw=1.0, ls="--")
        ax.legend(fontsize=8)
    ax.set_xlabel("Bhv time (s)", fontsize=9)
    ax.set_ylabel("Residual (ms)", fontsize=9)
    ax.set_title("Warp residuals", fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── (0,2) Classification pie ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    classes = ["MOTOR", "REWARD", "SUPPRESSED", "UNRESPONSIVE"]
    counts  = [int((summary_df["unit_class"] == c).sum()) for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]
    wedges, _, autotexts = ax.pie(
        counts,
        labels=[f"{c}\n{n}" for c, n in zip(classes, counts)],
        autopct="%1.0f%%", colors=colors, startangle=90,
        wedgeprops=dict(edgecolor="white", lw=1.5),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(f"Classification (n={sum(counts)})", fontsize=10)

    # ── (1,0) Firing rate violin ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    data_by_class = [
        summary_df.loc[summary_df["unit_class"] == c, "baseline_fr"].values
        for c in classes
    ]
    data_by_class = [d[~np.isnan(d)] for d in data_by_class]
    valid = [(i, d) for i, d in enumerate(data_by_class) if len(d) > 0]
    if valid:
        positions, data_v = zip(*valid)
        parts = ax.violinplot(list(data_v), positions=list(positions),
                              showmedians=True, showextrema=False)
        for pc, pos in zip(parts["bodies"], positions):
            pc.set_facecolor(colors[pos])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c[:3] for c in classes], fontsize=9)
    ax.set_ylabel("Baseline FR (Hz)", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Firing rate by class", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # ── (1,1) Latency histogram ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    lats = summary_df["peak_lat_s"].values
    for cls in ["MOTOR", "REWARD", "SUPPRESSED"]:
        sub = summary_df.loc[summary_df["unit_class"] == cls, "peak_lat_s"].values
        if len(sub):
            ax.hist(sub, bins=30, alpha=0.65, color=CLASS_COLORS[cls], label=cls)
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_xlabel("Peak latency (s)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Peak latency distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── (1,2) Session ITI ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if len(events_bhv) > 2:
        iti = np.diff(events_bhv)
        ax.plot(np.arange(len(iti)) + 1, iti, "o-",
                markersize=3, lw=0.8, alpha=0.6, color="#2ca02c")
        ax.axhline(iti.mean(), color="black", lw=1.2, ls="--",
                   label=f"mean={iti.mean():.1f}s")
        ax.set_xlabel("Trial #", fontsize=9)
        ax.set_ylabel("ITI (s)", fontsize=9)
        ax.set_title(f"Trial spacing  (n={len(events_bhv)})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard overview → {Path(out_path).name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive session visualisation dashboard")
    parser.add_argument("--sync",        required=True,
                        help="spike_sync.json")
    parser.add_argument("--summary",     required=True,
                        help="summary.csv from plot_neural_circuits.py")
    parser.add_argument("--bhv",         required=True,
                        help="GlobalLogInt*.csv")
    parser.add_argument("--ks-dir",      default=None,
                        help="Kilosort dir (needed for anchor unit plot)")
    parser.add_argument("--target-code", type=int, default=11,
                        help="Behavioral event code (default 11 = Pull)")
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  plot_session_dashboard.py  |  code={args.target_code}')
    print(f'{"="*60}\n')

    print("Loading inputs…")
    sync       = load_sync(args.sync)
    summary_df = pd.read_csv(args.summary)
    bhv_df     = load_behavior(args.bhv)
    print(f"  sync    : {args.sync}")
    print(f"  summary : {len(summary_df)} units")
    print(f"  bhv     : {len(bhv_df)} events")

    plot_warp_analysis(sync, out_dir / "1_warp_analysis.png")
    plot_classification(summary_df, out_dir / "2_classification.png")
    plot_latency_map(summary_df, out_dir / "3_latency_map.png", args.target_code)

    if args.ks_dir:
        plot_anchor_unit(sync, bhv_df, args.ks_dir,
                         args.target_code, out_dir / "4_anchor_unit.png")
    else:
        print("  [INFO] --ks-dir not provided — skipping anchor unit plot")

    plot_session_timeline(sync, bhv_df, args.target_code,
                          out_dir / "5_session_timeline.png")
    plot_dashboard_overview(sync, bhv_df, summary_df,
                            args.target_code, out_dir / "dashboard_summary.png")

    print(f'\n{"="*60}')
    print(f'  DONE  →  {out_dir}')
    print(f'  Files generated:')
    for p in sorted(out_dir.glob("*.png")):
        print(f'    {p.name}')
    print(f'{"="*60}\n')


if __name__ == "__main__":
    main()
