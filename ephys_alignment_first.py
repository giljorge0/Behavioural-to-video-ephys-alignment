#!/usr/bin/env python3
"""
ephys_alignment_run_ready.py
============================
Ephys-to-behavior temporal alignment pipeline.
Direct analog of video_alignment_run_ready.py — same math, same train/test
framework, same sync-file output format — but the "droplet" is now a
neural/electrical feature extracted from SpikeGLX recordings.

CANDIDATE FEATURES (analogs of the video droplet):
  1. solenoid_artifact  — sharp sub-ms transient in raw AP or LF at reward
  2. lfp_deflection     — reward-triggered LFP dip in 1-100 Hz filtered trace
  3. mua_envelope       — rising edge of RMS(300-3000 Hz) multiunit burst
  4. lick_band          — rhythmic 5-8 Hz power burst (licking resonance)

INPUTS:
  --bhv   GlobalLog*.csv       (same behavior log as video pipeline)
  --ap    *_t0.imec0.ap.bin    (SpikeGLX AP band, ~30kHz)
  --lf    *_t0.imec0.lf.bin    (SpikeGLX LF band, ~2.5kHz)
  --ks    kilosort_output/     (folder with spike_times.npy etc.)
  --out   output_folder/

OUTPUTS (same format as video pipeline):
  sync_<session>.json          — warp + per-trial mapping + confidence
  syncfix_<session>.mat        — MATLAB-compatible
  <session>_diagnostics.npz   — all signals + metadata
  hp_sweep_<session>.csv       — hyperparameter grid results (if --sweep)

USAGE EXAMPLES:
  # single session, solenoid artifact feature
  python ephys_alignment_run_ready.py \
    --bhv   /data/R1/GlobalLog.csv \
    --ap    /data/R1/probe0/R1.ap.bin \
    --lf    /data/R1/probe0/R1.lf.bin \
    --out   /data/R1/ephys_out/ \
    --feature solenoid_artifact \
    --make-sync

  # run hyperparameter sweep across all features
  python ephys_alignment_run_ready.py \
    --bhv   /data/R1/GlobalLog.csv \
    --ap    /data/R1/probe0/R1.ap.bin \
    --lf    /data/R1/probe0/R1.lf.bin \
    --out   /data/R1/ephys_out/ \
    --sweep

  # evaluate against video-derived ground truth
  python ephys_alignment_run_ready.py \
    --bhv   /data/R1/GlobalLog.csv \
    --ap    /data/R1/probe0/R1.ap.bin \
    --lf    /data/R1/probe0/R1.lf.bin \
    --gt    /data/R1/out_dryrun/sync_Camera1.json \
    --out   /data/R1/ephys_out/ \
    --feature all --sweep
"""

import os
import warnings
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import (butter, sosfiltfilt, find_peaks, medfilt,
                           hilbert, welch)
from scipy.optimize import least_squares, minimize
from scipy.interpolate import interp1d
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
    _use_tqdm = True
except ImportError:
    _use_tqdm = False

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers  (copied from video pipeline for consistency)
# ──────────────────────────────────────────────────────────────────────────────

def normalize01(x, lo_perc=5.0, hi_perc=95.0):
    x = np.asarray(x, dtype=float)
    nan_mask = np.isnan(x)
    valid = x[~nan_mask]
    if valid.size == 0:
        return np.full_like(x, np.nan)
    lo, hi = np.percentile(valid, lo_perc), np.percentile(valid, hi_perc)
    if hi == lo:
        lo, hi = valid.min(), valid.max()
    if hi == lo:
        return np.zeros_like(x)
    out = np.empty_like(x)
    out.fill(np.nan)
    out[~nan_mask] = np.clip((valid - lo) / (hi - lo), 0.0, 1.0)
    return out


def odd_int(x):
    x = int(x)
    return max(1, x + (x % 2 == 0))


def safe_medfilt(x, wf):
    wf = odd_int(max(1, int(wf)))
    try:
        return medfilt(x, kernel_size=wf)
    except Exception:
        out = np.copy(x)
        half = wf // 2
        for i in range(len(x)):
            a, b = max(0, i - half), min(len(x), i + half + 1)
            out[i] = np.nanmedian(x[a:b])
        return out


def tempwarperrfun(B, t, exp_ephys, exp_bhv):
    s, o = float(B[0]), float(B[1])
    if s == 0:
        s = 1.0
    try:
        f = interp1d(t, exp_ephys, bounds_error=False, fill_value=np.nan)
        v_at_b = f(s * t + o)
        return np.nansum(np.nan_to_num(v_at_b - exp_bhv) ** 2)
    except Exception:
        return 1e9


def fit_sync_warp(behavior_times, ephys_times,
                  bounds=(0.9, 1.1), offset_bounds=(-10.0, 10.0)):
    """Robust linear warp: ephys_time = stretch * behavior_time + offset."""
    behavior_times = np.asarray(behavior_times, dtype=float)
    ephys_times    = np.asarray(ephys_times, dtype=float)
    mask = np.isfinite(behavior_times) & np.isfinite(ephys_times)
    if mask.sum() < 2:
        return dict(stretch=1.0, offset=0.0, residuals=np.array([]),
                    rmse=np.nan, n=int(mask.sum()), opt_res=None)
    x, y = behavior_times[mask], ephys_times[mask]
    res = least_squares(lambda p: y - (p[0]*x + p[1]),
                        [1.0, 0.0], loss='huber',
                        bounds=([bounds[0], offset_bounds[0]],
                                [bounds[1], offset_bounds[1]]),
                        max_nfev=5000)
    s, o = float(res.x[0]), float(res.x[1])
    resid = y - (s*x + o)
    rmse = float(np.sqrt(np.nanmean(resid**2)))
    return dict(stretch=s, offset=o, residuals=resid, rmse=rmse,
                n=int(mask.sum()), opt_res=res)


def evaluate_against_ground_truth(pred_events, true_events, tolerance_ms=50):
    pred = np.asarray(pred_events, dtype=float)
    true = np.asarray(true_events, dtype=float)
    m = np.isfinite(true)
    if m.sum() == 0:
        return dict(n=0)
    p, t = pred[m], true[m]
    ae = np.where(np.isfinite(p), np.abs(p - t), np.nan)
    tol = tolerance_ms / 1000.0
    tp = int(np.nansum(ae <= tol))
    fn = int(np.sum(~np.isfinite(p)))
    fp = int(np.sum(np.isfinite(p) & ~np.isfinite(t)))
    prec  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec   = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1    = (2*prec*rec/(prec+rec) if not (np.isnan(prec) or np.isnan(rec)
             or (prec+rec) == 0) else np.nan)
    return dict(n=int(m.sum()), mae=float(np.nanmean(ae)),
                rmse=float(np.sqrt(np.nanmean(ae**2))),
                tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def write_sync_file(path, warp, events_map, metadata=None):
    out = dict(
        warp=dict(stretch=float(warp.get('stretch', 1.0)),
                  offset=float(warp.get('offset', 0.0)),
                  rmse=float(warp.get('rmse', np.nan))),
        events=events_map,
        metadata=metadata or {}
    )
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=lambda v: None if (
            isinstance(v, float) and np.isnan(v)) else v)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# SpikeGLX binary reader
# ──────────────────────────────────────────────────────────────────────────────

def read_spikeglx_meta(bin_path):
    """Parse companion .meta file and return dict.
    Handles both file.ap.bin→file.ap.meta and file.ap→file.ap.meta naming.
    """
    bin_path = Path(bin_path)
    # Case 1: file.ap.bin → file.ap.meta  (replace .bin suffix)
    meta_path = bin_path.with_suffix('.meta')
    if not meta_path.exists():
        # Case 2: file.ap → file.ap.meta  (append .meta)
        meta_path = Path(str(bin_path) + '.meta')
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta file not found. Tried:\n"
            f"  {bin_path.with_suffix('.meta')}\n"
            f"  {str(bin_path) + '.meta'}")
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
    return meta


def parse_spikeglx_meta(meta):
    """Return (n_channels, sample_rate, n_samples, uV_per_bit)."""
    # total channels including sync
    n_ch_total = int(meta.get('nSavedChans', meta.get('nChans', 385)))
    # sample rate
    sr = float(meta.get('imSampRate', meta.get('sRateHz', 30000)))
    # recording duration in samples
    n_samples = int(meta.get('fileSizeBytes', 0)) // (2 * n_ch_total)
    # voltage scaling: 1 count → µV
    # SpikeGLX stores AP as int16; gain is in meta
    # imProbeOpt or imAiRangeMax used for scaling
    try:
        ai_range = float(meta.get('imAiRangeMax', 0.6))  # Volts
        gain = float(meta.get('imGain', 500))
        uv_per_bit = 1e6 * ai_range / (gain * 32768.0)
    except Exception:
        uv_per_bit = 1.0  # fallback: keep counts
    return n_ch_total, sr, n_samples, uv_per_bit


def load_ephys_chunk(bin_path, start_sample, end_sample,
                     channel_ids=None, dtype=np.int16):
    """
    Memory-efficient loader: read [start_sample, end_sample) from a
    SpikeGLX .bin file for selected channels only.

    Parameters
    ----------
    bin_path     : path to .bin file
    start_sample : first sample index (inclusive)
    end_sample   : last sample index (exclusive)
    channel_ids  : list/array of channel indices (0-based); None = all
    dtype        : data type stored in file (default int16)

    Returns
    -------
    data : ndarray shape (n_channels_selected, n_samples)
    """
    meta          = read_spikeglx_meta(bin_path)
    n_ch, sr, _, _ = parse_spikeglx_meta(meta)

    n_samples_to_read = end_sample - start_sample
    itemsize = np.dtype(dtype).itemsize
    byte_offset = start_sample * n_ch * itemsize

    mm = np.memmap(bin_path, dtype=dtype, mode='r',
                   offset=byte_offset,
                   shape=(n_samples_to_read, n_ch))

    if channel_ids is None:
        data = mm[:, :].T.copy().astype(float)
    else:
        data = mm[:, np.asarray(channel_ids)].T.copy().astype(float)
    return data  # shape: (n_selected_channels, n_samples_to_read)


def load_ephys_full(bin_path, channel_ids=None, dtype=np.int16,
                    max_gb=4.0):
    """
    Load entire recording for selected channels.
    Warns if file exceeds max_gb and suggests chunked loading.
    """
    meta          = read_spikeglx_meta(bin_path)
    n_ch, sr, n_samples, uv_per_bit = parse_spikeglx_meta(meta)
    size_gb = n_ch * n_samples * 2 / 1e9
    if size_gb > max_gb:
        warnings.warn(
            f"File is {size_gb:.1f} GB > {max_gb} GB limit. "
            "Consider using --chunk-size to load in pieces.")
    data = load_ephys_chunk(bin_path, 0, n_samples, channel_ids, dtype)
    return data, sr, uv_per_bit


# ──────────────────────────────────────────────────────────────────────────────
# Sync channel reader
# Last channel (index n_ch-1, typically 384) is a digital TTL that toggles
# on every behavior event.  Reading it gives direct behavior→ephys alignment
# for sessions where it is present, and serves as ground truth to validate
# the feature-based method.
# ──────────────────────────────────────────────────────────────────────────────

def extract_sync_channel(ap_bin, bhv_df, sr_ap=None, sync_ch_idx=None,
                         verbose=True):
    """
    Read the last channel from the AP bin, detect all TTL edges, match them
    to behavior event times, and return GT ephys times for reward events.

    The sync channel toggles 0↔1 on every behavior event logged in bhv_df.
    We fit a linear warp (behavior_time → ephys_time) using ALL matched edges,
    then return where each reward event (code==3) falls in ephys time.

    Parameters
    ----------
    ap_bin     : path to AP .bin or .ap file
    bhv_df     : DataFrame from load_behavior() — columns [timestamp, code]
    sr_ap      : sample rate (read from meta if None)
    sync_ch_idx: channel index of sync signal (None = last channel)

    Returns
    -------
    gt_reward_ephys_times : array length n_rewards, ephys times of rewards
    sync_edge_times_ephys : all edge times in ephys seconds (for diagnostics)
    warp_sync             : warp dict from the sync-channel fit
    """
    meta = read_spikeglx_meta(ap_bin)
    n_ch, sr, n_samp, _ = parse_spikeglx_meta(meta)
    if sr_ap is None:
        sr_ap = sr
    if sync_ch_idx is None:
        sync_ch_idx = n_ch - 1   # last channel = sync

    if verbose:
        print(f"[SYNC] Reading sync channel {sync_ch_idx} "
              f"({n_samp/sr_ap/60:.1f} min @ {sr_ap:.0f} Hz) …")

    # Load only the sync channel — 1 channel × n_samp × 2 bytes (manageable)
    sync_raw = load_ephys_chunk(ap_bin, 0, n_samp,
                                 channel_ids=[sync_ch_idx])[0]  # 1-D

    # Detect edges: any sample where value changes
    diff = np.diff(sync_raw.astype(np.int32))
    edge_samples = np.where(diff != 0)[0] + 1   # +1: edge is at the new value
    edge_times_ephys = edge_samples / sr_ap      # seconds in ephys clock

    if verbose:
        print(f"[SYNC] {len(edge_samples)} edges detected in sync channel")

    # Get all behavior event times (same order as edges)
    if bhv_df.shape[1] >= 2:
        bhv_event_times = bhv_df['timestamp'].values.astype(float)
        bhv_codes       = bhv_df.iloc[:, 1].values
    else:
        bhv_event_times = bhv_df['timestamp'].values.astype(float)
        bhv_codes       = np.ones(len(bhv_event_times))

    n_edges = len(edge_times_ephys)
    n_events = len(bhv_event_times)

    if n_edges == 0:
        if verbose:
            print("[SYNC] No edges found — sync channel is flat. "
                  "Falling back to feature-based GT.")
        n_rewards = int(np.sum(bhv_codes == 3))
        return (np.full(n_rewards, np.nan),
                np.array([]),
                {'stretch': 1.0, 'offset': 0.0, 'rmse': np.nan, 'n': 0})

    # Match edges to behavior events.
    # Both sequences are ordered in time; trim whichever is longer.
    n_match = min(n_edges, n_events)
    if verbose and abs(n_edges - n_events) > 5:
        print(f"[SYNC] Warning: {n_events} behavior events vs "
              f"{n_edges} sync edges — using first {n_match} of each.")

    bhv_t_match   = bhv_event_times[:n_match]
    ephys_t_match = edge_times_ephys[:n_match]

    # Fit warp using ALL matched events (much more data than reward-only)
    warp_sync = fit_sync_warp(bhv_t_match, ephys_t_match,
                               bounds=(0.9, 1.1),
                               offset_bounds=(-300.0, 300.0))
    if verbose:
        print(f"[SYNC] Warp from sync channel: "
              f"stretch={warp_sync['stretch']:.6f}  "
              f"offset={warp_sync['offset']:.3f}s  "
              f"rmse={warp_sync['rmse']:.5f}s  "
              f"n={warp_sync['n']}")

    # Compute GT ephys times for reward events specifically
    reward_mask       = (bhv_codes == 3)
    reward_bhv_times  = bhv_event_times[reward_mask]
    gt_reward_ephys   = (warp_sync['stretch'] * reward_bhv_times
                         + warp_sync['offset'])

    return gt_reward_ephys, edge_times_ephys, warp_sync


# ──────────────────────────────────────────────────────────────────────────────
# Feature cache  — save/load extracted traces so 105 GB is only read once
# ──────────────────────────────────────────────────────────────────────────────

def save_feature_cache(cache_path, feature_traces, feature_times,
                       sync_edge_times=None, sr_ap=None, sr_lf=None):
    """Save feature traces to .npz so re-runs skip the binary loading."""
    payload = {}
    for k, v in feature_traces.items():
        payload[f'trace_{k}'] = np.asarray(v)
    for k, v in feature_times.items():
        payload[f'time_{k}']  = np.asarray(v)
    if sync_edge_times is not None:
        payload['sync_edge_times'] = np.asarray(sync_edge_times)
    if sr_ap is not None:
        payload['sr_ap'] = np.array([sr_ap])
    if sr_lf is not None:
        payload['sr_lf'] = np.array([sr_lf])
    np.savez_compressed(str(cache_path), **payload)
    return cache_path


def load_feature_cache(cache_path):
    """
    Load cached feature traces.
    Returns (feature_traces, feature_times, sync_edge_times, sr_ap, sr_lf).
    """
    d = np.load(str(cache_path), allow_pickle=False)
    feature_traces = {}
    feature_times  = {}
    for key in d.files:
        if key.startswith('trace_'):
            feature_traces[key[6:]] = d[key]
        elif key.startswith('time_'):
            feature_times[key[5:]] = d[key]
    sync_edge_times = d['sync_edge_times'] if 'sync_edge_times' in d.files else None
    sr_ap = float(d['sr_ap'][0]) if 'sr_ap' in d.files else None
    sr_lf = float(d['sr_lf'][0]) if 'sr_lf' in d.files else None
    return feature_traces, feature_times, sync_edge_times, sr_ap, sr_lf

# ──────────────────────────────────────────────────────────────────────────────
# Behaviour log loader  (same as video pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def load_behavior(bhv_file):
    df = pd.read_csv(bhv_file, header=None)
    df.columns = ['timestamp', 'code'] if df.shape[1] >= 2 else ['timestamp']
    if df.shape[1] >= 2:
        sr_mask = df.iloc[:, 1] == 1000
        if sr_mask.any():
            t0 = df['timestamp'][sr_mask].iloc[0]
            df['timestamp'] = df['timestamp'] - t0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 1 — Solenoid / valve artifact
# Sharp, sub-ms electrical transient when reward valve fires.
# Best detected on raw AP trace (30 kHz); appears on ALL channels simultaneously.
# ──────────────────────────────────────────────────────────────────────────────

def extract_solenoid_artifact(ap_data, sr_ap, uv_per_bit=1.0,
                              threshold_sd=8.0, min_distance_samples=None,
                              n_channels_for_consensus=20):
    """
    Detect reward-valve electromagnetic artifacts in the AP band.

    Strategy
    --------
    1. Compute the cross-channel median trace (common-mode signal).
       Solenoid fires affect ALL channels simultaneously → they survive
       median-across-channels; single-unit spikes are local and cancel out.
    2. Z-score and threshold.
    3. Keep peaks separated by at least min_distance (default 100 ms).

    Parameters
    ----------
    ap_data   : (n_ch, n_samples) float array, in counts or µV
    sr_ap     : AP sample rate (Hz), typically 30000
    threshold_sd : how many SDs above mean to call an artifact
    n_channels_for_consensus : how many channels to median over

    Returns
    -------
    event_samples : 1-D array of sample indices for each detected artifact
    common_mode   : 1-D float array of the median trace (for diagnostics)
    """
    if min_distance_samples is None:
        min_distance_samples = int(0.1 * sr_ap)  # 100 ms

    # Use a random subset if many channels (faster)
    n_ch = ap_data.shape[0]
    if n_ch > n_channels_for_consensus:
        rng = np.random.RandomState(42)
        ch_idx = rng.choice(n_ch, n_channels_for_consensus, replace=False)
    else:
        ch_idx = np.arange(n_ch)

    # Common-mode trace: captures electrical artefacts shared across all channels
    common_mode = np.median(ap_data[ch_idx, :], axis=0).astype(float)

    # Absolute value so we detect both polarities
    abs_cm = np.abs(common_mode)

    # Z-score
    mu, sd = np.nanmean(abs_cm), np.nanstd(abs_cm)
    if sd == 0:
        return np.array([], dtype=int), common_mode
    z = (abs_cm - mu) / sd

    # Peak detection
    event_samples, _ = find_peaks(z, height=threshold_sd,
                                  distance=min_distance_samples)
    return event_samples, common_mode


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 2 — LFP deflection
# Reward-triggered slow cortical potential in 1-100 Hz band.
# ──────────────────────────────────────────────────────────────────────────────

def extract_lfp_deflection(lf_data, sr_lf, uv_per_bit=1.0,
                            bandpass=(1.0, 100.0),
                            n_channels_pca=20,
                            smooth_ms=50.0):
    """
    Extract LFP deflection signal suitable for event detection.

    Strategy
    --------
    1. Bandpass filter (default 1-100 Hz).
    2. Compute PC1 across channels (captures global state changes).
    3. Smooth and z-score → lfp_pc1_z.

    Returns
    -------
    lfp_time   : time vector (seconds from recording start)
    lfp_pc1_z  : smoothed, z-scored PC1 trace
    lfp_pc1    : raw (unsmoothed) PC1
    explained  : explained variance ratios
    """
    n_ch, n_samp = lf_data.shape

    # Bandpass filter
    lo, hi = bandpass
    sos = butter(4, [lo / (sr_lf/2), hi / (sr_lf/2)],
                 btype='band', output='sos')
    # Filter a subset of channels for speed
    rng = np.random.RandomState(42)
    ch_idx = (rng.choice(n_ch, min(n_ch, n_channels_pca), replace=False)
               if n_ch > n_channels_pca else np.arange(n_ch))
    filt = np.zeros((len(ch_idx), n_samp), dtype=float)
    for ii, ci in enumerate(ch_idx):
        filt[ii] = sosfiltfilt(sos, lf_data[ci].astype(float))

    # PCA
    sc = StandardScaler()
    Xs = sc.fit_transform(filt.T)   # (n_samp, n_ch_subset)
    pca = PCA(n_components=min(10, Xs.shape[1]))
    Z = pca.fit_transform(Xs)
    explained = pca.explained_variance_ratio_
    pc1 = Z[:, 0]

    # Smooth
    wf = odd_int(max(1, int(smooth_ms / 1000.0 * sr_lf)))
    pc1_smooth = safe_medfilt(pc1, wf)

    # Normalize and z-score
    pc1_norm = normalize01(pc1_smooth)
    m, s = np.nanmean(pc1_norm), np.nanstd(pc1_norm)
    pc1_z = (pc1_norm - m) / (s if s > 0 else 1.0)

    lfp_time = np.arange(n_samp) / sr_lf
    return lfp_time, pc1_z, pc1, explained


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 3 — MUA envelope
# Rising edge of multi-unit activity when mouse initiates movement.
# ──────────────────────────────────────────────────────────────────────────────

def extract_mua_envelope(ap_data, sr_ap, uv_per_bit=1.0,
                          bandpass=(300.0, 3000.0),
                          smooth_ms=25.0,
                          n_channels_use=20,
                          downsample_to_hz=1000.0):
    """
    Compute MUA envelope: RMS of high-pass filtered AP band.

    Strategy
    --------
    1. High-pass / bandpass filter (300-3000 Hz).
    2. Full-wave rectify and smooth → instantaneous power.
    3. Average across channels.
    4. Optionally downsample for speed.

    Returns
    -------
    mua_time     : time vector in seconds (downsampled)
    mua_env_z    : z-scored envelope trace
    mua_env_raw  : raw (unscaled) envelope
    """
    n_ch = ap_data.shape[0]
    rng = np.random.RandomState(42)
    ch_idx = (rng.choice(n_ch, min(n_ch, n_channels_use), replace=False)
               if n_ch > n_channels_use else np.arange(n_ch))

    lo, hi = bandpass
    sos = butter(4, [lo / (sr_ap/2), hi / (sr_ap/2)],
                 btype='band', output='sos')

    # Smooth kernel
    smooth_samp = max(1, int(smooth_ms / 1000.0 * sr_ap))

    envelopes = []
    for ci in ch_idx:
        filt = sosfiltfilt(sos, ap_data[ci].astype(float))
        # RMS via squared + smooth
        sq = filt ** 2
        env = np.sqrt(safe_medfilt(sq, odd_int(smooth_samp)))
        envelopes.append(env)
    mua_env_raw = np.mean(envelopes, axis=0)

    # Downsample
    ds_factor = max(1, int(sr_ap / downsample_to_hz))
    mua_ds  = mua_env_raw[::ds_factor]
    sr_ds   = sr_ap / ds_factor
    mua_time = np.arange(len(mua_ds)) / sr_ds

    # Z-score
    m, s = np.nanmean(mua_ds), np.nanstd(mua_ds)
    mua_env_z = (mua_ds - m) / (s if s > 0 else 1.0)

    return mua_time, mua_env_z, mua_ds


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 4 — Lick-band power (5-8 Hz)
# Rhythmic lick artifacts in LFP create a spectral signature.
# ──────────────────────────────────────────────────────────────────────────────

def extract_lick_band(lf_data, sr_lf, uv_per_bit=1.0,
                       lick_band=(5.0, 8.0),
                       smooth_ms=200.0,
                       n_channels_use=20):
    """
    Extract instantaneous power in the lick-frequency band (5-8 Hz).

    Returns
    -------
    lick_time  : seconds
    lick_pow_z : z-scored band power
    lick_pow   : raw band power
    """
    n_ch = lf_data.shape[0]
    rng = np.random.RandomState(42)
    ch_idx = (rng.choice(n_ch, min(n_ch, n_channels_use), replace=False)
               if n_ch > n_channels_use else np.arange(n_ch))

    lo, hi = lick_band
    sos = butter(4, [lo / (sr_lf/2), hi / (sr_lf/2)],
                 btype='band', output='sos')

    powers = []
    for ci in ch_idx:
        filt = sosfiltfilt(sos, lf_data[ci].astype(float))
        # Amplitude envelope via Hilbert
        try:
            env = np.abs(hilbert(filt))
        except Exception:
            env = np.abs(filt)
        powers.append(env)
    lick_pow = np.mean(powers, axis=0)

    # Smooth
    wf = odd_int(max(1, int(smooth_ms / 1000.0 * sr_lf)))
    lick_pow = safe_medfilt(lick_pow, wf)

    lick_time = np.arange(len(lick_pow)) / sr_lf
    m, s = np.nanmean(lick_pow), np.nanstd(lick_pow)
    lick_pow_z = (lick_pow - m) / (s if s > 0 else 1.0)
    return lick_time, lick_pow_z, lick_pow


# ──────────────────────────────────────────────────────────────────────────────
# Universal event detector
# (same logic as detect_droplet_events in video pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def detect_ephys_events(feature_traces, feature_times,
                        trial_rwd_times,
                        search_window=(-0.1, 0.5),
                        weights=None,
                        min_confidence=0.3,
                        min_candidates=2):
    """
    Per-trial event detector using one or multiple ephys feature traces.

    Parameters
    ----------
    feature_traces : dict  {name: 1-D z-scored trace}
    feature_times  : dict  {name: 1-D time vector in seconds}
                     (may have different sample rates — all are interpolated)
    trial_rwd_times : 1-D array of reward times in behavior clock (seconds)
    search_window   : (pre_s, post_s) around reward time to search
    weights         : dict {feature_name: weight}, normalised internally
    min_confidence  : minimum score to accept an event
    Returns
    -------
    events         : list of per-trial dicts (or None if not detected)
    all_candidates : list of per-trial candidate lists
    """
    feat_names = list(feature_traces.keys())
    if weights is None:
        # Equal weights by default
        weights = {n: 1.0 / len(feat_names) for n in feat_names}
    # Normalise weights
    total_w = sum(weights.values())
    weights = {n: w / total_w for n, w in weights.items()}

    n_trials = len(trial_rwd_times)
    events        = [None] * n_trials
    all_candidates = [None] * n_trials

    for tt in range(n_trials):
        t_rwd   = trial_rwd_times[tt]
        t_start = t_rwd + search_window[0]
        t_end   = t_rwd + search_window[1]

        candidates = {}  # key: time (s), value: feature scores

        for fname in feat_names:
            trace = feature_traces[fname]
            times = feature_times[fname]

            # Crop to search window
            in_win = (times >= t_start) & (times <= t_end)
            if in_win.sum() < 2:
                continue

            t_win  = times[in_win]
            tr_win = trace[in_win]

            # --- Feature-specific peak direction ---
            # solenoid & mua: positive peak (energy burst)
            # lfp_deflection: negative dip (deflection = drop in PC1)
            # lick_band: positive burst
            if fname == 'lfp_deflection':
                tr_search = -tr_win  # invert for trough detection
            else:
                tr_search = tr_win

            # Find local maxima in the search window
            peaks, props = find_peaks(tr_search,
                                       height=np.percentile(tr_search, 50),
                                       distance=max(1, len(tr_win)//20))

            if peaks.size == 0:
                # Fallback: use global argmax in window
                peaks = np.array([int(np.argmax(tr_search))])

            for pk in peaks:
                t_event = float(t_win[pk])
                score_f = float(tr_win[pk])
                if t_event not in candidates:
                    candidates[t_event] = {}
                candidates[t_event][fname] = score_f

        # Score each candidate
        scored = []
        for t_cand, feat_dict in candidates.items():
            score = 0.0
            for fn in feat_names:
                s = feat_dict.get(fn, 0.0)
                score += weights.get(fn, 0.0) * max(0.0, s)
            scored.append({'time': t_cand, 'scores': feat_dict,
                            'confidence': score, 'trial_idx': tt})

        # Normalise confidence across candidates in this trial
        if scored:
            n_cands = len(scored)
            cvals = np.array([c['confidence'] for c in scored])
            cmin, cmax = cvals.min(), cvals.max()

            if n_cands < min_candidates:
                # Too few candidates to trust within-trial normalisation.
                # Use raw score scaled by a fixed reference (3 SD = strong signal).
                # This prevents a single weak candidate from scoring 1.0.
                for i, c in enumerate(scored):
                    c['confidence_norm'] = float(min(1.0, max(0.0,
                                                    c['confidence'] / 3.0)))
                    c['n_candidates'] = n_cands
            elif cmax > cmin:
                cvals_n = (cvals - cmin) / (cmax - cmin)
                for i, c in enumerate(scored):
                    c['confidence_norm'] = float(cvals_n[i])
                    c['n_candidates'] = n_cands
            else:
                for i, c in enumerate(scored):
                    c['confidence_norm'] = 1.0
                    c['n_candidates'] = n_cands

            best_idx = int(np.argmax([c['confidence_norm'] for c in scored]))
            best = scored[best_idx]

            # Reject if too few candidates OR score below threshold
            if n_cands >= min_candidates and best['confidence_norm'] >= min_confidence:
                events[tt] = {
                    'trial_idx': tt,
                    'time': best['time'],
                    'confidence': best['confidence_norm'],
                    'n_candidates': n_cands,
                    'feature_scores': best['scores'],
                    'method': '+'.join(feat_names)
                }

        all_candidates[tt] = scored

    return events, all_candidates


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter sweep  (mirrors video pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def hyperparameter_sweep(session_dict, out_folder, session_stem,
                         train_frac=0.8,
                         minconf_list=None, weight_combos=None,
                         seed=0, tolerance_ms=50):
    """
    Sweep (min_confidence × feature_weights) and report best by MAE.
    session_dict must contain keys from process_one_session return value.
    """
    if minconf_list is None:
        minconf_list = [0.2, 0.3, 0.4, 0.5]
    if weight_combos is None:
        # (solenoid_w, lfp_w, mua_w, lick_w) grid — only rows that sum ≤ 1
        weight_combos = [
            {'solenoid_artifact': 1.0, 'lfp_deflection': 0.0,
             'mua_envelope': 0.0, 'lick_band': 0.0},
            {'solenoid_artifact': 0.0, 'lfp_deflection': 1.0,
             'mua_envelope': 0.0, 'lick_band': 0.0},
            {'solenoid_artifact': 0.0, 'lfp_deflection': 0.0,
             'mua_envelope': 1.0, 'lick_band': 0.0},
            {'solenoid_artifact': 0.0, 'lfp_deflection': 0.0,
             'mua_envelope': 0.0, 'lick_band': 1.0},
            {'solenoid_artifact': 0.5, 'lfp_deflection': 0.5,
             'mua_envelope': 0.0, 'lick_band': 0.0},
            {'solenoid_artifact': 0.4, 'lfp_deflection': 0.3,
             'mua_envelope': 0.3, 'lick_band': 0.0},
            {'solenoid_artifact': 0.25, 'lfp_deflection': 0.25,
             'mua_envelope': 0.25, 'lick_band': 0.25},
        ]

    feature_traces = session_dict['feature_traces']
    feature_times  = session_dict['feature_times']
    trial_rwd_times = np.asarray(session_dict['trial_rwd_times'], dtype=float)
    gt_ephys_times  = np.asarray(
        session_dict.get('gt_ephys_times',
                         np.full(len(trial_rwd_times), np.nan)), dtype=float)

    rows = []
    rng  = np.random.RandomState(seed)

    for wcombo in weight_combos:
        # Only keep features actually available
        avail_w = {k: v for k, v in wcombo.items() if k in feature_traces}
        if not avail_w:
            continue
        for min_conf in minconf_list:
            events, _ = detect_ephys_events(
                feature_traces, feature_times, trial_rwd_times,
                weights=avail_w, min_confidence=min_conf)

            pred_times = np.array(
                [e['time'] if e is not None else np.nan for e in events],
                dtype=float)

            # Fit warp and evaluate only where we have GT
            valid_mask = np.isfinite(trial_rwd_times) & np.isfinite(gt_ephys_times)
            n_valid = int(valid_mask.sum())
            metrics  = dict(n=n_valid, mae=np.nan, rmse=np.nan,
                            precision=np.nan, recall=np.nan, f1=np.nan)
            warp_info = dict(stretch=np.nan, offset=np.nan, rmse=np.nan,
                             n_train=0, n_test=0)

            if n_valid >= 3:
                idxs = np.where(valid_mask)[0].copy()
                rng.shuffle(idxs)
                n_train = max(1, int(len(idxs) * train_frac))
                train_idx, test_idx = idxs[:n_train], idxs[n_train:]
                warp = fit_sync_warp(trial_rwd_times[train_idx],
                                      gt_ephys_times[train_idx])
                pred_gt = warp['stretch'] * trial_rwd_times[test_idx] + warp['offset']
                metrics = evaluate_against_ground_truth(
                    pred_gt, gt_ephys_times[test_idx],
                    tolerance_ms=tolerance_ms)
                warp_info = dict(stretch=warp['stretch'],
                                 offset=warp['offset'],
                                 rmse=warp['rmse'],
                                 n_train=len(train_idx),
                                 n_test=len(test_idx))

            rows.append({
                'weights': json.dumps({k: round(v, 3) for k, v in avail_w.items()}),
                'min_conf': min_conf,
                'n_valid': n_valid,
                **{f'w_{k}': avail_w.get(k, 0.0) for k in
                   ['solenoid_artifact', 'lfp_deflection',
                    'mua_envelope', 'lick_band']},
                'n_train': warp_info['n_train'],
                'n_test':  warp_info['n_test'],
                'warp_stretch': warp_info['stretch'],
                'warp_offset':  warp_info['offset'],
                'warp_rmse':    warp_info['rmse'],
                'mae':       metrics.get('mae', np.nan),
                'rmse':      metrics.get('rmse', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall':    metrics.get('recall', np.nan),
                'f1':        metrics.get('f1', np.nan),
            })

    # Save CSV
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    csv_path = out_folder / f"hp_sweep_{session_stem}.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: ('' if (isinstance(r[k], float)
                                       and np.isnan(r[k])) else r[k])
                             for k in keys})

    # Print best
    valid_rows = [r for r in rows if not (isinstance(r['mae'], float)
                                           and np.isnan(r['mae']))]
    if valid_rows:
        best = min(valid_rows, key=lambda x: x['mae'])
        print(f"[SWEEP DONE] best → weights={best['weights']}  "
              f"min_conf={best['min_conf']}  "
              f"mae={best['mae']:.4f}s  rmse={best['rmse']:.4f}s  "
              f"precision={best['precision']}  recall={best['recall']}")
    else:
        print("[SWEEP DONE] no valid configurations (need ≥3 GT trials).")

    return str(csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# Save diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def save_diagnostics(out_folder, name_prefix, **arrays):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    npz_path = out_folder / f"{name_prefix}_diagnostics.npz"
    saveable = {}
    for k, v in arrays.items():
        try:
            arr = np.asarray(v)
            saveable[k] = arr
        except Exception:
            pass
    try:
        np.savez_compressed(str(npz_path), **saveable)
    except Exception as e:
        warnings.warn(f"Failed saving diagnostics: {e}")
    return npz_path


# ──────────────────────────────────────────────────────────────────────────────
# Main session processor
# ──────────────────────────────────────────────────────────────────────────────

def process_one_session(bhv_file, ap_bin=None, lf_bin=None, ks_folder=None,
                        out_folder=None, gt_sync_json=None,
                        feature_list=None,
                        make_sync=False, train_frac=0.8,
                        min_conf=0.3,
                        ap_channels=None, lf_channels=None,
                        chunk_sec=60.0,
                        min_candidates=2,
                        use_cache=True,
                        verbose=True):
    """
    Full ephys-to-behavior alignment for a single session.

    Parameters
    ----------
    bhv_file     : path to GlobalLog*.csv
    ap_bin       : path to *.ap.bin  (SpikeGLX AP band)
    lf_bin       : path to *.lf.bin  (SpikeGLX LF band)
    ks_folder    : path to Kilosort output folder (optional)
    out_folder   : where to save outputs
    gt_sync_json : video-pipeline sync JSON for ground-truth comparison
    feature_list : list of feature names to extract; None = all available
    make_sync    : whether to write the canonical sync JSON
    train_frac   : fraction of GT trials used for warp training
    min_conf     : minimum confidence for event acceptance
    ap_channels  : channel indices to use for AP features (None = auto subset)
    lf_channels  : channel indices to use for LF features (None = auto subset)
    chunk_sec    : seconds per chunk when memory-efficient loading needed
    verbose      : print progress

    Returns
    -------
    session_dict with all signals and results.
    """
    bhv_file   = Path(bhv_file)
    out_folder = Path(out_folder) if out_folder else bhv_file.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    if feature_list is None:
        feature_list = []
        if ap_bin is not None:
            feature_list += ['solenoid_artifact', 'mua_envelope']
        if lf_bin is not None:
            feature_list += ['lfp_deflection', 'lick_band']

    # ── Load behavior ──────────────────────────────────────────────────────
    bhv = load_behavior(bhv_file)
    if verbose:
        print(f"[BHV] Loaded {bhv_file.name}: {len(bhv)} rows")

    code_col = bhv.iloc[:, 1] if bhv.shape[1] >= 2 else None
    if code_col is not None:
        trial_rwd_times = bhv['timestamp'][code_col == 3].values
        if len(trial_rwd_times) == 0:
            trial_rwd_times = bhv['timestamp'][code_col == 300].values
    else:
        trial_rwd_times = bhv['timestamp'].values
    if verbose:
        print(f"[BHV] {len(trial_rwd_times)} reward events found")

    # ── Load ground truth from video sync (if provided) ───────────────────
    gt_ephys_times = np.full(len(trial_rwd_times), np.nan)
    if gt_sync_json is not None and Path(gt_sync_json).exists():
        with open(gt_sync_json) as f:
            gt_data = json.load(f)
        warp_gt = gt_data.get('warp', {})
        s_gt = warp_gt.get('stretch', 1.0)
        o_gt = warp_gt.get('offset',  0.0)
        # Convert behavior times → video times (used as GT for ephys)
        gt_ephys_times = s_gt * trial_rwd_times + o_gt
        if verbose:
            print(f"[GT] Loaded ground truth from {Path(gt_sync_json).name}")

    # ── Load ephys data (or restore from cache) ───────────────────────────
    feature_traces = {}
    feature_times  = {}

    # Determine cache path
    session_stem_for_cache = Path(ap_bin or lf_bin or bhv_file).stem
    cache_path = (Path(out_folder) /
                  f"{session_stem_for_cache}_features_cache.npz")

    sync_edge_times_cached = None
    cache_loaded = False

    if use_cache and cache_path.exists():
        if verbose:
            print(f"[CACHE] Loading feature cache: {cache_path.name}")
        try:
            (feature_traces, feature_times,
             sync_edge_times_cached, _, _) = load_feature_cache(cache_path)
            cache_loaded = True
            if verbose:
                print(f"[CACHE] Loaded features: {list(feature_traces.keys())}")
        except Exception as e:
            warnings.warn(f"Cache load failed ({e}), re-extracting from binary.")
            feature_traces, feature_times = {}, {}
            cache_loaded = False

    # --- AP band — chunked loading (handles files >> 10 GB) ---
    if ap_bin is not None and Path(ap_bin).exists():
        if verbose:
            print(f"[AP] Reading {Path(ap_bin).name} …")
        try:
            meta_ap = read_spikeglx_meta(ap_bin)
            n_ch_ap, sr_ap, n_samp_ap, uv_ap = parse_spikeglx_meta(meta_ap)

            # Use all neural channels; exclude last (sync) channel
            if ap_channels is None:
                ap_channels = list(range(n_ch_ap - 1))

            size_gb = len(ap_channels) * n_samp_ap * 2 / 1e9
            if verbose:
                print(f"[AP] {n_ch_ap} ch total, {sr_ap:.0f} Hz, "
                      f"{n_samp_ap/sr_ap/60:.1f} min — using {len(ap_channels)} ch "
                      f"({size_gb:.1f} GB) — chunked @ {chunk_sec}s")

            do_solenoid = 'solenoid_artifact' in feature_list
            do_mua      = 'mua_envelope'      in feature_list

            chunk_samp   = int(chunk_sec * sr_ap)   # samples per core chunk
            overlap_samp = int(0.5 * sr_ap)         # 0.5 s overlap for filter settling
            ds_factor    = max(1, int(sr_ap / 1000.0))  # AP → 1 kHz for MUA

            cm_chunks  = []   # solenoid common-mode (full SR)
            mua_chunks = []   # MUA raw envelope (downsampled, z-score later)

            n_chunks = int(np.ceil(n_samp_ap / chunk_samp))

            for ci in range(n_chunks):
                core_start = ci * chunk_samp
                core_end   = min(n_samp_ap, core_start + chunk_samp)

                load_start = max(0, core_start - overlap_samp)
                load_end   = min(n_samp_ap, core_end + overlap_samp)

                pre_pad  = core_start - load_start   # overlap samples before core
                post_pad = load_end   - core_end      # overlap samples after core

                if verbose:
                    pct = 100.0 * core_end / n_samp_ap
                    print(f"[AP] chunk {ci+1}/{n_chunks}  "
                          f"{core_start/sr_ap/60:.1f}–{core_end/sr_ap/60:.1f} min  "
                          f"({pct:.0f}%)", end='\r', flush=True)

                chunk = load_ephys_chunk(ap_bin, load_start, load_end,
                                         channel_ids=ap_channels)
                chunk = (chunk * uv_ap).astype(np.float32)

                if do_solenoid:
                    # Median across channels at each sample — no temporal filter,
                    # so overlap is only needed for bookkeeping (trim it back).
                    cm = np.median(chunk, axis=0).astype(np.float64)
                    end_idx = len(cm) - post_pad if post_pad > 0 else None
                    cm_chunks.append(cm[pre_pad:end_idx])

                if do_mua:
                    # extract_mua_envelope applies bandpass + medfilt internally.
                    # We use the raw (non-z-scored) output and z-score after assembly.
                    _, _, mua_raw_c = extract_mua_envelope(chunk, sr_ap)
                    pre_ds  = pre_pad  // ds_factor
                    post_ds = post_pad // ds_factor if post_pad > 0 else 0
                    end_idx_ds = len(mua_raw_c) - post_ds if post_ds > 0 else None
                    mua_chunks.append(mua_raw_c[pre_ds:end_idx_ds])

                del chunk  # free memory immediately

            if verbose:
                print()  # newline after \r progress line

            # --- Assemble solenoid feature from chunks ---
            if do_solenoid:
                cm_full = np.concatenate(cm_chunks)
                abs_cm  = np.abs(cm_full)
                mu, sd  = np.nanmean(abs_cm), np.nanstd(abs_cm)
                cm_z    = (abs_cm - mu) / (sd if sd > 0 else 1.0)
                t_art   = np.arange(len(cm_z)) / sr_ap
                min_dist_samp = int(0.1 * sr_ap)
                ev_samp, _ = find_peaks(cm_z, height=8.0,
                                         distance=min_dist_samp)
                feature_traces['solenoid_artifact'] = cm_z
                feature_times['solenoid_artifact']  = t_art
                if verbose:
                    print(f"[FEAT] solenoid: {len(ev_samp)} candidate events")
                del cm_full, cm_chunks

            # --- Assemble MUA feature from chunks ---
            if do_mua:
                mua_full = np.concatenate(mua_chunks)
                mu, sd   = np.nanmean(mua_full), np.nanstd(mua_full)
                mua_z    = (mua_full - mu) / (sd if sd > 0 else 1.0)
                sr_ds    = sr_ap / ds_factor
                mua_t    = np.arange(len(mua_z)) / sr_ds
                feature_traces['mua_envelope'] = mua_z
                feature_times['mua_envelope']  = mua_t
                if verbose:
                    print(f"[FEAT] MUA: {len(mua_t)} samples @ {sr_ds:.0f} Hz")
                del mua_full, mua_chunks

        except Exception as e:
            warnings.warn(f"AP extraction failed: {e}")
            import traceback; traceback.print_exc()

    # --- LF band ---
    if lf_bin is not None and Path(lf_bin).exists():
        if verbose:
            print(f"[LF] Reading {Path(lf_bin).name} …")
        try:
            meta_lf = read_spikeglx_meta(lf_bin)
            n_ch_lf, sr_lf, n_samp_lf, uv_lf = parse_spikeglx_meta(meta_lf)

            if lf_channels is None:
                lf_channels = list(range(n_ch_lf - 1))
            if verbose:
                print(f"[LF] {n_ch_lf} ch, {sr_lf:.0f} Hz, "
                      f"{n_samp_lf/sr_lf:.0f} s — using {len(lf_channels)} ch")

            lf_data = load_ephys_chunk(lf_bin, 0, n_samp_lf,
                                        channel_ids=lf_channels)
            lf_data = (lf_data * uv_lf).astype(np.float32)

            if 'lfp_deflection' in feature_list:
                if verbose:
                    print("[FEAT] Extracting LFP deflection …")
                lf_t, lf_z, lf_raw, lf_exp = extract_lfp_deflection(
                    lf_data, sr_lf)
                feature_traces['lfp_deflection'] = lf_z
                feature_times['lfp_deflection']  = lf_t
                if verbose:
                    print(f"[FEAT] LFP PC1 explained var: "
                          f"{lf_exp[0]*100:.1f}%")

            if 'lick_band' in feature_list:
                if verbose:
                    print("[FEAT] Extracting lick-band power …")
                lk_t, lk_z, lk_raw = extract_lick_band(lf_data, sr_lf)
                feature_traces['lick_band'] = lk_z
                feature_times['lick_band']  = lk_t
                if verbose:
                    print("[FEAT] lick_band done")

        except Exception as e:
            warnings.warn(f"LF extraction failed: {e}")

    if not feature_traces:
        raise RuntimeError("No ephys features could be extracted. "
                           "Check --ap / --lf paths and meta files.")

    # ── Save feature cache if we just extracted (not loaded from cache) ───
    if not cache_loaded and use_cache:
        try:
            save_feature_cache(cache_path, feature_traces, feature_times,
                                sync_edge_times=sync_edge_times_cached)
            if verbose:
                print(f"[CACHE] Saved feature cache → {cache_path.name}")
        except Exception as e:
            warnings.warn(f"Cache save failed: {e}")

    # ── Extract sync channel GT (if AP bin available and not cached) ───────
    sync_warp = None
    if ap_bin is not None and Path(ap_bin).exists():
        try:
            if sync_edge_times_cached is not None and len(sync_edge_times_cached) > 0:
                # Already have edges from cache — re-fit warp from behavior
                if verbose:
                    print("[SYNC] Re-fitting warp from cached sync edges …")
                bhv_event_times = bhv['timestamp'].values.astype(float)
                n_match = min(len(sync_edge_times_cached), len(bhv_event_times))
                sync_warp = fit_sync_warp(
                    bhv_event_times[:n_match],
                    sync_edge_times_cached[:n_match],
                    bounds=(0.9, 1.1), offset_bounds=(-300.0, 300.0))
                reward_mask = (bhv.iloc[:, 1].values == 3) if bhv.shape[1] >= 2 else np.ones(len(bhv), dtype=bool)
                gt_ephys_times = (sync_warp['stretch'] *
                                  bhv['timestamp'].values[reward_mask] +
                                  sync_warp['offset'])
            else:
                gt_ephys_times_sync, sync_edges, sync_warp = extract_sync_channel(
                    ap_bin, bhv, verbose=verbose)
                # Cache the edges for future reruns
                if not cache_loaded and use_cache and len(sync_edges) > 0:
                    save_feature_cache(cache_path, feature_traces, feature_times,
                                        sync_edge_times=sync_edges)
                # Use as GT if sync looks valid (rmse < 50 ms)
                if (sync_warp['n'] > 5 and
                        not np.isnan(sync_warp['rmse']) and
                        sync_warp['rmse'] < 0.05):
                    gt_ephys_times = gt_ephys_times_sync
                    if verbose:
                        print(f"[SYNC] Using sync-channel GT "
                              f"({len(gt_ephys_times)} reward times)")
                else:
                    if verbose:
                        print("[SYNC] Sync channel present but poor fit "
                              f"(rmse={sync_warp.get('rmse', 'n/a')}) — "
                              "no GT available for this session.")
        except Exception as e:
            warnings.warn(f"Sync channel extraction failed: {e}")

    # ── Detect per-trial events ────────────────────────────────────────────
    if verbose:
        print(f"[DET] Detecting events across {len(trial_rwd_times)} trials …")

    events, all_candidates = detect_ephys_events(
        feature_traces, feature_times, trial_rwd_times,
        search_window=(-0.1, 0.5),
        min_confidence=min_conf,
        min_candidates=min_candidates)

    detected_times = np.array(
        [e['time'] if e is not None else np.nan for e in events],
        dtype=float)

    n_detected = int(np.isfinite(detected_times).sum())
    if verbose:
        print(f"[DET] {n_detected}/{len(trial_rwd_times)} trials detected "
              f"(min_conf={min_conf})")

    # ── Fit warp (behavior → ephys time) ──────────────────────────────────
    # Match behavior reward times to detected ephys event times
    bhv_times_for_warp  = trial_rwd_times.copy()
    ephys_times_for_warp = detected_times.copy()

    valid_for_fit = np.isfinite(bhv_times_for_warp) & np.isfinite(ephys_times_for_warp)
    warp = {'stretch': 1.0, 'offset': 0.0, 'rmse': np.nan, 'n': 0}

    if valid_for_fit.sum() >= 3:
        idxs = np.where(valid_for_fit)[0].copy()
        rng  = np.random.RandomState(0)
        rng.shuffle(idxs)
        n_train = max(1, int(len(idxs) * train_frac))
        train_idx, test_idx = idxs[:n_train], idxs[n_train:]

        warp = fit_sync_warp(bhv_times_for_warp[train_idx],
                              ephys_times_for_warp[train_idx],
                              bounds=(0.8, 1.25),
                              offset_bounds=(-100.0, 100.0))
        if verbose:
            print(f"[WARP] stretch={warp['stretch']:.6f}  "
                  f"offset={warp['offset']:.3f}s  "
                  f"rmse={warp['rmse']:.4f}s  "
                  f"n_train={len(train_idx)}")

        # Evaluate on test set
        if len(test_idx) > 0:
            pred  = warp['stretch'] * bhv_times_for_warp[test_idx] + warp['offset']
            true  = ephys_times_for_warp[test_idx]
            test_metrics = evaluate_against_ground_truth(pred, true,
                                                          tolerance_ms=50)
            if verbose:
                print(f"[EVAL] test set metrics: {test_metrics}")
    else:
        if verbose:
            warnings.warn("Not enough matched trials (≥3) for warp fitting.")

    # ── Evaluate against video GT (if available) ──────────────────────────
    gt_metrics = {}
    if np.isfinite(gt_ephys_times).sum() >= 3:
        # Apply warp to predict ephys times from behavior times
        pred_ephys = warp['stretch'] * trial_rwd_times + warp['offset']
        gt_metrics = evaluate_against_ground_truth(
            pred_ephys, gt_ephys_times, tolerance_ms=50)
        if verbose:
            print(f"[GT_EVAL] vs video GT: {gt_metrics}")

    # ── Write canonical sync file ──────────────────────────────────────────
    session_stem = Path(ap_bin or lf_bin or bhv_file).stem
    if make_sync:
        events_map = []
        for ti in range(len(trial_rwd_times)):
            bt = (float(trial_rwd_times[ti])
                  if np.isfinite(trial_rwd_times[ti]) else None)
            et = (float(warp['stretch'] * bt + warp['offset'])
                  if bt is not None else None)
            conf = (events[ti]['confidence']
                    if (events[ti] is not None) else None)
            events_map.append({'trial_idx': int(ti),
                                'behavior_time': bt,
                                'ephys_time': et,
                                'confidence': conf})
        metadata = {
            'bhv_file':     str(bhv_file),
            'ap_bin':       str(ap_bin) if ap_bin else None,
            'lf_bin':       str(lf_bin) if lf_bin else None,
            'features_used': list(feature_traces.keys()),
            'created':       datetime.now().isoformat(),
        }
        sync_out = out_folder / f"sync_ephys_{session_stem}.json"
        write_sync_file(str(sync_out), warp, events_map, metadata)
        if verbose:
            print(f"[SYNC] Wrote {sync_out}")

        # MATLAB-compatible output
        try:
            mat_out = out_folder / f"syncfix_ephys_{session_stem}.mat"
            savemat(str(mat_out), {
                'trial_rwd_times':    trial_rwd_times,
                'detected_ephys_times': detected_times,
                'warp_stretch':       float(warp['stretch']),
                'warp_offset':        float(warp['offset']),
                'warp_rmse':          float(warp['rmse'])
                                      if not np.isnan(warp['rmse']) else -1.0,
            })
            if verbose:
                print(f"[SYNC] Wrote MAT: {mat_out}")
        except Exception as e:
            warnings.warn(f"MAT save failed: {e}")

    # ── Save diagnostics ───────────────────────────────────────────────────
    save_diagnostics(out_folder, session_stem,
                     trial_rwd_times=trial_rwd_times,
                     detected_times=detected_times,
                     gt_ephys_times=gt_ephys_times,
                     warp_bi=np.array([warp['stretch'], warp['offset']]),
                     **{f'trace_{k}': feature_traces[k]
                        for k in feature_traces},
                     **{f'time_{k}': feature_times[k]
                        for k in feature_times})

    return {
        'feature_traces':    feature_traces,
        'feature_times':     feature_times,
        'trial_rwd_times':   trial_rwd_times,
        'detected_times':    detected_times,
        'gt_ephys_times':    gt_ephys_times,
        'warp':              warp,
        'events':            events,
        'gt_metrics':        gt_metrics,
        'session_stem':      session_stem,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ephys-to-behavior temporal alignment (SpikeGLX + Kilosort)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input files
    parser.add_argument('--bhv', required=True,
                        help='Behavior log CSV (GlobalLog*.csv)')
    parser.add_argument('--ap',  default=None,
                        help='SpikeGLX AP band binary (*.ap.bin)')
    parser.add_argument('--lf',  default=None,
                        help='SpikeGLX LF band binary (*.lf.bin)')
    parser.add_argument('--ks',  default=None,
                        help='Kilosort output folder (optional)')
    parser.add_argument('--gt',  default=None,
                        help='Video-pipeline sync JSON for ground-truth comparison')
    parser.add_argument('--out', default=None,
                        help='Output folder')

    # Feature selection
    parser.add_argument('--feature', default='all',
                        choices=['all', 'solenoid_artifact', 'lfp_deflection',
                                 'mua_envelope', 'lick_band'],
                        help='Which ephys feature to extract (all = try all available)')

    # Alignment
    parser.add_argument('--min-conf',   type=float, default=0.3,
                        help='Minimum confidence score to accept an event')
    parser.add_argument('--train-frac', type=float, default=0.8,
                        help='Fraction of GT trials for training')
    parser.add_argument('--make-sync',  action='store_true',
                        help='Write canonical sync JSON + MAT')
    parser.add_argument('--sweep',      action='store_true',
                        help='Run hyperparameter sweep across feature weights')

    # Channel selection
    parser.add_argument('--ap-channels', type=str, default=None,
                        help='Comma-separated AP channel indices (e.g. 0,1,2,…,63)')
    parser.add_argument('--lf-channels', type=str, default=None,
                        help='Comma-separated LF channel indices (default: all)')
    # Hyperparameters for feature extraction and event detection

    
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore and overwrite any existing feature cache.')

    # Memory / quality control
    parser.add_argument('--chunk-sec', type=float, default=60.0,
                        help='Seconds per AP chunk during loading (default 60). '
                             'Reduce if RAM is tight; 60s @ 64ch ≈ 220 MB.')
    parser.add_argument('--min-candidates', type=int, default=2,
                        help='Minimum number of candidates per trial to trust '
                             'confidence score. Trials with fewer candidates are '
                             'rejected (confidence 1.0 from single candidate is '
                             'uninformative). Default 2.')

    args = parser.parse_args()

    # Parse channel lists
    def parse_channels(s):
        if s is None:
            return None
        return [int(c.strip()) for c in s.split(',') if c.strip()]

    ap_ch = parse_channels(args.ap_channels)
    lf_ch = parse_channels(args.lf_channels)

    # Feature list
    if args.feature == 'all':
        feat_list = None   # process_one_session will auto-detect
    else:
        feat_list = [args.feature]

    # Run
    session_res = process_one_session(
        bhv_file       = args.bhv,
        ap_bin         = args.ap,
        lf_bin         = args.lf,
        ks_folder      = args.ks,
        out_folder     = args.out,
        gt_sync_json   = args.gt,
        feature_list   = feat_list,
        make_sync      = args.make_sync,
        train_frac     = args.train_frac,
        min_conf       = args.min_conf,
        ap_channels    = ap_ch,
        lf_channels    = lf_ch,
        chunk_sec      = args.chunk_sec,
        min_candidates = args.min_candidates,
        use_cache      = not args.no_cache,
    )

    if args.sweep:
        out_folder = args.out or str(Path(args.bhv).parent)
        csv_path = hyperparameter_sweep(
            session_res, out_folder, session_res['session_stem'],
            train_frac=args.train_frac)
        print(f"[SWEEP] Results saved → {csv_path}")