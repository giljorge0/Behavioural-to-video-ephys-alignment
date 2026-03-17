#!/usr/bin/env python3
"""
Main improvements + simple hyperparameter sweep for video-only sync.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from skimage.feature import match_template
from scipy.signal import medfilt, find_peaks
from scipy.optimize import minimize
from scipy.optimize import least_squares
import json
from scipy.interpolate import interp1d
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import warnings
import argparse
import math
import itertools
import csv
from datetime import datetime

# Try to import tqdm for progress bars; fallback to simple printing
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False

# ----------------------- small helpers -----------------------
def normalize01(x, lo_perc=5.0, hi_perc=95.0, clip_extremes=True):
    """
    Robust normalization to [0,1] using percentile clipping.

    - x : array-like
    - lo_perc, hi_perc : percentiles to use for scaling (default 5th..95th)
    - clip_extremes : if True, values outside percentiles are clipped to percentiles
    Returns: array same shape as x, with NaNs preserved.
    """
    x = np.asarray(x, dtype=float)
    # preserve NaNs
    nan_mask = np.isnan(x)
    valid = x[~nan_mask]
    if valid.size == 0:
        return np.full_like(x, np.nan)

    if clip_extremes:
        lo = np.percentile(valid, lo_perc)
        hi = np.percentile(valid, hi_perc)
        if hi == lo:
            # fallback to min/max if percentiles degenerate
            lo = np.nanmin(valid)
            hi = np.nanmax(valid)
        clipped = np.clip(valid, lo, hi)
    else:
        lo = np.nanmin(valid)
        hi = np.nanmax(valid)
        clipped = valid.copy()

    # avoid division by zero
    if hi == lo:
        out_valid = np.zeros_like(clipped)
    else:
        out_valid = (clipped - lo) / (hi - lo)

    out = np.empty_like(x)
    out.fill(np.nan)
    out[~nan_mask] = out_valid
    return out

def odd_int(x):
    x = int(x)
    if x % 2 == 0:
        x += 1
    return max(1, x)

def safe_medfilt(x, wf):
    wf = max(1, int(wf))
    wf = odd_int(wf)
    try:
        return medfilt(x, kernel_size=wf)
    except Exception:
        # fallback naive median
        out = np.copy(x)
        half = wf // 2
        for i in range(len(x)):
            a = max(0, i-half)
            b = min(len(x), i+half+1)
            out[i] = np.nanmedian(x[a:b])
        return out

def tempwarperrfun(B, t, exp_vid, exp_bhv):
    s, o = float(B[0]), float(B[1])
    if s == 0:
        s = 1.0
    # warp behavior into video clock
    b = exp_bhv
    vt = s * t + o
    # simple squared error between convolved traces sampled at t
    # resample exp_vid to the warped t positions (approx)
    try:
        f = interp1d(t, exp_vid, bounds_error=False, fill_value=np.nan)
        v_at_b = f(vt)
        res = np.nan_to_num(v_at_b - b)
        return np.nansum(res**2)
    except Exception:
        return 1e9

def tempwarp(B, t, exp_bhv):
    s, o = float(B[0]), float(B[1])
    return np.interp(t, s * t + o, exp_bhv, left=np.nan, right=np.nan)

def subtracttime(time, x, s):
    """subtract s seconds according to time lookup: x - sampled shift.
    if s is scalar we subtract s; otherwise we align per-frame using nearest index.
    """
    time = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    if np.isscalar(s):
        return x - s, np.full_like(x, s)
    # s is per-frame: assume same length as time
    y = np.asarray(s, dtype=float)
    if len(y) != len(time):
        # try interpolation
        f = interp1d(time, y, bounds_error=False, fill_value='extrapolate')
        y = f(time)
    zero_flags = (y == 0)
    idcs_clamped = np.searchsorted(time, y)
    idcs_clamped[zero_flags] = 1
    idcs_clamped = np.clip(idcs_clamped - 1, 0, max(0, len(y)-1))
    s = y[idcs_clamped].astype(float)
    s[zero_flags] = np.nan
    x_adj = x - s
    return x_adj, s

# ----------------------- main pipeline functions -----------------------
def load_behavior(bhv_file):
    """
    Load behavioral csv into a pandas DataFrame with named columns.
    Will try to infer whether timestamps are seconds or ms (simple heuristic).
    """
    df = pd.read_csv(bhv_file, header=None)
    df.columns = ['timestamp', 'code'] if df.shape[1] >= 2 else ['timestamp']
    # normalize to session start using event code 1000 (session ready)
    # this matches the MATLAB original: event_times - session_ready_time
    if df.shape[1] >= 2:
        codes = df.iloc[:, 1]
        session_ready_mask = codes == 1000
        if session_ready_mask.any():
            session_ready_time = df['timestamp'][session_ready_mask].iloc[0]
            df['timestamp'] = df['timestamp'] - session_ready_time
    return df

def load_sync(sync_file):
    """
    load sync CSV – expects at least three columns [flag, frameid, timestamp]
    returns Nx3 numpy array
    """

    try:
        # camera timestamp file is space-separated and has no header
        df = pd.read_csv(sync_file, sep=r"\s+", header=None)

        if df.shape[1] < 3:
            raise ValueError("sync file must have at least 3 columns")

        # convert to numpy to match rest of code
        s = df.iloc[:, :3].values.astype(float)

        return s

    except Exception as e:
        raise


def sample_templates(cap, rois, start_time, stop_time, fps, sample_max_frames=200, resize=1.0):
    """
    Grab a small sample from the video to create grayscale/resized template crops for each ROI.
    Returns templates dict keyed by roi index (grayscale uint8).
    """
    res = {}
    # position to start
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
    n = int(min(sample_max_frames, max(1, round((stop_time - start_time) * fps))))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Couldn't read frame to sample templates")
    # convert to grayscale and optionally resize
    if resize != 1.0:
        neww = max(1, int(frame.shape[1] * resize))
        newh = max(1, int(frame.shape[0] * resize))
        frame = cv2.resize(frame, (neww, newh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i, r in enumerate(rois):
        x0, y0, x1, y1 = r['box']
        # scale ROI coordinates if resizing
        if resize != 1.0:
            x0 = int(round(x0 * resize)); x1 = int(round(x1 * resize))
            y0 = int(round(y0 * resize)); y1 = int(round(y1 * resize))
        # safety clamp
        x0 = max(0, min(x0, gray.shape[1]-1)); x1 = max(0, min(x1, gray.shape[1]))
        y0 = max(0, min(y0, gray.shape[0]-1)); y1 = max(0, min(y1, gray.shape[0]))
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            # fallback tiny template
            crop = gray[0:1, 0:1]
        res[i] = crop.copy()
    return res

def extract_roi_scores(cap, rois, templates, frame_count, width, height,
                       use_tqdm=True, frame_step=10, search_margin=50, resize=1.0):
    """
    Fast ROI scoring with optimized seeking and interpolation.
    """
    n_rois = len(rois)
    # 1. Define the sampled indices (the "tiny change" is here)
    sampled_inds = np.arange(0, frame_count, frame_step, dtype=int)
    n_sampled = len(sampled_inds)

    # Arrays to store results for only the frames we actually look at
    roi_scores_sampled = np.full((n_rois, n_sampled), np.nan, dtype=float)
    roi_coords_sampled = np.full((n_rois, n_sampled, 2), np.nan, dtype=float)

    # 2. The Main Loop: Jumps through the video
    it = enumerate(sampled_inds)
    if use_tqdm and _use_tqdm:
        it = tqdm(it, total=n_sampled, desc="Processing sampled frames")

    for si, frame_idx in it:
        # THE FIX: Explicitly jump the 'needle' of the record player to the right spot
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        
        if not ret:
            # If we hit a bad frame, try to skip it and keep going
            continue

        # Resize the whole frame once (if requested) to speed up everything else
        if resize != 1.0:
            neww = max(1, int(frame.shape[1] * resize))
            newh = max(1, int(frame.shape[0] * resize))
            frame = cv2.resize(frame, (neww, newh), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, r in enumerate(rois):
            # Get ROI and expand it by the margin
            x0, y0, x1, y1 = r['box']
            if resize != 1.0:
                x0, x1 = int(x0 * resize), int(x1 * resize)
                y0, y1 = int(y0 * resize), int(y1 * resize)

            # Create the small search window
            sx0, sy0 = max(0, x0 - search_margin), max(0, y0 - search_margin)
            sx1, sy1 = min(gray.shape[1], x1 + search_margin), min(gray.shape[0], y1 + search_margin)
            
            search_region = gray[sy0:sy1, sx0:sx1]
            tpl = templates.get(i, None)
            
            if tpl is None or search_region.size == 0:
                continue

            try:
                # Use the fastest C++ implementation of template matching
                res = cv2.matchTemplate(search_region, tpl, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                
                # Store the result
                roi_scores_sampled[i, si] = float(maxv)
                # Convert local search box coords back to full image coords
                roi_coords_sampled[i, si, :] = (float(sy0 + maxloc[1]), float(sx0 + maxloc[0]))
            except Exception:
                continue

    # 3. FILLING THE GAPS: Interpolate the 90% of frames we skipped
    roi_scores_full = np.full((n_rois, frame_count), np.nan, dtype=float)
    roi_coords_full = np.full((n_rois, frame_count, 2), np.nan, dtype=float)
    full_idx = np.arange(frame_count)

    for i in range(n_rois):
        # Interpolate the Scores (PC1 signal)
        vals = roi_scores_sampled[i, :]
        mask = np.isfinite(vals)
        if np.sum(mask) >= 2:
            roi_scores_full[i, :] = np.interp(full_idx, sampled_inds[mask], vals[mask])
        
        # Interpolate the Coords (X and Y)
        for c in range(2):
            c_vals = roi_coords_sampled[i, :, c]
            c_mask = np.isfinite(c_vals)
            if np.sum(c_mask) >= 2:
                roi_coords_full[i, :, c] = np.interp(full_idx, sampled_inds[c_mask], c_vals[c_mask])

    return roi_scores_full, roi_coords_full

def compute_pc1(Xs):
    """
    compute PCA PC1 over vectorized ROI templates or frames.
    """
    sc = StandardScaler()
    Xs_s = sc.fit_transform(Xs)
    pca = PCA(n_components=min(10, Xs_s.shape[1]))
    Z = pca.fit_transform(Xs_s)  # frames x PCs
    explained = pca.explained_variance_ratio_
    pc1 = Z[:,0]
    return pc1, explained, pca.components_.T

def fit_time_warp(t, exp_vid, exp_bhv, bounds=(0.9, 1.1), offset_bounds=None):
    """
    Fit linear warp bi = [stretch, offset] with constrained optimizer.
    bounds for stretch are provided; offset bounds in seconds optional.
    Uses L-BFGS-B (derivative-free minimize uses gradient if available).
    """
    x0 = np.array([1.0, 0.0])
    bnds = [(bounds[0], bounds[1])]
    if offset_bounds is None:
        offset_bounds = (-5.0, 5.0)
    bnds.append((offset_bounds[0], offset_bounds[1]))
    res = minimize(lambda B: tempwarperrfun(B, t, exp_vid, exp_bhv),
                   x0=x0, method='L-BFGS-B', bounds=bnds,
                   options={'maxiter': 5000})
    return res

def infer_reach_times_from_trace(time, droplet_trace, trial_rwd_times_hat, derivative_quantile=0.99):
    """
    Find falling-edge-based reach times within intervals defined by trial_rwd_times_hat.
    Returns inferred reach times array (len equals number of trials).
    """
    droplet_trace_derivative = np.concatenate(([0.0], np.diff(droplet_trace)))
    derivative_threshold = -np.quantile(np.abs(droplet_trace_derivative[np.isfinite(droplet_trace_derivative)]), derivative_quantile)
    trial_count = len(trial_rwd_times_hat)
    trial_reach_times_inferred = np.full(trial_count, np.nan)
    for tt in range(trial_count):
        start_t = trial_rwd_times_hat[tt]
        end_t = trial_rwd_times_hat[tt+1] if tt < trial_count-1 else np.inf
        flags = (time >= start_t) & (time < end_t)
        valid_flags = flags & (droplet_trace_derivative < derivative_threshold)
        if np.sum(valid_flags) == 0:
            continue
        temp = droplet_trace_derivative.copy()
        temp[~valid_flags] = np.nan
        idx = int(np.nanargmin(temp))
        trial_reach_times_inferred[tt] = time[idx]
    return trial_reach_times_inferred

# ---------- ADDED HELPERS: droplet detection, warp fit, sync writer, evaluation ----------
def detect_droplet_events(time, pc1_z, roi_scores, roi_coords, trial_rwd_times,
                          search_window=(0.0, 2.0), weights=None, min_confidence=0.3):
    """
    Per-trial combined droplet detector returning selected event per trial and all candidates.
    Uses an ensemble "voting" system with PC1 peaks, template match scores, 
    coordinate validity, and isolation.
    """
    # Define feature weights for the ensemble vote
    if weights is None:
        weights = {'pc1': 1.0, 'tpl': 0.0, 'coord': 0.0, 'isolation': 0.0}

    nframes = len(time)
    tpl_scores_max = np.nanmax(roi_scores, axis=0) if (roi_scores is not None) else np.zeros(nframes)
    best_roi_idx = np.nanargmax(roi_scores, axis=0) if (roi_scores is not None) else np.zeros(nframes, dtype=int)

    # 1. Detect candidate triggers: Negative PC1 peaks (dips in brightness)
    neg_pc1 = -np.nan_to_num(pc1_z)
    peaks_all, _ = find_peaks(neg_pc1, distance=1)
    peaks_set = set(peaks_all.tolist())

    trial_count = len(trial_rwd_times)
    events = [None] * trial_count
    all_candidates = [None] * trial_count

    # 2. Iterate through each trial to find the best candidate
    for tt in range(trial_count):
        start = trial_rwd_times[tt] + search_window[0]
        end = trial_rwd_times[tt] + search_window[1]
        flags = (time >= start) & (time <= end)
        frame_inds = np.where(flags)[0]
        candidates = []

        # PC1-based candidates
        pc1_cands = [int(f) for f in frame_inds if int(f) in peaks_set]
        for fi in pc1_cands:
            candidates.append({'frame': int(fi), 'pc1': float(pc1_z[fi]),
                               'tpl': float(tpl_scores_max[fi]) if tpl_scores_max is not None else 0.0,
                               'coords': tuple(roi_coords[best_roi_idx[fi], fi]) if roi_coords is not None else (np.nan, np.nan)})

        # Template-local-max candidates
        if roi_scores is not None:
            for fi in frame_inds:
                if fi <= 0 or fi >= nframes-1:
                    continue
                if tpl_scores_max[fi] > tpl_scores_max[fi-1] and tpl_scores_max[fi] >= tpl_scores_max[fi+1] and tpl_scores_max[fi] > 0:
                    candidates.append({'frame': int(fi), 'pc1': float(pc1_z[fi]),
                                       'tpl': float(tpl_scores_max[fi]),
                                       'coords': tuple(roi_coords[best_roi_idx[fi], fi])})

        # deduplicate
        uniq = {}
        for c in candidates:
            uniq[c['frame']] = c
        candidates = list(uniq.values())

        # compute isolation and coord_ok
        frames = np.array([c['frame'] for c in candidates]) if candidates else np.array([])
        for i, c in enumerate(candidates):
            coord_ok = 0.0 if np.any(np.isnan(c.get('coords', (np.nan, np.nan)))) else 1.0
            if frames.size > 1:
                dists = np.abs(frames - c['frame'])
                nearest = np.min(dists[dists > 0]) if np.any(dists > 0) else np.inf
                isolation = float(min(1.0, nearest / 10.0))
            else:
                isolation = 1.0
            c['coord_ok'] = coord_ok
            c['isolation'] = isolation

        # Normalize features and compute score
        if candidates:
            pc1s = np.array([c['pc1'] for c in candidates], dtype=float)
            tpls = np.array([c['tpl'] for c in candidates], dtype=float)
            isols = np.array([c['isolation'] for c in candidates], dtype=float)
            coord_ok = np.array([c['coord_ok'] for c in candidates], dtype=float)

            pc1n = normalize01(pc1s, lo_perc=5.0, hi_perc=95.0, clip_extremes=True)
            tpln = normalize01(tpls, lo_perc=5.0, hi_perc=95.0, clip_extremes=True)
            isoln = normalize01(isols, lo_perc=5.0, hi_perc=95.0, clip_extremes=True)
            coordn = coord_ok

            scores = weights.get('pc1',0.4)*pc1n + weights.get('tpl',0.4)*tpln + weights.get('coord',0.1)*coordn + weights.get('isolation',0.1)*isoln

            for ii, c in enumerate(candidates):
                c['score'] = float(scores[ii])
                c['time'] = float(time[c['frame']])

            best_idx = int(np.nanargmax(scores))
            best = candidates[best_idx]
            if best['score'] >= min_confidence:
                events[tt] = {'trial_idx': tt, 'time': float(best['time']), 'frame': int(best['frame']), 'score': float(best['score']), 'method': 'combined'}
            else:
                events[tt] = None
        else:
            events[tt] = None

        all_candidates[tt] = candidates

    return events, all_candidates

def fit_sync_warp(behavior_times, video_times, bounds=(0.9, 1.1), offset_bounds=(-5.0, 5.0)):
    """Fit linear mapping video_time = s * behavior_time + o using robust least squares (Huber)."""
    behavior_times = np.asarray(behavior_times, dtype=float)
    video_times = np.asarray(video_times, dtype=float)
    mask = np.isfinite(behavior_times) & np.isfinite(video_times)
    if np.sum(mask) < 2:
        return {'stretch': 1.0, 'offset': 0.0, 'residuals': np.array([]), 'rmse': np.nan, 'n': int(np.sum(mask)), 'opt_res': None}
    x = behavior_times[mask]
    y = video_times[mask]

    def fun(p):
        s, o = p
        return y - (s * x + o)

    p0 = np.array([1.0, 0.0])
    lb = [bounds[0], offset_bounds[0]]
    ub = [bounds[1], offset_bounds[1]]
    res = least_squares(fun, p0, loss='huber', bounds=(lb, ub), max_nfev=5000)
    s, o = float(res.x[0]), float(res.x[1])
    residuals = fun(res.x)
    rmse = float(np.sqrt(np.nanmean(residuals**2))) if residuals.size > 0 else np.nan
    return {'stretch': s, 'offset': o, 'residuals': residuals, 'rmse': rmse, 'n': int(np.sum(mask)), 'opt_res': res}

def write_sync_file(path, warp, events_map, metadata=None):
    out = {
        'warp': {'stretch': float(warp.get('stretch', 1.0)), 'offset': float(warp.get('offset', 0.0)), 'rmse': float(warp.get('rmse', np.nan))},
        'events': events_map,
        'metadata': metadata or {}
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    return path

def evaluate_against_ground_truth(pred_events, true_events, tolerance_ms=50):
    pred = np.asarray(pred_events, dtype=float)
    true = np.asarray(true_events, dtype=float)
    m = np.isfinite(true)
    if np.sum(m) == 0:
        return {'n': 0}
    pred_sub = pred[m]
    true_sub = true[m]
    abs_err = np.full(len(true_sub), np.nan)
    finite_mask = np.isfinite(pred_sub)
    abs_err[finite_mask] = np.abs(pred_sub[finite_mask] - true_sub[finite_mask])
    mae = float(np.nanmean(abs_err))
    rmse = float(np.sqrt(np.nanmean(np.square(abs_err))))
    tol = tolerance_ms / 1000.0
    tp = int(np.nansum((abs_err <= tol) & np.isfinite(abs_err)))
    fn = int(np.nansum(~np.isfinite(pred_sub) & np.isfinite(true_sub)))
    fp = int(np.nansum(np.isfinite(pred_sub) & ~np.isfinite(true_sub)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    f1 = float(2 * precision * recall / (precision + recall)) if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0) else np.nan
    return {'n': int(len(true_sub)), 'mae': mae, 'rmse': rmse, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

# ---------- SIMPLE HYPERPARAMETER SWEEP ----------
def hyperparameter_sweep(session_dict, out_folder, video_stem, train_frac=0.8,
                         minconf_list=None, pc1w_list=None, seed=0, tolerance_ms=50):
    """
    session_dict must contain:
      'time', 'pc1_z', 'roi_scores', 'roi_coords', 'trial_rwd_times', 'trial_reach_times_inferred_behavior'
    This function sweeps (min_conf, pc1_weight) grid and records metrics.
    Saves CSV results to out_folder/hp_sweep_<video_stem>.csv
    """
    if minconf_list is None:
        minconf_list = [0.2, 0.3, 0.35, 0.4, 0.5]
    if pc1w_list is None:
        pc1w_list = [0.3, 0.45, 0.6]

    time = session_dict['time']
    pc1_z = session_dict['pc1_z']
    roi_scores = session_dict['roi_scores']
    roi_coords = session_dict['roi_coords']
    trial_rwd_times = np.asarray(session_dict['trial_rwd_times'], dtype=float)
    vid_times_behavior = np.asarray(session_dict['trial_reach_times_inferred_behavior'], dtype=float)

    rows = []
    rng = np.random.RandomState(seed)

    for min_conf in minconf_list:
        for pc1_w in pc1w_list:
            tpl_w = max(0.0, 1.0 - pc1_w)
            weights = {'pc1': float(pc1_w), 'tpl': float(tpl_w), 'coord': 0.05, 'isolation': 0.05}
            events, candidates = detect_droplet_events(time, pc1_z, roi_scores, roi_coords,
                                                       trial_rwd_times, search_window=(0.0, 2.0),
                                                       weights=weights, min_confidence=float(min_conf))
            bhv_times = trial_rwd_times.copy()
            vid_times = vid_times_behavior.copy()
            valid_mask = np.isfinite(bhv_times) & np.isfinite(vid_times)
            n_valid = int(np.sum(valid_mask))
            metrics = {'n': n_valid, 'mae': np.nan, 'rmse': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan}
            warp_info = {'stretch': np.nan, 'offset': np.nan, 'rmse': np.nan, 'n_train': 0, 'n_test': 0}

            if n_valid >= 3:
                idxs = np.where(valid_mask)[0]
                rng.shuffle(idxs)
                n_train = max(1, int(len(idxs) * float(train_frac)))
                train_idx = idxs[:n_train]
                test_idx = idxs[n_train:]
                warp = fit_sync_warp(bhv_times[train_idx], vid_times[train_idx], bounds=(0.95, 1.05), offset_bounds=(-5.0, 5.0))
                pred_vid = warp['stretch'] * bhv_times[test_idx] + warp['offset']
                true_vid = vid_times[test_idx]
                metrics = evaluate_against_ground_truth(pred_vid, true_vid, tolerance_ms=int(tolerance_ms))
                warp_info.update({'stretch': float(warp.get('stretch', np.nan)), 'offset': float(warp.get('offset', np.nan)), 'rmse': float(warp.get('rmse', np.nan)), 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx))})
            else:
                # not enough valid pairs
                metrics = {'n': n_valid, 'mae': np.nan, 'rmse': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan}

            rows.append({
                'min_conf': float(min_conf), 'pc1_w': float(pc1_w), 'tpl_w': float(tpl_w),
                'n_valid_pairs': int(n_valid),
                'n_train': int(warp_info.get('n_train', 0)), 'n_test': int(warp_info.get('n_test', 0)),
                'warp_stretch': float(warp_info.get('stretch', np.nan)), 'warp_offset': float(warp_info.get('offset', np.nan)), 'warp_rmse': float(warp_info.get('rmse', np.nan)),
                'mae': float(metrics.get('mae', np.nan)), 'rmse': float(metrics.get('rmse', np.nan)),
                'precision': float(metrics.get('precision', np.nan)), 'recall': float(metrics.get('recall', np.nan)), 'f1': float(metrics.get('f1', np.nan))
            })

    # save CSV
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    csv_path = out_folder / f"hp_sweep_{video_stem}.csv"
    keys = ['min_conf','pc1_w','tpl_w','n_valid_pairs','n_train','n_test','warp_stretch','warp_offset','warp_rmse','mae','rmse','precision','recall','f1']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})

    # print best by mae (lowest)
    valid_rows = [r for r in rows if not np.isnan(r['mae'])]
    if len(valid_rows) > 0:
        best = min(valid_rows, key=lambda x: x['mae'])
        print(f"[SWEEP DONE] best (by mae) min_conf={best['min_conf']}, pc1_w={best['pc1_w']}, mae={best['mae']:.4f}, rmse={best['rmse']:.4f}, precision={best['precision']}, recall={best['recall']}")
    else:
        print("[SWEEP DONE] no valid configurations produced train/test metrics (too few matched trials).")

    return str(csv_path)

# ---------- diagnostics saving ----------
def save_diagnostics(out_folder, name_prefix, **arrays):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    # save CSV for main arrays if present
    if 'time' in arrays and 'pc1_z' in arrays:
        df = pd.DataFrame({'time': arrays['time'], 'pc1_z': arrays['pc1_z']})
        csvp = out_folder / f"{name_prefix}_signal.csv"
        df.to_csv(csvp, index=False)
    # save npz for arrays
    npzpath = out_folder / f"{name_prefix}_diagnostics.npz"
    try:
        np.savez_compressed(str(npzpath), **arrays)
        if 'pc1_z' in arrays and 'roi_scores' in arrays:
            savemat(str(out_folder / f"{name_prefix}_diagnostics.mat"), {'pc1_z': arrays['pc1_z'], 'roi_scores': arrays['roi_scores']})
    except Exception as e:
        warnings.warn(f"Failed saving diagnostics npz/mat: {e}")
    return npzpath

# ----------------------- orchestrator / main -----------------------
def process_one_session(bhv_file, video_file, sync_file, out_folder=None,
                        expected_fps=None, interactive_roi=False, verbose=True,
                        make_sync=False, train_frac=0.8, min_conf=0.35,
                        frame_step=10, search_margin=50, resize=1.0, roi=None):
    bhv_file = Path(bhv_file)
    video_file = Path(video_file)
    sync_file = Path(sync_file)
    out_folder = Path(out_folder) if out_folder is not None else bhv_file.parent

    # load behavior
    bhv = load_behavior(bhv_file)
    if verbose:
        print(f"Loaded behavior file {bhv_file}; {len(bhv)} rows")

    # load video
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_file}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or np.nan
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) or 0
    duration = frame_count / fps if fps and frame_count else 0.0
    if verbose:
        print(f"Video: {video_file} {width}x{height} fps={fps:.2f} frames={frame_count} dur~{duration:.2f}s")
    if expected_fps is not None and not math.isnan(fps):
        if abs(fps - expected_fps) / expected_fps > 0.05:
            warnings.warn(f"fps mismatch: video {fps} expected {expected_fps}")

    # load sync
    sync = load_sync(sync_file)
    sync_frame_idcs = sync[:,1] - sync[0,1]
    sync_frame_timestamps = sync[:,2] - sync[0,2]

    # create robust frame_time by normalizing sync timestamps then scaling to duration
    frame_time_norm = normalize01(sync_frame_timestamps)
    transition_flags = np.concatenate(([False], np.diff(frame_time_norm) < 0))
    transition_idcs = np.where(transition_flags)[0]
    time = np.linspace(0.0, duration, frame_count) if frame_count > 0 else np.array([])

    # build ROIs (and optionally ask user to draw new ones interactively)
    if roi is not None:
        x0,y0,x1,y1 = roi
        rois = [{'box':(x0,y0,x1,y1),'center':((x0+x1)//2,(y0+y1)//2)}]
    else:
        rois = [{'box':(118,280,162,314),'center':(140,297)}]
    if False:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr_frame = 0

        # helper: compute a display-scaled frame and scale factor
        def make_preview(frame, max_width=1200, max_height=800):
            h, w = frame.shape[:2]
            scale = min(1.0, float(max_width) / w, float(max_height) / h)
            if scale != 1.0:
                disp = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                disp = frame.copy()
            return disp, scale

        cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
        rois = []  # will store chosen ROI boxes as dicts
        while True:
            # seek to curr_frame and read
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame))
            ret, frame0 = cap.read()
            if not ret:
                print(f"Reached end or cannot read frame {curr_frame}.")
                # clamp and continue or break
                curr_frame = max(0, min(curr_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)))
                break

            disp, scale = make_preview(frame0)
            # overlay status instructions
            instr = "n/N=+500/+50, p/P=-500/-50, >/< = +1/-1, s=select ROI, c=clear ROI, q=quit"
            txt = f"Frame: {curr_frame}  |  {instr}"
            cv2.putText(disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            # show previously selected ROI on preview (if any)
            if rois:
                b = rois[-1]['box']  # box in original coords
                x0, y0, x1, y1 = b
                # scale to preview coords
                sx0, sy0 = int(round(x0 * scale)), int(round(y0 * scale))
                sx1, sy1 = int(round(x1 * scale)), int(round(y1 * scale))
                cv2.rectangle(disp, (sx0, sy0), (sx1, sy1), (0,255,255), 2)

            cv2.imshow("ROI Picker", disp)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('n'):
                curr_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), curr_frame + 500)
            elif key == ord('N'):
                curr_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), curr_frame + 50)
            elif key == ord('p'):
                curr_frame = max(0, curr_frame - 500)
            elif key == ord('P'):
                curr_frame = max(0, curr_frame - 50)
            elif key == ord('>') or key == ord('.'):
                curr_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), curr_frame + 1)
            elif key == ord('<') or key == ord(','):
                curr_frame = max(0, curr_frame - 1)
            elif key == ord('s'):
                # call selectROI on the preview image (user draws on preview)
                # ensure the ROI window has focus - this call blocks until user finishes the selection
                r = cv2.selectROI("ROI Picker", disp, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("ROI Picker")  # selectROI may open its own window; close and re-open next loop
                if r is None or r == (0,0,0,0):
                    # user canceled selection - continue browsing
                    cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
                    continue
                rx, ry, rw, rh = r
                # map ROI back to original frame coords (undo scale)
                ox0 = int(round(rx / scale))
                oy0 = int(round(ry / scale))
                ox1 = int(round((rx + rw) / scale))
                oy1 = int(round((ry + rh) / scale))
                # clamp bounds
                ox0 = max(0, min(ox0, frame0.shape[1]-1))
                oy0 = max(0, min(oy0, frame0.shape[0]-1))
                ox1 = max(0, min(ox1, frame0.shape[1]))
                oy1 = max(0, min(oy1, frame0.shape[0]))
                rois = [{'box': (ox0, oy0, ox1, oy1), 'center': (int((ox0+ox1)/2), int((oy0+oy1)/2))}]
                print("Selected ROI (orig coords):", rois[0]['box'])
                # reopen window for potential further operations
                cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
                # break if you want to stop after selecting one ROI
                break
            elif key == ord('c'):
                rois = []
                print("Cleared ROI.")
                cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
            elif key == ord('q') or key == 27:  # ESC
                print("Exiting interactive ROI selection.")
                cv2.destroyWindow("ROI Picker")
                break
            else:
                # unrecognized key — just continue loop
                cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
                continue

        # If we have no ROI at the end, keep default full-frame ROI (or handle as you do)
        if not rois:
            print("No ROI selected; using default full-frame ROI.")
            rois = [{'box': (0, 0, frame0.shape[1], frame0.shape[0]), 'center': (frame0.shape[1]//2, frame0.shape[0]//2)}]
        # close any open windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
# ---------- end interactive ROI selector ----------

    # sample templates in a small sample window; use first few seconds or provided window
    start_time = 0.0
    stop_time = min(duration, 2.0)
    templates = sample_templates(cap, rois, start_time, stop_time, fps, sample_max_frames=200)

    # iterate full video to compute roi scores and coords
    roi_scores, roi_coords = extract_roi_scores(cap, rois, templates, frame_count, width, height,
                                                use_tqdm=True)

    # compute PC1 and normalize
    if roi_scores is not None and roi_scores.size > 0:
        Xs = roi_scores.T  # frames x rois
        pc1, explained, _ = compute_pc1(Xs)
    else:
        pc1 = np.zeros(len(time))
        explained = []

    # smoothing: median filter with safe odd window (half-fps)
    wf = odd_int(max(1, int(round(fps / 2.0))))
    pc1_smooth = safe_medfilt(pc1, wf)
    # normalize and z-score (important to allow universal thresholds)
    pc1_norm = normalize01(pc1_smooth)
    pc1_z = (pc1_norm - np.nanmean(pc1_norm)) / (np.nanstd(pc1_norm) if np.nanstd(pc1_norm) > 0 else 1.0)

    # extract event times from behavior file - matches MATLAB exactly
    code_col = bhv.iloc[:, 1]
    # session_start_time: event 1001 (SessionStarted) - already relative to session_ready (1000)
    session_start_mask = code_col == 1001
    session_start_time = float(bhv['timestamp'][session_start_mask].iloc[0]) if session_start_mask.any() else 0.0
    print(f"session_start_time: {session_start_time:.3f}s")

    # trial reward times: code 3 only (matches MATLAB trial_rwd_times)
    trial_rwd_times = bhv['timestamp'][code_col == 3].values
    if len(trial_rwd_times) == 0:
        trial_rwd_times = bhv['timestamp'][code_col == 300].values

    # all reward times: codes 3 OR 300 (matches MATLAB all_rwd_times)
    all_rwd_times = bhv['timestamp'][(code_col == 3) | (code_col == 300)].values

    # time_flags: exclude pre-session frames (matches MATLAB time_flags = time > session_start_time)
    time_flags = (time > session_start_time) & np.isfinite(time)

    # envelope/rising edges based on quantile - only over valid session time
    envelope_threshold = np.quantile(pc1_norm[time_flags], 0.9) if np.sum(time_flags) > 0 else 0.9
    droplet_envelope = pc1_norm > envelope_threshold

    # find rising edges in the envelope (boolean to int)
    rising_edges = np.zeros_like(pc1_norm, dtype=bool)
    rising_edges[1:] = (pc1_norm[1:] > envelope_threshold) & (pc1_norm[:-1] <= envelope_threshold)

    # convolution kernel (exponential) - matches MATLAB expkernel(mu=5, binwidth=1/fps)
    mu = 5.0
    binwidth = 1.0 / fps if fps and fps > 0 else 0.01
    n_samples = max(1, int(round(min(10.0 * fps, len(time)))))
    t_kernel = np.arange(0, n_samples * binwidth, binwidth)
    lam = 1.0 / mu if mu > 0 else 1.0/5.0
    kernel_pdf = lam * np.exp(-lam * t_kernel)
    kernel_pdf = kernel_pdf / (kernel_pdf.sum() if kernel_pdf.sum() > 0 else 1.0)

    t = time[time_flags]
    # build all_rwd_counts histogram - matches MATLAB: histcounts(all_rwd_times, bin_edges)
    bin_edges_hist = np.linspace(0.0, duration, frame_count + 1)
    all_rwd_counts_full, _ = np.histogram(all_rwd_times, bins=bin_edges_hist)
    all_rwd_counts = all_rwd_counts_full[time_flags]
    exp_vid = np.convolve(rising_edges[time_flags].astype(float), kernel_pdf, mode='same')
    exp_bhv = np.convolve(all_rwd_counts.astype(float), kernel_pdf, mode='same')

    # fit warp with constrained optimizer (stretch in [0.9,1.1], offset bounds +-5s)
    try:
        res = fit_time_warp(t, exp_vid, exp_bhv, bounds=(0.9, 1.1), offset_bounds=(-100.0, 100.0))
        bi = res.x
        if res.success:
            if None is not None:
                pass
        if True:
            print("Warp fit success (initial):", res.success, "bi:", bi)
    except Exception as e:
        warnings.warn(f"Warp fit failed: {e}")
        bi = np.array([1.0, 0.0])

    exp_bhv_warped = tempwarp(bi, t, exp_bhv)

    # reverse warping to get estimated video times for events
    trial_init_times_fix = np.array(trial_rwd_times)  # placeholder
    trial_rwd_times_fix = np.array(trial_rwd_times)
    trial_reach_times_fix = np.array(trial_rwd_times)  # placeholder
    # matches MATLAB: trial_rwd_times_hat = trial_rwd_times_fix / bi(1) - bi(2)
    trial_init_times_hat = trial_rwd_times / bi[0] - bi[1]
    trial_rwd_times_hat = trial_rwd_times / bi[0] - bi[1]
    trial_reach_times_hat = trial_rwd_times / bi[0] - bi[1]

    # infer reach times from the pc1 trace (fallback method)
    trial_reach_times_inferred = infer_reach_times_from_trace(time, pc1_norm, trial_rwd_times_hat)

    # subtracttime to align inferred hits back to behavior log space if needed (reverse of time_fix)
    trial_reach_times_inferred_hat, _ = subtracttime(time, trial_reach_times_inferred, -0.0)
    # warp back into behavior clock
    trial_reach_times_inferred_behavior = trial_reach_times_inferred_hat * bi[0] + bi[1]

    # final drift correction using robust regression if enough data
    trial_count = min(len(trial_rwd_times), len(trial_reach_times_inferred_behavior))
    driftmdl = None
    if trial_count > 2:
        reach_delays = np.array([])  # placeholder (not used here)
        reach_delays_inferred = np.array([])  # placeholder
        valid_flags = np.array([])
        if np.sum(valid_flags) > 2:
            huber = HuberRegressor().fit(reach_delays_inferred[valid_flags].reshape(-1,1),
                                         reach_delays[valid_flags])
            driftmdl = {'coef': float(huber.coef_[0]), 'intercept': float(huber.intercept_)}
            trial_reach_times_inferred_approxlog = huber.predict(reach_delays_inferred.reshape(-1,1)) + trial_rwd_times[:trial_count]
        else:
            trial_reach_times_inferred_approxlog = trial_reach_times_inferred_behavior[:trial_count].copy()
    else:
        trial_reach_times_inferred_approxlog = trial_reach_times_inferred_behavior[:trial_count].copy()

    # Print summary head
    summary_df = pd.DataFrame({
        'inferred_behavior': trial_reach_times_inferred_behavior[:trial_count],
        'inferred_approxlog': trial_reach_times_inferred_approxlog,
        'logged': np.array(trial_rwd_times)[:trial_count]
    })
    if True:
        print("Summary (first rows):")
        print(summary_df.head(10))

    # ------------- TRAIN/TEST EVALUATION & optional sync writer ---------------
    try:
        # detect per-trial combined events + candidates (uses pc1_z & roi_scores)
        events, candidates = detect_droplet_events(time, pc1_z, roi_scores, roi_coords,
                                                   trial_rwd_times, search_window=(0.0, 2.0),
                                                   weights=None, min_confidence=min_conf)

        # Pair behavior times <-> video inferred times (video-inferred in behavior clock)
        bhv_times = np.array(trial_rwd_times, dtype=float)          # behavior times (logged)
        vid_times = np.array(trial_reach_times_inferred_behavior, dtype=float) # video-inferred times warped into behavior clock
        valid_mask = np.isfinite(bhv_times) & np.isfinite(vid_times)

        if np.sum(valid_mask) >= 3:
            idxs = np.where(valid_mask)[0]
            # deterministic split (seeded) to keep reproducibility
            rng = np.random.RandomState(0)
            rng.shuffle(idxs)
            n_train = max(1, int(len(idxs) * float(train_frac)))   # train_frac can be passed in
            train_idx = idxs[:n_train]
            test_idx = idxs[n_train:]

            # fit warp on training subset (behavior->video mapping in behavior clock space)
            warp = fit_sync_warp(bhv_times[train_idx], vid_times[train_idx], bounds=(0.95, 1.05), offset_bounds=(-100.0, 100.0))

            # predict on test subset and evaluate (predicted video time in behavior clock)
            pred_vid = warp['stretch'] * bhv_times[test_idx] + warp['offset']
            true_vid = vid_times[test_idx]
            metrics = evaluate_against_ground_truth(pred_vid, true_vid, tolerance_ms=50)

            print("TRAIN/TEST eval results:", metrics)
            print("warp:", warp.get('stretch'), warp.get('offset'), "rmse:", warp.get('rmse'))

            # optionally write canonical sync file when requested
            if make_sync:
                events_map = []
                for ti in range(len(bhv_times)):
                    bt = float(bhv_times[ti]) if np.isfinite(bhv_times[ti]) else None
                    vt = float(warp['stretch'] * bt + warp['offset']) if bt is not None else None
                    conf = events[ti]['score'] if (events and events[ti] is not None and 'score' in events[ti]) else None
                    events_map.append({'trial_idx': int(ti), 'behavior_time': bt, 'video_time': vt, 'confidence': conf})
                metadata = {'video_file': str(video_file), 'bhv_file': str(bhv_file), 'fps': float(fps), 'frame_count': int(frame_count)}
                sync_out = Path(out_folder) / f"sync_{video_file.stem}.json"
                write_sync_file(str(sync_out), warp, events_map, metadata=metadata)
                print("WROTE SYNC:", sync_out)
        else:
            print("Not enough matched trials (>=3 required) to run train/test sync evaluation.")
    except Exception as e:
        warnings.warn(f"Train/test sync evaluation failed: {e}")
    # ------------- end train/test block -------------------------------------

    # save diagnostics
    name_prefix = video_file.stem
    save_diagnostics(out_folder, name_prefix,
                     time=time, pc1_z=pc1_z, roi_scores=roi_scores, roi_coords=roi_coords,
                     trial_rwd_times=trial_rwd_times, trial_reach_times=np.array(trial_rwd_times),
                     trial_reach_times_inferred=trial_reach_times_inferred_behavior,
                     warp_bi=bi, explained_variance=explained)

    # save CSV of reconstructed camera toggle like original
    try:
        save_syncfix = out_folder / f"syncfix_{name_prefix}.mat"
        savemat(str(save_syncfix), {
            'sync_state_hat': np.zeros(frame_count, dtype=bool),
            'trial_reach_times_inferred': trial_reach_times_inferred_behavior,
            'trial_reach_times_inferred_approxlog': trial_reach_times_inferred_approxlog,
            'trial_rwd_times': trial_rwd_times
        })
        print("Wrote syncfix MAT:", save_syncfix)
    except Exception as e:
        warnings.warn(f"Failed saving syncfix MAT: {e}")

    return {
        'time': time,
        'pc1_z': pc1_z,
        'roi_scores': roi_scores,
        'roi_coords': roi_coords,
        'trial_rwd_times': trial_rwd_times,
        'trial_reach_times_inferred_behavior': trial_reach_times_inferred_behavior,
        'warp_bi': bi
    }

# ----------------------- folder-level helpers (unchanged) -----------------------
def scan_and_process_folder(root_folder, group, setup, out_folder=None, expected_fps=None, interactive_roi=False):
    root = Path(root_folder)
    for session_dir in root.glob(f"{group}/{setup}/*"):
        try:
            bhv_files = list(session_dir.glob("GlobalLog*.csv"))
            video_files = list(session_dir.glob("Camera*.avi"))
            sync_files = list(session_dir.glob("Camera*.csv"))
            if not bhv_files or not video_files or not sync_files:
                continue
            out = out_folder if out_folder is not None else session_dir
            process_one_session(bhv_files[0], video_files[0], sync_files[0],
                                out_folder=out, expected_fps=expected_fps, interactive_roi=interactive_roi)
        except Exception as e:
            warnings.warn(f"Failed processing {session_dir}: {e}")

# ----------------------- entrypoint -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-step', type=int, default=10, help='Process 1 every N frames (speedup).')
    parser.add_argument('--search-margin', type=int, default=50, help='Search window margin (px) around ROI for template matching.')
    parser.add_argument('--resize', type=float, default=1.0, help='Optional frame/template resize factor (<1 speeds up).')
    parser.add_argument('--bhv', help='Behavior CSV file (GlobalLog*.csv)', default=None)
    parser.add_argument('--video', help='Video file (Camera*.avi)', default=None)
    parser.add_argument('--sync', help='Sync CSV file (Camera*.csv)', default=None)
    parser.add_argument('--scan', help='Root folder to scan for group/setup/animals', default=None)
    parser.add_argument('--group', help='Group folder name (used with --scan)', default=None)
    parser.add_argument('--setup', help='Setup folder name (used with --scan)', default=None)
    parser.add_argument('--out', help='Output folder', default=None)
    parser.add_argument('--fps', help='Expected FPS for warning', type=float, default=None)
    parser.add_argument('--interactive_roi', action='store_true', help='Use cv2.selectROI to draw ROIs')
    parser.add_argument('--make-sync', action='store_true', help='Write a sync JSON file using train subset')
    parser.add_argument('--train-frac', type=float, default=0.8, help='Fraction of matched trials used for training (0-1)')
    parser.add_argument('--min-conf', type=float, default=0.35, help='Min detection confidence used by detector')
    parser.add_argument('--sweep', action='store_true', help='Run a simple hyperparameter sweep (min_conf x pc1_weight)')
    parser.add_argument('--roi', type=str, default=None, help='ROI box as x0,y0,x1,y1 e.g. 130,265,180,315')
    parser.add_argument('--minconf-grid', type=str, default=None, help='Comma separated min_conf grid, e.g. 0.2,0.3,0.4')
    parser.add_argument('--pc1w-grid', type=str, default=None, help='Comma separated pc1 weight grid, e.g. 0.3,0.45,0.6')
    args = parser.parse_args()

    if args.scan:
        if args.group is None or args.setup is None:
            parser.error("--scan requires --group and --setup")
        scan_and_process_folder(args.scan, args.group, args.setup,
                                out_folder=args.out, expected_fps=args.fps, interactive_roi=args.interactive_roi)
    else:
        if args.bhv is None or args.video is None or args.sync is None:
            parser.error("Provide --bhv, --video, and --sync or use --scan")
        # first run the session processing to compute traces and preliminary warp
        roi_box = [int(v) for v in args.roi.split(',')] if args.roi else None
        session_res = process_one_session(args.bhv, args.video, args.sync, out_folder=args.out,
                        expected_fps=args.fps, interactive_roi=args.interactive_roi,
                        make_sync=args.make_sync, train_frac=args.train_frac, min_conf=args.min_conf,
                        frame_step=args.frame_step, search_margin=args.search_margin, resize=args.resize,
                        roi=roi_box)
        # if sweep requested, run it now using the returned session data
        if args.sweep:
            # parse grids if provided
            if args.minconf_grid:
                try:
                    minconf_list = [float(x.strip()) for x in args.minconf_grid.split(',') if x.strip()!='']
                except Exception:
                    minconf_list = None
            else:
                minconf_list = None

            if args.pc1w_grid:
                try:
                    pc1w_list = [float(x.strip()) for x in args.pc1w_grid.split(',') if x.strip()!='']
                except Exception:
                    pc1w_list = None
            else:
                pc1w_list = None

            out_folder = args.out if args.out is not None else Path(args.bhv).parent
            csv_path = hyperparameter_sweep(session_res, out_folder, Path(args.video).stem, args.train_frac, minconf_list, pc1w_list, seed=0, tolerance_ms=50)
            print("Hyperparameter sweep results saved to:", csv_path)