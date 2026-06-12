#!/usr/bin/env python3
"""
coincidence_audit.py
====================
Multi-feature peak intersection auditor.
Finds combinations of features whose peaks co-occur within a tight time
window more often than chance, isolating true event-locked signals from noise.
"""
import sys, argparse, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from itertools import combinations

warnings.filterwarnings('ignore')

# ── Cache ─────────────────────────────────────────────────────────────────────
def load_cache(cache_path):
    d = np.load(str(cache_path), allow_pickle=False)
    traces, times = {}, {}
    for k in d.files:
        if k.startswith('trace_'):   traces[k[6:]] = d[k]
        elif k.startswith('time_'):  times[k[5:]]  = d[k]
    reserved = {'sync_edge_times', 'sr_ap', 'sr_lf'}
    for k in d.files:
        if k not in reserved and not k.startswith(('trace_', 'time_')) \
                and k not in traces:
            traces[k] = d[k]
    return traces, times

def get_time_array(fname, traces, times, session_dur):
    if fname in times and len(times[fname]) > 1:
        return times[fname].astype(np.float64)
    n  = len(traces[fname])
    sr = (n / session_dur) if session_dur > 0 else 1000.0
    return np.arange(n, dtype=np.float64) / sr

def infer_sr(t_arr):
    if t_arr is None or len(t_arr) < 2:
        return 1000.0
    return float(1.0 / np.median(np.diff(t_arr[:min(10000, len(t_arr))])))

# ── Behaviour ─────────────────────────────────────────────────────────────────
def load_behavior(bhv_path, target_code):
    df = pd.read_csv(str(bhv_path), header=None, names=['ts', 'code'])
    df['ts']   = pd.to_numeric(df['ts'],   errors='coerce')
    df['code'] = pd.to_numeric(df['code'], errors='coerce')
    df = df.dropna()
    df['ts'] -= df['ts'].iloc[0]          # zero-anchor to first event
    return df.loc[df['code'] == target_code, 'ts'].values.astype(np.float64)

# ── Peak detection ────────────────────────────────────────────────────────────
def get_peak_times(trace, t_arr, prominence, min_distance_s=0.050):
    sr = infer_sr(t_arr)
    dist = max(1, int(min_distance_s * sr))
    try:
        idx, _ = find_peaks(trace.astype(float), 
                            prominence=prominence, distance=dist)
        return t_arr[idx].astype(np.float64)
    except Exception:
        return np.array([], dtype=np.float64)

# ── Sparse binary representation (10 ms bins, ±window smear) ─────────────────
BIN_S = 0.010   # 10 ms bins
def peaks_to_bins(peak_times, n_bins, window_s):
    """Expand peak times into a sorted array of occupied bin indices."""
    if len(peak_times) == 0:
        return np.array([], dtype=np.int32)
    w    = max(1, int(window_s / BIN_S))
    base = np.clip((peak_times / BIN_S).astype(np.int32), 0, n_bins - 1)
    off  = np.arange(-w, w + 1, dtype=np.int32)
    idx  = (base[:, None] + off[None, :]).ravel()
    return np.unique(np.clip(idx, 0, n_bins - 1)).astype(np.int32)

def bins_to_events(bins, min_gap_bins=5):
    """Cluster adjacent bins into single event times (seconds)."""
    if len(bins) == 0:
        return np.array([], dtype=np.float64)
    gaps    = np.diff(bins) > min_gap_bins
    ends    = np.concatenate([np.where(gaps)[0] + 1, [len(bins)]])
    starts  = np.concatenate([[0], ends[:-1]])
    centers = np.array([bins[s:e].mean() for s, e in zip(starts, ends)],
                       dtype=np.float64)
    return centers * BIN_S

def intersect_bins(bin_list):
    """Intersection of N sorted bin-index arrays (all must co-occur)."""
    result = bin_list[0]
    for b in bin_list[1:]:
        result = np.intersect1d(result, b, assume_unique=True)
        if len(result) == 0:
            break
    return result

# ── Scoring (fully vectorised) ────────────────────────────────────────────────
def score(event_times, behavior_times, hit_window_s):
    """
    Returns (hit_rate, false_alarm_rate, yield_index, n_peaks).
    Uses sorted-array binary search — O(n log n) total.
    hit_window_s is generous (default 0.5s) to cover the ~200ms physics offset.
    """
    n_rwd = len(behavior_times)
    n_evt = len(event_times)
    if n_evt == 0:
        return 0.0, 1.0, 0.0, 0
    et = np.sort(event_times)
    bt = np.sort(behavior_times)

    # ── Hit rate: fraction of rewards with a nearby detected peak ──────────
    idx  = np.searchsorted(et, bt).clip(0, n_evt - 1)
    idxm = (idx - 1).clip(0, n_evt - 1)
    min_d_bt = np.minimum(np.abs(et[idx] - bt), np.abs(et[idxm] - bt))
    hit_rate = np.mean(min_d_bt < hit_window_s)

    # ── False alarm rate: fraction of peaks not near any reward ──────────
    idx2  = np.searchsorted(bt, et).clip(0, n_rwd - 1)
    idx2m = (idx2 - 1).clip(0, n_rwd - 1)
    min_d_et = np.minimum(np.abs(bt[idx2] - et), np.abs(bt[idx2m] - et))
    far = np.mean(min_d_et >= hit_window_s)
    
    yi = hit_rate * (1.0 - far)
    return float(hit_rate), float(far), float(yi), n_evt

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Multi-feature coincidence auditor.')
    ap.add_argument('--cache',          required=True)
    ap.add_argument('--bhv',            required=True)
    ap.add_argument('--target-code',    type=int,   default=2)
    ap.add_argument('--window-ms',      type=float, default=25.0,
                    help='Coincidence window ±ms (default 25)')
    ap.add_argument('--hit-window-ms',  type=float, default=500.0,
                    help='Scoring tolerance ±ms (default 500, covers 200ms physics offset)')
    ap.add_argument('--prominence',     type=float, default=2.0)
    ap.add_argument('--top-n',          type=int,   default=40,
                    help='Pre-filter to top-N features before combinatorics')
    ap.add_argument('--max-combo',      type=int,   default=3,
                    help='Max combination size: 2=pairs only, 3=pairs+triplets')
    ap.add_argument('--features',       type=str,   default=None,
                    help='Comma-separated feature names, or "better" for '
                         'car_/sdf_/rmi_/pac_ families')
    args = ap.parse_args()

    window_s    = args.window_ms    / 1000.0
    hit_win_s   = args.hit_window_ms / 1000.0

    print(f'\n{"="*65}')
    print(f' coincidence_audit.py  |  code={args.target_code}  '
          f'window={args.window_ms}ms  prom={args.prominence}')
    print(f'{"="*65}\n')

    # ── Load ────────────────────────────────────────────────────────────────
    print('Loading cache...')
    traces, times = load_cache(args.cache)
    
    print(f'Loading behavior...')
    bhv_times = load_behavior(args.bhv, args.target_code)
    n_rwd = len(bhv_times)
    if n_rwd == 0:
        print(f'[ERROR] No events with code={args.target_code} found.'); sys.exit(1)
    
    print(f'  {n_rwd} target events  '
          f'(t={bhv_times.min():.0f}–{bhv_times.max():.0f}s)\n')
          
    session_dur = float(bhv_times.max()) + 60.0
    n_bins      = int(session_dur / BIN_S) + 1

    # ── Feature selection ───────────────────────────────────────────────────
    if args.features and args.features not in ('better',):
        feat_names = [f.strip() for f in args.features.split(',')
                      if f.strip() in traces]
        print(f'Using {len(feat_names)} user-specified features.\n')
    else:
        feat_names = sorted(k for k in traces
                            if k.startswith(('car_', 'sdf_', 'rmi_', 'pac_'))
                            and 'sol_depth' not in k)
        print(f'Found {len(feat_names)} better-family features.\n')
        
    if not feat_names:
        print('[ERROR] No matching features in cache.'); sys.exit(1)

    # ── Extract peaks + individual score ────────────────────────────────────
    print(f'Extracting peaks (prominence={args.prominence})...')
    feat_bins   = {}
    feat_yields = {}
    
    for i, fn in enumerate(feat_names, 1):
        if i % 100 == 0:
            print(f'  {i}/{len(feat_names)}...', flush=True)
        t_arr      = get_time_array(fn, traces, times, session_dur)
        peak_t     = get_peak_times(traces[fn], t_arr, args.prominence)
        bins       = peaks_to_bins(peak_t, n_bins, window_s)
        feat_bins[fn] = bins
        
        evt_t      = bins_to_events(bins)
        _, _, yi, _ = score(evt_t, bhv_times, hit_win_s)
        feat_yields[fn] = yi
        
    print('Done.\n')

    # ── Pre-filter to top-N unless user gave explicit list ──────────────────
    if args.features and args.features not in ('better',):
        selected = feat_names
    else:
        ranked   = sorted(feat_yields.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in ranked[:args.top_n]]
        
        print(f'Top {len(selected)} features by individual yield:')
        hdr = f"  {'Feature':<45} {'Peaks':>6} {'Yield':>6}"
        print(hdr); print('  ' + '-'*(len(hdr)-2))
        for fn, yi in ranked[:15]:
            pt = bins_to_events(feat_bins[fn])
            print(f'  {fn:<45} {len(pt):>6} {yi:>6.3f}')
        if len(ranked) > 15:
            print(f'  ... ({len(selected)-15} more)')
        print()

    # ── Combinatorial search ─────────────────────────────────────────────────
    results = []
    max_c   = min(args.max_combo, len(selected))
    
    for csz in range(2, max_c + 1):
        combos = list(combinations(selected, csz))
        print(f'Testing {len(combos):,} size-{csz} combinations...')
        
        for combo in combos:
            cb   = intersect_bins([feat_bins[f] for f in combo])
            evt  = bins_to_events(cb)
            hr, far, yi, np_ = score(evt, bhv_times, hit_win_s)
            results.append(dict(combo=combo, size=csz,
                                n_peaks=np_, hit_rate=hr, far=far, yld=yi))
        print(f'  Done.\n')

    # ── Leaderboard ──────────────────────────────────────────────────────────
    results.sort(key=lambda x: x['yld'], reverse=True)
    print('='*100)
    print(f' LEADERBOARD — top 25 by Yield Index  '
          f'(hit_win=±{args.hit_window_ms:.0f}ms, coinc_win=±{args.window_ms:.0f}ms)')
    print('='*100)
    print(f"{'Rk':<4} {'Combination':<70} {'Pk':>5} {'Hit%':>6} {'FAR%':>6} {'Yield':>6}")
    print('-'*100)
    
    for i, r in enumerate(results[:25], 1):
        combo_str = ' + '.join(r['combo'])
        if len(combo_str) > 68: combo_str = combo_str[:65] + '...'
        print(f"{i:<4} {combo_str:<70} {r['n_peaks']:>5} "
              f"{r['hit_rate']*100:>5.1f}% {r['far']*100:>5.1f}%  {r['yld']:>5.3f}")
    print('='*100)
    
    print(f'\n  Yield = HitRate × (1 − FAR) — higher = more reward-specific peaks')
    print(f'  HitRate = % of {n_rwd} rewards with a coincidence peak nearby')
    print(f'  FAR     = % of coincidence peaks NOT near any reward\n')

    # Best single-feature baseline for comparison
    best_single = max(feat_yields.items(), key=lambda x: x[1])
    print(f'  Best single-feature baseline: {best_single[0]}  yield={best_single[1]:.3f}')
    print(f'  Any combo above this is a genuine improvement.\n')

if __name__ == '__main__':
    main()
