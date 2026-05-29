#!/usr/bin/env python3
"""
extract_extra_features.py
==========================
Reads SpikeGLX bin files and appends additional feature candidates to an
existing features cache (.npz). Designed to run AFTER ephys_alignment_fusion.py
has already built the base cache.

NEW FEATURES ADDED
------------------
LF file (broadband already in cache as lfp_block_N):
  lfp_block_<N>_<band>         PC1 of band-filtered block N
  lfp_block_<N>_<band>_pc2     PC2
  lfp_block_<N>_<band>_pc3     PC3
  Bands: delta(1-4Hz), theta(4-8Hz), beta(8-30Hz), gamma(30-80Hz)

AP file:
  ap_block_<N>_mua             PC1 of 300-3000Hz RMS envelope, block N
  ap_block_<N>_mua_pc2/pc3     PC2/PC3
  ap_block_<N>_hfo             PC1 of 80-200Hz RMS envelope, block N
  ap_block_<N>_hfo_pc2/pc3     PC2/PC3
  ap_block_<N>_gamma           PC1 of 30-80Hz RMS envelope, block N
  ap_block_<N>_gamma_pc2/pc3   PC2/PC3
  sol_depth_<N>                abs(derivative) of AP median per 20-ch block

Template matching (uses existing cache + behaviour, no bin needed):
  tmpl_sol                     NCC of reward-triggered solenoid template
  tmpl_lfp_<N>                 NCC of reward-triggered LFP template, best block
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, fftconvolve
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import IncrementalPCA

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirrors of ephys_alignment_fusion.py)
# ──────────────────────────────────────────────────────────────────────────────
def odd_int(x):
    x = max(1, int(x))
    return x if x % 2 == 1 else x + 1

def load_spikeglx_meta(bin_path):
    meta_path = Path(bin_path).with_suffix('.meta')
    if not meta_path.exists():
        return {}
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
    return meta

def get_bin_params(bin_path):
    """Returns (n_channels, sample_rate, n_samples, uv_per_bit)."""
    meta = load_spikeglx_meta(bin_path)
    file_size = Path(bin_path).stat().st_size
    # Try meta first
    n_ch = int(meta.get('nSavedChans', 385))
    sr   = float(meta.get('imSampRate', 30000.0))
    if 'imSampRate' not in meta:
        # LF file
        sr = float(meta.get('imSampRate', 2500.0))
        if '2500' in str(bin_path) or 'lf' in str(bin_path).lower():
            sr = 2500.0
    # Infer from filename if needed
    if 'lf.bin' in str(bin_path).lower():
        sr = 2500.0
    n_samp = file_size // (n_ch * 2)  # int16 = 2 bytes
    uv_per_bit = float(meta.get('imAiRangeMax', 0.6)) / 512.0 if meta else 1.0
    return n_ch, sr, n_samp, uv_per_bit

def read_chunk(bin_path, n_ch, start_samp, n_samp):
    """Read a chunk of data from a SpikeGLX bin file. Returns (n_ch, n_samp) int16."""
    fp = np.memmap(bin_path, dtype='int16', mode='r', shape=(None,), offset=0)
    total_samp = len(fp) // n_ch
    end_samp = min(start_samp + n_samp, total_samp)
    actual   = end_samp - start_samp
    flat     = fp[start_samp * n_ch : end_samp * n_ch]
    data     = flat.reshape(actual, n_ch).T.copy().astype(np.float32)
    del fp
    return data

def load_feature_cache(cache_path):
    d = np.load(str(cache_path), allow_pickle=False)
    traces, times = {}, {}
    for key in d.files:
        if key.startswith('trace_'):
            traces[key[6:]] = d[key]
        elif key.startswith('time_'):
            times[key[5:]] = d[key]
    reserved = {'sync_edge_times', 'sr_ap', 'sr_lf'}
    for key in d.files:
        if key not in reserved and not key.startswith('trace_') \
                and not key.startswith('time_') and key not in traces:
            traces[key] = d[key]
    sr_ap = float(d['sr_ap'][0]) if 'sr_ap' in d.files else None
    sr_lf = float(d['sr_lf'][0]) if 'sr_lf' in d.files else None
    sync  = d['sync_edge_times'] if 'sync_edge_times' in d.files else None
    return traces, times, sync, sr_ap, sr_lf

def save_feature_cache(cache_path, traces, times, sync, sr_ap, sr_lf):
    payload = {}
    for k, v in traces.items():
        payload[f'trace_{k}'] = v.astype(np.float32)
    for k, v in times.items():
        payload[f'time_{k}'] = v.astype(np.float32)
    if sync is not None:
        payload['sync_edge_times'] = sync
    if sr_ap is not None:
        payload['sr_ap'] = np.array([sr_ap])
    if sr_lf is not None:
        payload['sr_lf'] = np.array([sr_lf])
    np.savez_compressed(str(cache_path), **payload)
    print(f'  [CACHE] Saved {len(traces)} traces → {Path(cache_path).name}')

def update_cache(cache_path, new_traces, new_times):
    """Merge new features into existing cache."""
    existing_traces, existing_times, sync, sr_ap, sr_lf = load_feature_cache(cache_path)
    existing_traces.update(new_traces)
    existing_times.update(new_times)
    save_feature_cache(cache_path, existing_traces, existing_times, sync, sr_ap, sr_lf)

def z_score(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s if s > 0 else 1.0)

def make_bandpass(lo, hi, sr):
    nyq = sr / 2.0
    lo_n = lo / nyq
    hi_n = hi / nyq
    lo_n = max(1e-4, min(lo_n, 0.9999))
    hi_n = max(1e-4, min(hi_n, 0.9999))
    if lo_n >= hi_n:
        raise ValueError(f'Bad band: {lo}-{hi} Hz at sr={sr}')
    return butter(4, [lo_n, hi_n], btype='band', output='sos')

# ──────────────────────────────────────────────────────────────────────────────
# Band definitions
# ──────────────────────────────────────────────────────────────────────────────
LFP_BANDS = {
    'delta': (1.0,   4.0),
    'theta': (4.0,   8.0),
    'beta':  (8.0,  30.0),
    'gamma': (30.0, 80.0),
}

AP_BANDS = {
    'mua':   (300.0, 3000.0),
    'hfo':   (80.0,  200.0),
    'gamma': (30.0,   80.0),
}

# ──────────────────────────────────────────────────────────────────────────────
# LFP bands + PC1/PC2/PC3
# ──────────────────────────────────────────────────────────────────────────────
def extract_lfp_bands_pcs(lf_path, sr_lf, n_ch_lf, n_pcs, band_names, block_size, chunk_sec, out_sr=1000.0):
    n_ch_data = n_ch_lf - 1 
    n_blocks  = max(1, (n_ch_data) // block_size)
    chunk_samp = int(chunk_sec * sr_lf)
    ds_factor  = max(1, int(sr_lf / out_sr))
    total_samp = int(Path(lf_path).stat().st_size // (n_ch_lf * 2))
    
    out_traces = {}
    out_times  = {}
    
    for band_name in band_names:
        if band_name not in LFP_BANDS:
            print(f'  [LFP BAND] Unknown band "{band_name}", skipping.')
            continue
        lo, hi = LFP_BANDS[band_name]
        print(f'\n  [LFP BAND] {band_name} ({lo}-{hi} Hz), blocks 0-{n_blocks-1}, {n_pcs} PCs')
        try:
            sos = make_bandpass(lo, hi, sr_lf)
        except ValueError as e:
            print(f'    [SKIP] {e}')
            continue
            
        ipcas = [IncrementalPCA(n_components=min(n_pcs, block_size)) for _ in range(n_blocks)]
        
        # ── Pass 1: fit ──
        start = 0
        chunk_idx = 0
        n_chunks = int(np.ceil(total_samp / chunk_samp))
        while start < total_samp:
            data = read_chunk(lf_path, n_ch_lf, start, chunk_samp)
            chunk_idx += 1
            print(f'    fit  chunk {chunk_idx}/{n_chunks}', end='\r')
            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = data[ch0:ch1, :]
                if blk.shape[0] < 2: continue
                filt = np.zeros_like(blk)
                for ci in range(blk.shape[0]):
                    filt[ci] = sosfiltfilt(sos, blk[ci].astype(float))
                ds = filt[:, ::ds_factor].T
                ipcas[bi].partial_fit(ds)
            start += chunk_samp
        print()
        
        # ── Pass 2: transform ──
        block_pcs = [[] for _ in range(n_blocks)]
        start = 0
        chunk_idx = 0
        while start < total_samp:
            data = read_chunk(lf_path, n_ch_lf, start, chunk_samp)
            chunk_idx += 1
            print(f'    xfm  chunk {chunk_idx}/{n_chunks}', end='\r')
            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = data[ch0:ch1, :]
                if blk.shape[0] < 2: continue
                filt = np.zeros_like(blk)
                for ci in range(blk.shape[0]):
                    filt[ci] = sosfiltfilt(sos, blk[ci].astype(float))
                ds = filt[:, ::ds_factor].T
                projected = ipcas[bi].transform(ds)
                block_pcs[bi].append(projected)
            start += chunk_samp
        print()
        
        # ── Assemble + store ──
        actual_sr = sr_lf / ds_factor
        for bi in range(n_blocks):
            if not block_pcs[bi]: continue
            assembled = np.vstack(block_pcs[bi])
            t_arr = np.arange(assembled.shape[0]) / actual_sr
            for pc_i in range(assembled.shape[1]):
                pc_trace = assembled[:, pc_i].astype(np.float32)
                pc_z = z_score(pc_trace)
                if pc_i == 0:
                    fname = f'lfp_block_{bi}_{band_name}'
                else:
                    fname = f'lfp_block_{bi}_{band_name}_pc{pc_i+1}'
                out_traces[fname] = pc_z
                out_times[fname]  = t_arr
                print(f'    → {fname}  len={len(pc_z)}')
                
    return out_traces, out_times

# ──────────────────────────────────────────────────────────────────────────────
# AP block bands (MUA / HFO / gamma) + PC1/PC2/PC3
# ──────────────────────────────────────────────────────────────────────────────
def extract_ap_block_bands(ap_path, sr_ap, n_ch_ap, n_pcs, band_names, block_size, chunk_sec, out_sr=1000.0):
    n_ch_data = n_ch_ap - 1
    n_blocks  = max(1, n_ch_data // block_size)
    chunk_samp = int(chunk_sec * sr_ap)
    ds_factor  = max(1, int(sr_ap / out_sr))
    total_samp = int(Path(ap_path).stat().st_size // (n_ch_ap * 2))
    smooth_samp = max(1, int(0.025 * sr_ap)) 
    
    out_traces = {}
    out_times  = {}
    
    for band_name in band_names:
        if band_name not in AP_BANDS:
            print(f'  [AP BAND] Unknown band "{band_name}", skipping.')
            continue
        lo, hi = AP_BANDS[band_name]
        print(f'\n  [AP BAND] {band_name} ({lo}-{hi} Hz), {n_blocks} blocks, {n_pcs} PCs')
        try:
            sos = make_bandpass(lo, hi, sr_ap)
        except ValueError as e:
            print(f'    [SKIP] {e}')
            continue
            
        ipcas = [IncrementalPCA(n_components=min(n_pcs, block_size)) for _ in range(n_blocks)]
        
        # ── Pass 1: fit ──
        start = 0
        n_chunks = int(np.ceil(total_samp / chunk_samp))
        chunk_idx = 0
        while start < total_samp:
            data = read_chunk(ap_path, n_ch_ap, start, chunk_samp)
            chunk_idx += 1
            print(f'    fit  chunk {chunk_idx}/{n_chunks}', end='\r')
            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = data[ch0:ch1, :]
                if blk.shape[0] < 2: continue
                envs = np.zeros((blk.shape[0], blk.shape[1]))
                for ci in range(blk.shape[0]):
                    filt = sosfiltfilt(sos, blk[ci].astype(float))
                    sq   = filt ** 2
                    envs[ci] = np.sqrt(uniform_filter1d(sq, size=smooth_samp))
                ds = envs[:, ::ds_factor].T
                ipcas[bi].partial_fit(ds)
            del data
            start += chunk_samp
        print()
        
        # ── Pass 2: transform ──
        block_pcs = [[] for _ in range(n_blocks)]
        start = 0
        chunk_idx = 0
        while start < total_samp:
            data = read_chunk(ap_path, n_ch_ap, start, chunk_samp)
            chunk_idx += 1
            print(f'    xfm  chunk {chunk_idx}/{n_chunks}', end='\r')
            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = data[ch0:ch1, :]
                if blk.shape[0] < 2: continue
                envs = np.zeros((blk.shape[0], blk.shape[1]))
                for ci in range(blk.shape[0]):
                    filt = sosfiltfilt(sos, blk[ci].astype(float))
                    sq   = filt ** 2
                    envs[ci] = np.sqrt(uniform_filter1d(sq, size=smooth_samp))
                ds = envs[:, ::ds_factor].T
                block_pcs[bi].append(ipcas[bi].transform(ds))
            del data
            start += chunk_samp
        print()
        
        # ── Assemble + store ──
        actual_sr = sr_ap / ds_factor
        for bi in range(n_blocks):
            if not block_pcs[bi]: continue
            assembled = np.vstack(block_pcs[bi])
            t_arr = np.arange(assembled.shape[0]) / actual_sr
            for pc_i in range(assembled.shape[1]):
                pc_z = z_score(assembled[:, pc_i].astype(np.float32))
                if pc_i == 0:
                    fname = f'ap_block_{bi}_{band_name}'
                else:
                    fname = f'ap_block_{bi}_{band_name}_pc{pc_i+1}'
                out_traces[fname] = pc_z
                out_times[fname]  = t_arr
                print(f'    → {fname}  len={len(pc_z)}')
                
    return out_traces, out_times

# ──────────────────────────────────────────────────────────────────────────────
# Solenoid by depth blocks
# ──────────────────────────────────────────────────────────────────────────────
def extract_solenoid_depth(ap_path, sr_ap, n_ch_ap, block_size, chunk_sec):
    n_ch_data = n_ch_ap - 1
    n_blocks  = max(1, n_ch_data // block_size)
    chunk_samp = int(chunk_sec * sr_ap)
    total_samp = int(Path(ap_path).stat().st_size // (n_ch_ap * 2))
    
    print(f'\n  [SOL DEPTH] {n_blocks} depth blocks')
    block_traces = [[] for _ in range(n_blocks)]
    start = 0
    n_chunks = int(np.ceil(total_samp / chunk_samp))
    chunk_idx = 0
    while start < total_samp:
        data = read_chunk(ap_path, n_ch_ap, start, chunk_samp)
        chunk_idx += 1
        print(f'    chunk {chunk_idx}/{n_chunks}', end='\r')
        for bi in range(n_blocks):
            ch0 = bi * block_size
            ch1 = min(ch0 + block_size, n_ch_data)
            blk = data[ch0:ch1, :]
            if blk.shape[0] < 2: continue
            med = np.median(blk, axis=0).astype(float)
            deriv = np.abs(np.diff(med, prepend=med[0]))
            block_traces[bi].append(deriv.astype(np.float32))
        del data
        start += chunk_samp
    print()
    
    out_traces = {}
    out_times  = {}
    t_arr = np.arange(total_samp) / sr_ap
    for bi in range(n_blocks):
        if not block_traces[bi]: continue
        full = np.concatenate(block_traces[bi])
        full_z = z_score(full)
        fname = f'sol_depth_{bi}'
        out_traces[fname] = full_z
        out_times[fname]  = t_arr[:len(full_z)]
        print(f'    → {fname}  len={len(full_z)}')
        
    return out_traces, out_times

# ──────────────────────────────────────────────────────────────────────────────
# Template matching
# ──────────────────────────────────────────────────────────────────────────────
def fast_ncc(signal, template):
    T  = len(template)
    N  = len(signal)
    tmpl = template - template.mean()
    tmpl_e = np.sum(tmpl ** 2)
    if tmpl_e < 1e-12: return np.zeros(N)
    
    win_sum  = uniform_filter1d(signal, size=T, mode='constant') * T
    win_sum2 = uniform_filter1d(signal**2, size=T, mode='constant') * T
    win_mean = win_sum / T
    win_e    = win_sum2 - win_sum**2 / T
    win_e    = np.maximum(win_e, 1e-12)
    
    xcorr_full = fftconvolve(signal - signal.mean(), tmpl[::-1], mode='full')
    half_T = T // 2
    xcorr  = xcorr_full[T-1 - half_T : T-1 - half_T + N]
    
    if len(xcorr) < N:
        xcorr = np.pad(xcorr, (0, N - len(xcorr)))
        
    ncc_trace = xcorr / np.sqrt(win_e * tmpl_e)
    ncc_trace = np.clip(ncc_trace, -1.0, 1.0)
    return ncc_trace.astype(np.float32)

def build_template_features(cache_traces, cache_times, bhv_df, reward_code=2, pre_s=0.1, post_s=0.5, out_sr=1000.0, n_lfp_blocks_to_match=3):
    out_traces = {}
    out_times  = {}
    rwd_times = bhv_df.loc[bhv_df['code'] == reward_code, 'timestamp'].values
    if len(rwd_times) < 5:
        print('  [TEMPLATE] Not enough reward events, skipping.')
        return out_traces, out_times
        
    def infer_sr_from_times(t_arr):
        if t_arr is None or len(t_arr) < 2: return out_sr
        return 1.0 / float(np.median(np.diff(t_arr[:min(10000, len(t_arr))])))
        
    def resample_to(trace, t_arr, target_sr):
        src_sr = infer_sr_from_times(t_arr)
        if src_sr <= target_sr * 1.05: return trace, t_arr
        factor = int(round(src_sr / target_sr))
        if factor <= 1: return trace, t_arr
        ds = trace[::factor].copy()
        ts = t_arr[::factor] if t_arr is not None else np.arange(len(ds)) / target_sr
        return ds, ts
        
    def extract_windows(trace, t_arr, events, pre, post):
        sr  = infer_sr_from_times(t_arr)
        pre_n  = int(pre * sr)
        post_n = int(post * sr)
        windows = []
        for t in events:
            idx = np.searchsorted(t_arr, t)
            s, e = idx - pre_n, idx + post_n
            if s >= 0 and e <= len(trace):
                windows.append(trace[s:e])
        if not windows: return np.empty((0, pre_n + post_n))
        try:
            return np.vstack(windows)
        except ValueError:
            return np.empty((0, pre_n + post_n))
            
    def build_and_match(fname, trace, t_arr):
        tr_ds, t_ds = resample_to(trace, t_arr, out_sr)
        sr_ds = infer_sr_from_times(t_ds)
        wins = extract_windows(tr_ds, t_ds, rwd_times, pre_s, post_s)
        if wins.shape[0] < 5:
            print(f'    [SKIP] {fname}: only {wins.shape[0]} windows')
            return
        template = wins.mean(axis=0)
        print(f'    NCC: {fname}  template_len={len(template)}  n_trials={wins.shape[0]}  signal_len={len(tr_ds)}')
        ncc = fast_ncc(tr_ds.astype(float), template.astype(float))
        ncc_z = z_score(ncc)
        key = f'tmpl_{fname}'
        out_traces[key] = ncc_z
        out_times[key]  = t_ds
        print(f'    → {key}')
        
    # ── Solenoid derivative ──
    if 'solenoid_derivative' in cache_traces:
        t_arr = cache_times.get('solenoid_derivative')
        if t_arr is None:
            n = len(cache_traces['solenoid_derivative'])
            t_arr = np.arange(n) / 30000.0
        print('\n  [TEMPLATE] solenoid_derivative')
        build_and_match('solenoid_derivative', cache_traces['solenoid_derivative'], t_arr)
        
    # ── Best LFP blocks ──
    lfp_block_names = sorted(
        [k for k in cache_traces if k.startswith('lfp_block_') and '_' not in k.replace('lfp_block_', '')],
        key=lambda x: int(x.split('_')[-1])
    )
    for bi, bname in enumerate(lfp_block_names[:n_lfp_blocks_to_match]):
        t_arr = cache_times.get(bname)
        if t_arr is None:
            n = len(cache_traces[bname])
            t_arr = np.arange(n) / 2500.0
        print(f'\n  [TEMPLATE] {bname}')
        build_and_match(bname, cache_traces[bname], t_arr)
        
    return out_traces, out_times

# ──────────────────────────────────────────────────────────────────────────────
# Behaviour loader
# ──────────────────────────────────────────────────────────────────────────────
def load_bhv(path):
    df = pd.read_csv(str(path), header=None, names=['timestamp', 'code'])
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['code']      = pd.to_numeric(df['code'],      errors='coerce')
    df = df.dropna()
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    return df

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Extract extra features (bands, PCs, AP blocks, templates) '
                    'and append to an existing feature cache.')
    parser.add_argument('--cache',    required=True,                        help='.npz feature cache from ephys_alignment_fusion.py')
    parser.add_argument('--ap',       default=None,                         help='SpikeGLX AP .bin file (required unless --skip-ap)')
    parser.add_argument('--lf',       default=None,                         help='SpikeGLX LF .bin file (required unless --skip-lfp)')
    parser.add_argument('--bhv',      default=None,                         help='GlobalLogInt*.csv (required for templates)')
    parser.add_argument('--bands',    nargs='+', default=['delta', 'theta', 'beta', 'gamma'], help='LFP band names to extract (default: all 4)')
    parser.add_argument('--ap-bands', nargs='+', default=['mua', 'hfo', 'gamma'],             help='AP band names to extract (default: mua hfo gamma)')
    parser.add_argument('--n-pcs',    type=int, default=3,                  help='Number of PCs to extract per block (default: 3)')
    parser.add_argument('--block-size', type=int, default=20,               help='Channels per depth block (default: 20)')
    parser.add_argument('--chunk-sec', type=float, default=60.0,            help='Seconds per read chunk (default: 60)')
    parser.add_argument('--out-sr',   type=float, default=1000.0,           help='Output sample rate for LFP/AP features (default: 1000 Hz)')
    parser.add_argument('--n-tmpl-blocks', type=int, default=3,             help='Number of LFP blocks to template-match (default: 3)')
    
    parser.add_argument('--skip-lfp-bands',      action='store_true')
    parser.add_argument('--skip-ap-blocks',      action='store_true')
    parser.add_argument('--skip-solenoid-depth', action='store_true')
    parser.add_argument('--skip-templates',      action='store_true')
    args = parser.parse_args()
    
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f'[ERROR] Cache not found: {cache_path}')
        sys.exit(1)
        
    print(f'\n{"="*60}')
    print(f' extract_extra_features.py')
    print(f' Cache: {cache_path.name}')
    print(f'{"="*60}\n')
    
    # ── Load existing cache ──
    print('Loading existing cache...')
    cache_traces, cache_times, sync, sr_ap_cache, sr_lf_cache = load_feature_cache(cache_path)
    print(f'  Existing features: {sorted(cache_traces.keys())}')
    
    all_new_traces = {}
    all_new_times  = {}
    
    # ── LFP band + PC extraction ──
    if not args.skip_lfp_bands:
        if args.lf is None:
            print('\n[SKIP] --lf not provided; skipping LFP bands.')
        elif not Path(args.lf).exists():
            print(f'\n[SKIP] LF file not found: {args.lf}')
        else:
            n_ch_lf, sr_lf, _, _ = get_bin_params(args.lf)
            sr_lf = sr_lf_cache if sr_lf_cache else sr_lf
            print(f'\nLF file: {Path(args.lf).name}  sr={sr_lf:.0f}Hz  n_ch={n_ch_lf}')
            t, tm = extract_lfp_bands_pcs(
                args.lf, sr_lf, n_ch_lf, n_pcs=args.n_pcs, band_names=args.bands, 
                block_size=args.block_size, chunk_sec=args.chunk_sec, out_sr=args.out_sr)
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'\n  LFP bands: +{len(t)} features')
            
    # ── AP block band extraction ──
    if not args.skip_ap_blocks:
        if args.ap is None:
            print('\n[SKIP] --ap not provided; skipping AP blocks.')
        elif not Path(args.ap).exists():
            print(f'\n[SKIP] AP file not found: {args.ap}')
        else:
            n_ch_ap, sr_ap, _, _ = get_bin_params(args.ap)
            sr_ap = sr_ap_cache if sr_ap_cache else sr_ap
            print(f'\nAP file: {Path(args.ap).name}  sr={sr_ap:.0f}Hz  n_ch={n_ch_ap}')
            t, tm = extract_ap_block_bands(
                args.ap, sr_ap, n_ch_ap, n_pcs=args.n_pcs, band_names=args.ap_bands, 
                block_size=args.block_size, chunk_sec=args.chunk_sec, out_sr=args.out_sr)
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'\n  AP blocks: +{len(t)} features')
            
    # ── Solenoid by depth ──
    if not args.skip_solenoid_depth:
        if args.ap is None:
            print('\n[SKIP] --ap not provided; skipping solenoid depth.')
        elif not Path(args.ap).exists():
            print(f'\n[SKIP] AP file not found.')
        else:
            n_ch_ap, sr_ap, _, _ = get_bin_params(args.ap)
            sr_ap = sr_ap_cache if sr_ap_cache else sr_ap
            t, tm = extract_solenoid_depth(
                args.ap, sr_ap, n_ch_ap, block_size=args.block_size, chunk_sec=args.chunk_sec)
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'\n  Solenoid depth: +{len(t)} features')
            
    # ── Template matching ──
    if not args.skip_templates:
        if args.bhv is None:
            print('\n[SKIP] --bhv not provided; skipping templates.')
        else:
            bhv_files = sorted(Path('.').glob(args.bhv)) if '*' in args.bhv else [Path(args.bhv)]
            if not bhv_files or not bhv_files[0].exists():
                bhv_files = [Path(args.bhv)]
            bhv_df = load_bhv(bhv_files[0])
            n_rwd  = (bhv_df['code'] == 2).sum()
            print(f'\n[TEMPLATE] BHV loaded: {n_rwd} reward events')
            
            combined = {**cache_traces, **all_new_traces}
            combined_t = {**cache_times, **all_new_times}
            t, tm = build_template_features(
                combined, combined_t, bhv_df, n_lfp_blocks_to_match=args.n_tmpl_blocks)
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'\n  Templates: +{len(t)} features')
            
    # ── Save to cache ──
    if not all_new_traces:
        print('\nNo new features extracted. Cache unchanged.')
        return
        
    print(f'\nSaving {len(all_new_traces)} new features to cache...')
    update_cache(cache_path, all_new_traces, all_new_times)
    
    print(f'\n{"="*60}')
    print(f' Done. New features added:')
    for k in sorted(all_new_traces.keys()):
        print(f'   {k}  ({len(all_new_traces[k])} samples)')
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()