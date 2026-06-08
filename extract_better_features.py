#!/usr/bin/env python3
"""
extract_better_features.py
==========================
Four families of high-quality neural features designed to be event-locked
rather than just high-variance.  All write into the same .npz cache format
used by ephys_alignment_fusion.py and extract_extra_features.py.

FEATURE FAMILIES
----------------
1. reward_modulation  — per-channel post/pre-reward power ratio (supervised)
   Keys: rmi_block_<N>_<band>_top<K>

2. car_mua            — MUA/HFO/gamma envelope AFTER common-average referencing
   Keys: car_block_<N>_<band>_pc{1,2,3}

3. spike_density      — threshold-crossing density (better MUA proxy than RMS)
   Keys: sdf_block_<N>_pc{1,2,3}

4. pac_index          — delta-phase × gamma-amplitude coupling
   Keys: pac_block_<N>

USAGE
-----
python extract_better_features.py \\
    --cache   /path/to/features_cache.npz \\
    --ap      /path/to/recording.ap.bin   \\
    --lf      /path/to/recording.lf.bin   \\
    --bhv     /path/to/GlobalLogInt*.csv  \\
    --block-size 20 \\
    --chunk-sec  60 \\
    --out-sr     1000

Skip individual families with:
    --skip-rmi --skip-car-mua --skip-sdf --skip-pac

Requires: numpy, scipy, sklearn, pandas
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from sklearn.decomposition import IncrementalPCA

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers  (self-contained, no dependency on other scripts)
# ──────────────────────────────────────────────────────────────────────────────

def _load_spikeglx_meta(bin_path):
    """Parse .meta companion file → dict."""
    p = Path(bin_path)
    meta_path = p.with_suffix('.meta')
    if not meta_path.exists():
        meta_path = Path(str(p) + '.meta')
    if not meta_path.exists():
        return {}
    meta = {}
    with open(meta_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
    return meta


def _get_bin_params(bin_path):
    """Return (n_channels, sample_rate, n_samples, uv_per_bit)."""
    meta      = _load_spikeglx_meta(bin_path)
    file_size = Path(bin_path).stat().st_size
    n_ch      = int(meta.get('nSavedChans', 385))
    sr        = float(meta.get('imSampRate', 30000.0))
    if 'lf.bin' in str(bin_path).lower() or 'lf' in str(bin_path).lower():
        sr = float(meta.get('imSampRate', 2500.0))
    n_samp    = file_size // (n_ch * 2)
    uv        = float(meta.get('imAiRangeMax', 0.6)) / 512.0 if meta else 1.0
    return n_ch, sr, n_samp, uv


def _read_chunk(bin_path, n_ch, start_samp, n_samp_req):
    """Read a chunk from a SpikeGLX binary → float32 array (n_ch × n_samp)."""
    bin_path = Path(bin_path).resolve()
    bin_str  = str(bin_path)
    if sys.platform == 'win32' and not bin_str.startswith('\\\\?\\'):
        bin_str = '\\\\?\\' + bin_str

    row_bytes  = int(n_ch) * 2
    file_size  = bin_path.stat().st_size
    total_samp = file_size // row_bytes
    end_samp   = min(start_samp + n_samp_req, total_samp)
    actual     = end_samp - start_samp
    if actual <= 0:
        return np.empty((n_ch, 0), dtype=np.float32)

    data = np.empty((actual, n_ch), dtype=np.int16)
    MINI = 50_000
    with open(bin_str, 'rb') as fh:
        fh.seek(int(start_samp) * row_bytes)
        for bs in range(0, actual, MINI):
            be  = min(actual, bs + MINI)
            buf = fh.read((be - bs) * row_bytes)
            if not buf:
                break
            raw = np.frombuffer(buf, dtype=np.int16).reshape(be - bs, n_ch)
            data[bs:be] = raw
    return data.T.copy().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path):
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


def save_cache(cache_path, traces, times, sync, sr_ap, sr_lf):
    payload = {}
    for k, v in traces.items():
        payload[f'trace_{k}'] = np.asarray(v, dtype=np.float32)
    for k, v in times.items():
        payload[f'time_{k}'] = np.asarray(v, dtype=np.float32)
    if sync  is not None: payload['sync_edge_times'] = sync
    if sr_ap is not None: payload['sr_ap'] = np.array([sr_ap])
    if sr_lf is not None: payload['sr_lf'] = np.array([sr_lf])

    tmp = str(cache_path).replace('.npz', '_tmp_better.npz')
    print(f'  [CACHE] Writing {len(traces)} traces …')
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, str(cache_path))
        print(f'  [CACHE] Saved → {Path(cache_path).name}')
    except Exception as e:
        print(f'  [ERROR] Save failed: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def merge_into_cache(cache_path, new_traces, new_times):
    existing_traces, existing_times, sync, sr_ap, sr_lf = load_cache(cache_path)
    existing_traces.update(new_traces)
    existing_times.update(new_times)
    save_cache(cache_path, existing_traces, existing_times, sync, sr_ap, sr_lf)


# ──────────────────────────────────────────────────────────────────────────────
# Signal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bandpass(lo, hi, sr, order=4):
    nyq = sr / 2.0
    lo_n = max(1e-4, min(lo / nyq, 0.9999))
    hi_n = max(1e-4, min(hi / nyq, 0.9999))
    if lo_n >= hi_n:
        raise ValueError(f'Bad band: {lo}–{hi} Hz @ sr={sr}')
    return butter(order, [lo_n, hi_n], btype='band', output='sos')


def _z(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s if s > 0 else 1.0)


def _downsample(x, factor):
    """Simple decimation (no anti-alias filter — caller bandpasses first)."""
    return x[::factor]


def _pca_on_blocks(ipcas, n_blocks, block_pcs, actual_sr, tag, n_pcs,
                   out_traces, out_times):
    """Assemble IncrementalPCA results into named traces."""
    for bi in range(n_blocks):
        if not block_pcs[bi]:
            continue
        assembled = np.vstack(block_pcs[bi])   # (n_time, n_pcs)
        t_arr = np.arange(assembled.shape[0]) / actual_sr
        for pc_i in range(assembled.shape[1]):
            pc_z = _z(assembled[:, pc_i].astype(np.float32))
            name = f'{tag}_block_{bi}' if pc_i == 0 else f'{tag}_block_{bi}_pc{pc_i + 1}'
            out_traces[name] = pc_z
            out_times[name]  = t_arr
            print(f'    → {name}  len={len(pc_z)}')


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


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 1 — Reward Modulation Index (supervised, LF-based)
# ══════════════════════════════════════════════════════════════════════════════

def extract_reward_modulation(lf_path, sr_lf, n_ch_lf, bhv_df,
                               block_size=20, chunk_sec=60.0,
                               pre_s=0.5, post_s=1.0,
                               top_k=5, band_names=None,
                               out_sr=1000.0):
    """
    For each channel compute post/pre power ratio around reward events.
    Select the top-K channels per block, run PCA on those, return the scores.

    This is the *only* supervised feature: it uses reward times to pick which
    channels are most modulated, then re-projects the full session.

    Parameters
    ----------
    top_k      : how many top channels per block to keep before PCA
    band_names : list of LFP band names to process (default all)
    pre_s      : baseline window before reward (seconds)
    post_s     : response window after reward (seconds)
    """
    if band_names is None:
        band_names = list(LFP_BANDS.keys())

    n_ch_data  = n_ch_lf - 1
    n_blocks   = max(1, n_ch_data // block_size)
    total_samp = int(Path(lf_path).stat().st_size // (n_ch_lf * 2))
    chunk_samp = int(chunk_sec * sr_lf)
    ds_factor  = max(1, int(sr_lf / out_sr))
    actual_sr  = sr_lf / ds_factor

    reward_times = bhv_df.loc[bhv_df['code'] == 2, 'timestamp'].values.astype(float)
    if len(reward_times) < 5:
        print('  [RMI] Fewer than 5 reward events — skipping.')
        return {}, {}

    pre_n  = int(pre_s  * actual_sr)
    post_n = int(post_s * actual_sr)

    out_traces = {}
    out_times  = {}

    for band_name in band_names:
        if band_name not in LFP_BANDS:
            continue
        lo, hi = LFP_BANDS[band_name]
        print(f'\n  [RMI] {band_name} ({lo}–{hi} Hz)')

        try:
            sos = _bandpass(lo, hi, sr_lf)
        except ValueError as e:
            print(f'    [SKIP] {e}')
            continue

        # ── Pass 1: collect full-session downsampled power per channel ──
        # Store as list of chunks; we'll concatenate later.
        # Shape after assembly: (n_ch_data, n_time_ds)
        ch_power_chunks = [[] for _ in range(n_ch_data)]
        n_chunks = int(np.ceil(total_samp / chunk_samp))
        start = 0
        for ci in range(n_chunks):
            data = _read_chunk(lf_path, n_ch_lf, start, chunk_samp)
            print(f'    pass1 chunk {ci+1}/{n_chunks}', end='\r')
            for ch in range(n_ch_data):
                sig   = sosfiltfilt(sos, data[ch].astype(np.float64))
                power = sig[::ds_factor] ** 2
                ch_power_chunks[ch].append(power.astype(np.float32))
            del data
            start += chunk_samp
        print()

        # Assemble per-channel power traces
        n_time = sum(len(x) for x in ch_power_chunks[0])
        t_arr  = np.arange(n_time) / actual_sr

        # ── Compute reward modulation index per channel ──
        rmi = np.zeros(n_ch_data, dtype=np.float32)
        for ch in range(n_ch_data):
            power_trace = np.concatenate(ch_power_chunks[ch])
            pre_vals, post_vals = [], []
            for rt in reward_times:
                idx = int(rt * actual_sr)
                if idx - pre_n >= 0 and idx + post_n < len(power_trace):
                    pre_vals.append(power_trace[idx - pre_n : idx].mean())
                    post_vals.append(power_trace[idx : idx + post_n].mean())
            if pre_vals:
                pre_mu  = np.median(pre_vals)
                post_mu = np.median(post_vals)
                # log ratio (handles very small baselines safely)
                rmi[ch] = float(np.log1p(post_mu) - np.log1p(pre_mu + 1e-9))

        # ── Per block: pick top-K channels, PCA on their power traces ──
        for bi in range(n_blocks):
            ch0 = bi * block_size
            ch1 = min(ch0 + block_size, n_ch_data)
            blk_rmi = rmi[ch0:ch1]
            n_avail = ch1 - ch0
            k       = min(top_k, n_avail)
            top_idx = np.argsort(blk_rmi)[-k:][::-1]  # highest RMI first
            selected_global = [ch0 + i for i in top_idx]

            # Stack selected channels → (n_time, k)
            mat = np.column_stack([
                np.concatenate(ch_power_chunks[ch]) for ch in selected_global
            ])
            mat = _z(mat.astype(np.float64))   # z-score each column

            n_pcs = min(3, k)
            if k < 2 or n_time < n_pcs + 1:
                continue

            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_pcs)
            scores = pca.fit_transform(mat)

            for pc_i in range(n_pcs):
                pc_z = _z(scores[:, pc_i].astype(np.float32))
                suffix = '' if pc_i == 0 else f'_pc{pc_i + 1}'
                name = f'rmi_block_{bi}_{band_name}{suffix}'
                out_traces[name] = pc_z
                out_times[name]  = t_arr
                print(f'    → {name}  len={len(pc_z)}  '
                      f'top_channels={selected_global}  '
                      f'rmi={blk_rmi[top_idx].round(3)}')

    return out_traces, out_times


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 2 — CAR-MUA (common-average referenced, then envelope)
# ══════════════════════════════════════════════════════════════════════════════

def extract_car_mua(ap_path, sr_ap, n_ch_ap, n_pcs=3,
                    block_size=20, chunk_sec=60.0,
                    band_names=None, out_sr=1000.0,
                    smooth_ms=25.0):
    """
    For each chunk:
      1. Subtract per-timepoint median across ALL channels (CAR) to kill
         common-mode artifacts (solenoid clicks, motion, electrical noise).
      2. Bandpass filter each block in the chosen band.
      3. Compute power envelope (rectified + smoothed).
      4. Downsample and accumulate for IncrementalPCA.

    CAR is the key difference from extract_extra_features.py — it removes
    the solenoid artifact that contaminates everything else.
    """
    if band_names is None:
        band_names = list(AP_BANDS.keys())

    n_ch_data  = n_ch_ap - 1
    n_blocks   = max(1, n_ch_data // block_size)
    total_samp = int(Path(ap_path).stat().st_size // (n_ch_ap * 2))
    chunk_samp = int(chunk_sec * sr_ap)
    ds_factor  = max(1, int(sr_ap / out_sr))
    actual_sr  = sr_ap / ds_factor
    smooth_samp = max(1, int((smooth_ms / 1000.0) * sr_ap))

    out_traces = {}
    out_times  = {}
    n_chunks   = int(np.ceil(total_samp / chunk_samp))

    for band_name in band_names:
        if band_name not in AP_BANDS:
            continue
        lo, hi = AP_BANDS[band_name]
        print(f'\n  [CAR-MUA] {band_name} ({lo}–{hi} Hz), {n_blocks} blocks')

        try:
            sos = _bandpass(lo, hi, sr_ap)
        except ValueError as e:
            print(f'    [SKIP] {e}')
            continue

        ipcas     = [IncrementalPCA(n_components=min(n_pcs, block_size))
                     for _ in range(n_blocks)]
        block_pcs = [[] for _ in range(n_blocks)]

        # ── Pass 1: fit PCA (need two passes so incremental PCA sees all data) ──
        start = 0
        for ci in range(n_chunks):
            data = _read_chunk(ap_path, n_ch_ap, start, chunk_samp)
            print(f'    fit  {ci+1}/{n_chunks}', end='\r')

            # CAR: subtract per-sample median across neural channels
            car = np.median(data[:n_ch_data], axis=0)
            car_data = data[:n_ch_data] - car   # (n_ch_data, n_samp)

            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = car_data[ch0:ch1]
                if blk.shape[0] < 2:
                    continue
                envs = np.zeros((blk.shape[0], blk.shape[1] // ds_factor),
                                dtype=np.float32)
                for ci2 in range(blk.shape[0]):
                    filt = sosfiltfilt(sos, blk[ci2].astype(np.float64))
                    sq   = filt ** 2
                    env  = np.sqrt(uniform_filter1d(sq, size=smooth_samp))
                    envs[ci2] = env[::ds_factor][:envs.shape[1]]
                ipcas[bi].partial_fit(envs.T)

            del data
            start += chunk_samp
        print()

        # ── Pass 2: transform ──
        start = 0
        for ci in range(n_chunks):
            data = _read_chunk(ap_path, n_ch_ap, start, chunk_samp)
            print(f'    xfm  {ci+1}/{n_chunks}', end='\r')

            car      = np.median(data[:n_ch_data], axis=0)
            car_data = data[:n_ch_data] - car

            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = car_data[ch0:ch1]
                if blk.shape[0] < 2:
                    continue
                envs = np.zeros((blk.shape[0], blk.shape[1] // ds_factor),
                                dtype=np.float32)
                for ci2 in range(blk.shape[0]):
                    filt = sosfiltfilt(sos, blk[ci2].astype(np.float64))
                    sq   = filt ** 2
                    env  = np.sqrt(uniform_filter1d(sq, size=smooth_samp))
                    envs[ci2] = env[::ds_factor][:envs.shape[1]]
                block_pcs[bi].append(ipcas[bi].transform(envs.T))

            del data
            start += chunk_samp
        print()

        _pca_on_blocks(ipcas, n_blocks, block_pcs, actual_sr,
                       f'car_{band_name}', n_pcs, out_traces, out_times)

    return out_traces, out_times


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 3 — Spike Density Function (threshold-crossing density)
# ══════════════════════════════════════════════════════════════════════════════

def extract_spike_density(ap_path, sr_ap, n_ch_ap, n_pcs=3,
                          block_size=20, chunk_sec=60.0,
                          out_sr=1000.0,
                          threshold_sd=4.0,
                          smooth_ms=10.0,
                          hp_hz=300.0):
    """
    Threshold-crossing density — a cleaner MUA proxy than RMS.

    For each channel:
      1. High-pass filter at hp_hz (300 Hz default) with CAR.
      2. Compute per-channel noise SD from the MAD estimator
         (robust to bursts): noise_sd = median(|x|) / 0.6745
      3. Count samples exceeding threshold_sd * noise_sd per ms bin.
      4. Sum across channels in each block → spike density function.
      5. Smooth with a Gaussian kernel.
      6. IncrementalPCA across blocks.

    Advantages over RMS:
      • Insensitive to large-amplitude artifacts that don't cross threshold
        consistently (solenoid, motion).
      • Natural firing-rate interpretation.
      • Sparse → PCA finds more structured axes.
    """
    n_ch_data  = n_ch_ap - 1
    n_blocks   = max(1, n_ch_data // block_size)
    total_samp = int(Path(ap_path).stat().st_size // (n_ch_ap * 2))
    chunk_samp = int(chunk_sec * sr_ap)
    ds_factor  = max(1, int(sr_ap / out_sr))
    actual_sr  = sr_ap / ds_factor
    smooth_samp_ds = max(1, int((smooth_ms / 1000.0) * actual_sr))

    try:
        hp_sos = _bandpass(hp_hz, min(sr_ap / 2.0 * 0.95, 7000.0), sr_ap)
    except ValueError as e:
        print(f'  [SDF] HP filter failed: {e}')
        return {}, {}

    print(f'\n  [SDF] {n_blocks} blocks, thr={threshold_sd} SD, '
          f'smooth={smooth_ms} ms, out_sr={out_sr} Hz')

    n_chunks  = int(np.ceil(total_samp / chunk_samp))
    ipcas     = [IncrementalPCA(n_components=min(n_pcs, block_size))
                 for _ in range(n_blocks)]
    block_pcs = [[] for _ in range(n_blocks)]

    # ── Estimate per-channel noise SD in first chunk ──
    print('    Estimating noise …')
    first_chunk = _read_chunk(ap_path, n_ch_ap, 0, min(chunk_samp, total_samp))
    car0 = np.median(first_chunk[:n_ch_data], axis=0)
    car_first = first_chunk[:n_ch_data] - car0
    ch_noise = np.zeros(n_ch_data, dtype=np.float32)
    for ch in range(n_ch_data):
        filt = sosfiltfilt(hp_sos, car_first[ch].astype(np.float64))
        # MAD-based noise estimator (robust)
        ch_noise[ch] = float(np.median(np.abs(filt)) / 0.6745)
    ch_noise = np.maximum(ch_noise, 1e-3)   # safety floor
    del first_chunk, car_first

    def _sdf_block(car_data, bi):
        """Compute spike density per block → (n_time_ds,) float32."""
        ch0 = bi * block_size
        ch1 = min(ch0 + block_size, n_ch_data)
        blk = car_data[ch0:ch1]
        counts = np.zeros(blk.shape[1], dtype=np.float32)
        for ci2, ch in enumerate(range(ch0, ch1)):
            filt   = sosfiltfilt(hp_sos, blk[ci2].astype(np.float64))
            thr    = threshold_sd * ch_noise[ch]
            counts += (np.abs(filt) > thr).astype(np.float32)
        # downsample by summing bins
        n_ds = len(counts) // ds_factor
        counts_ds = counts[:n_ds * ds_factor].reshape(n_ds, ds_factor).sum(axis=1)
        # smooth
        smoothed = gaussian_filter1d(counts_ds.astype(np.float64),
                                     sigma=smooth_samp_ds)
        return smoothed.astype(np.float32)

    # ── Pass 1: fit ──
    start = 0
    for ci in range(n_chunks):
        data = _read_chunk(ap_path, n_ch_ap, start, chunk_samp)
        print(f'    fit  {ci+1}/{n_chunks}', end='\r')
        car      = np.median(data[:n_ch_data], axis=0)
        car_data = data[:n_ch_data] - car

        blk_sdfs = []
        for bi in range(n_blocks):
            sdf = _sdf_block(car_data, bi)
            blk_sdfs.append(sdf)

        # Stack blocks: (n_time_ds, n_blocks) for PCA
        # But IncrementalPCA expects (n_samples, n_features) where
        # features = blocks and samples = timepoints.
        # We do one PCA across blocks (spatial), not time.
        # Actually we want PCA within each block across channels — so
        # we run per-block SDF as a single feature, then do multi-block PCA.
        # Reshape: feed each block's SDF to its own IPCA (single feature = 1 block),
        # which means we run PCA across time-slices. For n_pcs we need multiple
        # channels, so we pass the per-channel SDFs instead of the sum.

        # Re-compute per-channel SDF for the IPCA fit (not the summed version)
        for bi in range(n_blocks):
            ch0 = bi * block_size
            ch1 = min(ch0 + block_size, n_ch_data)
            n_ch_blk = ch1 - ch0
            if n_ch_blk < 2:
                continue
            ch_sdfs = np.zeros((n_ch_blk, len(blk_sdfs[bi])), dtype=np.float32)
            for ci2, ch in enumerate(range(ch0, ch1)):
                filt   = sosfiltfilt(hp_sos,
                                     (car_data[ch]).astype(np.float64))
                thr    = threshold_sd * ch_noise[ch]
                cnt    = (np.abs(filt) > thr).astype(np.float32)
                n_ds   = len(cnt) // ds_factor
                cnt_ds = cnt[:n_ds * ds_factor].reshape(n_ds, ds_factor).sum(1)
                sm     = gaussian_filter1d(cnt_ds.astype(np.float64),
                                           sigma=smooth_samp_ds)
                ch_sdfs[ci2] = sm[:len(blk_sdfs[bi])]
            ipcas[bi].partial_fit(ch_sdfs.T)   # (n_time_ds, n_ch_blk)

        del data
        start += chunk_samp
    print()

    # ── Pass 2: transform ──
    start = 0
    for ci in range(n_chunks):
        data = _read_chunk(ap_path, n_ch_ap, start, chunk_samp)
        print(f'    xfm  {ci+1}/{n_chunks}', end='\r')
        car      = np.median(data[:n_ch_data], axis=0)
        car_data = data[:n_ch_data] - car

        for bi in range(n_blocks):
            ch0 = bi * block_size
            ch1 = min(ch0 + block_size, n_ch_data)
            n_ch_blk = ch1 - ch0
            if n_ch_blk < 2:
                continue
            ref_len = None
            ch_sdfs = []
            for ch in range(ch0, ch1):
                filt   = sosfiltfilt(hp_sos, car_data[ch].astype(np.float64))
                thr    = threshold_sd * ch_noise[ch]
                cnt    = (np.abs(filt) > thr).astype(np.float32)
                n_ds   = len(cnt) // ds_factor
                cnt_ds = cnt[:n_ds * ds_factor].reshape(n_ds, ds_factor).sum(1)
                sm     = gaussian_filter1d(cnt_ds.astype(np.float64),
                                           sigma=smooth_samp_ds).astype(np.float32)
                if ref_len is None:
                    ref_len = len(sm)
                ch_sdfs.append(sm[:ref_len])
            mat = np.column_stack(ch_sdfs)   # (n_time_ds, n_ch_blk)
            block_pcs[bi].append(ipcas[bi].transform(mat))

        del data
        start += chunk_samp
    print()

    out_traces = {}
    out_times  = {}
    _pca_on_blocks(ipcas, n_blocks, block_pcs, actual_sr,
                   'sdf', n_pcs, out_traces, out_times)

    return out_traces, out_times


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 4 — Phase-Amplitude Coupling (PAC index)
# ══════════════════════════════════════════════════════════════════════════════

def extract_pac_index(lf_path, sr_lf, n_ch_lf,
                      block_size=20, chunk_sec=60.0,
                      out_sr=10.0,
                      phase_band=None, amp_band=None,
                      n_phase_bins=18):
    """
    Delta-phase × gamma-amplitude coupling (mean vector length, Canolty et al.)

    For each block:
      1. Extract delta phase (via Hilbert on bandpassed signal).
      2. Extract gamma amplitude envelope (via Hilbert).
      3. Mean vector length (MVL) = |mean(amp * exp(i*phase))| — continuous
         instantaneous PAC index.
      4. Smooth to out_sr (default 10 Hz to capture slow modulation).

    The PAC index naturally peaks when gamma bursts are phase-locked to delta,
    which can be reward-triggered even when raw power isn't.

    Parameters
    ----------
    phase_band : (lo, hi) Hz for phase signal  (default: delta 1–4 Hz)
    amp_band   : (lo, hi) Hz for amplitude env  (default: gamma 30–80 Hz)
    n_phase_bins: bins for circular mean (not stored, used for quality check)
    out_sr     : output sample rate in Hz (PAC is slow; 10 Hz is sufficient)
    """
    if phase_band is None:
        phase_band = LFP_BANDS['delta']    # 1–4 Hz
    if amp_band is None:
        amp_band   = LFP_BANDS['gamma']    # 30–80 Hz

    n_ch_data  = n_ch_lf - 1
    n_blocks   = max(1, n_ch_data // block_size)
    total_samp = int(Path(lf_path).stat().st_size // (n_ch_lf * 2))
    chunk_samp = int(chunk_sec * sr_lf)
    # PAC is very slow: decimate heavily
    ds_factor  = max(1, int(sr_lf / out_sr))
    actual_sr  = sr_lf / ds_factor
    n_chunks   = int(np.ceil(total_samp / chunk_samp))

    # Overlap needed so Hilbert edge effects don't corrupt boundaries
    # Rule of thumb: ≥ 3 cycles of lowest frequency
    overlap_samp = max(int(3.0 / phase_band[0] * sr_lf), int(1.0 * sr_lf))

    try:
        phase_sos = _bandpass(*phase_band, sr_lf)
        amp_sos   = _bandpass(*amp_band,   sr_lf)
    except ValueError as e:
        print(f'  [PAC] Filter error: {e}')
        return {}, {}

    print(f'\n  [PAC] phase={phase_band} Hz  amp={amp_band} Hz  '
          f'{n_blocks} blocks  out_sr={out_sr} Hz')
    print(f'        overlap={overlap_samp/sr_lf:.1f}s per chunk boundary')

    block_pac_chunks = [[] for _ in range(n_blocks)]

    start = 0
    for ci in range(n_chunks):
        core_start = ci * chunk_samp
        core_end   = min(total_samp, core_start + chunk_samp)
        load_start = max(0, core_start - overlap_samp)
        load_end   = min(total_samp, core_end + overlap_samp)
        pre_pad    = core_start - load_start
        post_pad   = load_end  - core_end

        data = _read_chunk(lf_path, n_ch_lf, load_start, load_end - load_start)
        print(f'    chunk {ci+1}/{n_chunks}', end='\r')

        for bi in range(n_blocks):
            ch0 = bi * block_size
            ch1 = min(ch0 + block_size, n_ch_data)
            blk = data[ch0:ch1].astype(np.float64)
            if blk.shape[0] < 1:
                continue

            # Average across channels in block before Hilbert
            # (less noise in the phase estimate than single channel)
            mean_sig = blk.mean(axis=0)

            # Phase signal
            phase_filt = sosfiltfilt(phase_sos, mean_sig)
            phase_env  = np.angle(hilbert(phase_filt))   # instantaneous phase

            # Amplitude signal
            amp_filt  = sosfiltfilt(amp_sos, mean_sig)
            amp_env   = np.abs(hilbert(amp_filt))        # amplitude envelope

            # PAC: instantaneous MVL = amp * exp(i*phase), take |mean| over window
            # We compute it sample-by-sample as a sliding estimate:
            # pac_t = |amp_env * exp(i * phase)|  (magnitude of complex product)
            # Then smooth to out_sr.
            pac_complex = amp_env * np.exp(1j * phase_env)
            # Smooth real and imaginary parts, then take magnitude
            smooth_len = max(1, int(0.5 / phase_band[0] * sr_lf))   # half cycle
            re_sm = uniform_filter1d(np.real(pac_complex), size=smooth_len)
            im_sm = uniform_filter1d(np.imag(pac_complex), size=smooth_len)
            pac_trace = np.sqrt(re_sm ** 2 + im_sm ** 2)

            # Trim overlap and downsample
            core_len = core_end - core_start
            pac_core = pac_trace[pre_pad : pre_pad + core_len]
            pac_ds   = pac_core[::ds_factor]
            block_pac_chunks[bi].append(pac_ds.astype(np.float32))

        del data
        start += chunk_samp
    print()

    out_traces = {}
    out_times  = {}
    for bi in range(n_blocks):
        if not block_pac_chunks[bi]:
            continue
        full = np.concatenate(block_pac_chunks[bi])
        full_z = _z(full)
        t_arr  = np.arange(len(full_z)) / actual_sr
        name   = f'pac_block_{bi}'
        out_traces[name] = full_z
        out_times[name]  = t_arr
        print(f'    → {name}  len={len(full_z)}  sr={actual_sr:.1f} Hz')

    # Also compute theta-gamma PAC as a bonus
    try:
        theta_sos = _bandpass(*LFP_BANDS['theta'], sr_lf)
        print(f'\n  [PAC] theta-gamma bonus pass …')

        block_tg_chunks = [[] for _ in range(n_blocks)]
        start = 0
        for ci in range(n_chunks):
            core_start = ci * chunk_samp
            core_end   = min(total_samp, core_start + chunk_samp)
            load_start = max(0, core_start - overlap_samp)
            load_end   = min(total_samp, core_end + overlap_samp)
            pre_pad    = core_start - load_start
            post_pad   = load_end  - core_end

            data = _read_chunk(lf_path, n_ch_lf, load_start, load_end - load_start)
            print(f'    tg chunk {ci+1}/{n_chunks}', end='\r')

            for bi in range(n_blocks):
                ch0 = bi * block_size
                ch1 = min(ch0 + block_size, n_ch_data)
                blk = data[ch0:ch1].astype(np.float64)
                if blk.shape[0] < 1:
                    continue
                mean_sig   = blk.mean(axis=0)
                phase_filt = sosfiltfilt(theta_sos, mean_sig)
                phase_env  = np.angle(hilbert(phase_filt))
                amp_filt   = sosfiltfilt(amp_sos, mean_sig)
                amp_env    = np.abs(hilbert(amp_filt))
                pac_c      = amp_env * np.exp(1j * phase_env)
                sl         = max(1, int(0.5 / LFP_BANDS['theta'][0] * sr_lf))
                re_sm      = uniform_filter1d(np.real(pac_c), size=sl)
                im_sm      = uniform_filter1d(np.imag(pac_c), size=sl)
                pac_trace  = np.sqrt(re_sm ** 2 + im_sm ** 2)
                core_len   = core_end - core_start
                pac_core   = pac_trace[pre_pad : pre_pad + core_len]
                pac_ds     = pac_core[::ds_factor]
                block_tg_chunks[bi].append(pac_ds.astype(np.float32))

            del data
            start += chunk_samp
        print()

        for bi in range(n_blocks):
            if not block_tg_chunks[bi]:
                continue
            full   = np.concatenate(block_tg_chunks[bi])
            full_z = _z(full)
            t_arr  = np.arange(len(full_z)) / actual_sr
            name   = f'pac_tg_block_{bi}'
            out_traces[name] = full_z
            out_times[name]  = t_arr
            print(f'    → {name}  len={len(full_z)}')

    except Exception as e:
        print(f'  [PAC] theta-gamma pass failed: {e}')

    return out_traces, out_times


# ──────────────────────────────────────────────────────────────────────────────
# Behaviour loader
# ──────────────────────────────────────────────────────────────────────────────

def load_bhv(path):
    df = pd.read_csv(str(path), header=None, names=['timestamp', 'code'])
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['code']      = pd.to_numeric(df['code'],      errors='coerce')
    df = df.dropna()
    # Check for ephys-start code 1000; otherwise use first timestamp
    sr_mask = df['code'] == 1000
    t0 = float(df['timestamp'][sr_mask].iloc[0]) if sr_mask.any() \
         else float(df['timestamp'].min())
    df['timestamp'] = df['timestamp'] - t0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract four families of high-quality event-locked neural '
                    'features and append them to an existing .npz cache.')

    parser.add_argument('--cache',       required=True,
                        help='.npz feature cache (will be updated in-place)')
    parser.add_argument('--ap',          default=None,
                        help='SpikeGLX AP .bin file (needed for car-mua, sdf)')
    parser.add_argument('--lf',          default=None,
                        help='SpikeGLX LF .bin file (needed for rmi, pac)')
    parser.add_argument('--bhv',         default=None,
                        help='GlobalLogInt*.csv (needed for rmi)')

    parser.add_argument('--block-size',  type=int,   default=20,
                        help='Channels per depth block (default: 20)')
    parser.add_argument('--chunk-sec',   type=float, default=60.0,
                        help='Seconds per I/O chunk (default: 60)')
    parser.add_argument('--out-sr',      type=float, default=1000.0,
                        help='Output sample rate for AP features in Hz (default: 1000)')
    parser.add_argument('--pac-out-sr',  type=float, default=10.0,
                        help='Output sample rate for PAC index in Hz (default: 10)')
    parser.add_argument('--n-pcs',       type=int,   default=3,
                        help='PCs per block for car-mua and sdf (default: 3)')
    parser.add_argument('--top-k',       type=int,   default=5,
                        help='Top channels per block for RMI (default: 5)')
    parser.add_argument('--threshold-sd', type=float, default=4.0,
                        help='Threshold for SDF in noise SD units (default: 4.0)')

    parser.add_argument('--ap-bands',   nargs='+',
                        default=['mua', 'hfo', 'gamma'],
                        help='AP bands for car-mua (default: mua hfo gamma)')
    parser.add_argument('--lf-bands',   nargs='+',
                        default=['delta', 'theta', 'beta', 'gamma'],
                        help='LFP bands for rmi (default: all)')

    parser.add_argument('--skip-rmi',     action='store_true',
                        help='Skip reward modulation index features')
    parser.add_argument('--skip-car-mua', action='store_true',
                        help='Skip CAR-MUA features')
    parser.add_argument('--skip-sdf',     action='store_true',
                        help='Skip spike density function features')
    parser.add_argument('--skip-pac',     action='store_true',
                        help='Skip phase-amplitude coupling features')

    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f'[ERROR] Cache not found: {cache_path}')
        sys.exit(1)

    print(f'\n{"="*64}')
    print(f'  extract_better_features.py')
    print(f'  Cache: {cache_path.name}')
    print(f'{"="*64}\n')

    all_new_traces = {}
    all_new_times  = {}

    # ── Resolve LF / AP params ──────────────────────────────────────────────
    lf_ok = args.lf and Path(args.lf).exists()
    ap_ok = args.ap and Path(args.ap).exists()

    if lf_ok:
        n_ch_lf, sr_lf, n_samp_lf, uv_lf = _get_bin_params(args.lf)
        print(f'LF  {Path(args.lf).name}  sr={sr_lf:.0f} Hz  '
              f'n_ch={n_ch_lf}  dur={n_samp_lf/sr_lf/60:.1f} min')
    if ap_ok:
        n_ch_ap, sr_ap, n_samp_ap, uv_ap = _get_bin_params(args.ap)
        print(f'AP  {Path(args.ap).name}  sr={sr_ap:.0f} Hz  '
              f'n_ch={n_ch_ap}  dur={n_samp_ap/sr_ap/60:.1f} min')

    # ── 1. Reward Modulation Index ───────────────────────────────────────────
    if not args.skip_rmi:
        if not lf_ok:
            print('\n[SKIP] RMI: --lf not provided or not found.')
        elif not args.bhv or not Path(args.bhv).exists():
            print('\n[SKIP] RMI: --bhv not provided or not found.')
        else:
            bhv_df = load_bhv(args.bhv)
            print(f'\n[1/4] Reward Modulation Index')
            t, tm = extract_reward_modulation(
                args.lf, sr_lf, n_ch_lf, bhv_df,
                block_size=args.block_size,
                chunk_sec=args.chunk_sec,
                top_k=args.top_k,
                band_names=args.lf_bands,
                out_sr=args.out_sr,
            )
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'  RMI: +{len(t)} features')

    # ── 2. CAR-MUA ──────────────────────────────────────────────────────────
    if not args.skip_car_mua:
        if not ap_ok:
            print('\n[SKIP] CAR-MUA: --ap not provided or not found.')
        else:
            print(f'\n[2/4] CAR-MUA (common-average referenced envelope PCA)')
            t, tm = extract_car_mua(
                args.ap, sr_ap, n_ch_ap,
                n_pcs=args.n_pcs,
                block_size=args.block_size,
                chunk_sec=args.chunk_sec,
                band_names=args.ap_bands,
                out_sr=args.out_sr,
            )
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'  CAR-MUA: +{len(t)} features')

    # ── 3. Spike Density Function ────────────────────────────────────────────
    if not args.skip_sdf:
        if not ap_ok:
            print('\n[SKIP] SDF: --ap not provided or not found.')
        else:
            print(f'\n[3/4] Spike Density Function (threshold-crossing density)')
            t, tm = extract_spike_density(
                args.ap, sr_ap, n_ch_ap,
                n_pcs=args.n_pcs,
                block_size=args.block_size,
                chunk_sec=args.chunk_sec,
                out_sr=args.out_sr,
                threshold_sd=args.threshold_sd,
            )
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'  SDF: +{len(t)} features')

    # ── 4. PAC Index ────────────────────────────────────────────────────────
    if not args.skip_pac:
        if not lf_ok:
            print('\n[SKIP] PAC: --lf not provided or not found.')
        else:
            print(f'\n[4/4] Phase-Amplitude Coupling index (delta×gamma + theta×gamma)')
            t, tm = extract_pac_index(
                args.lf, sr_lf, n_ch_lf,
                block_size=args.block_size,
                chunk_sec=args.chunk_sec,
                out_sr=args.pac_out_sr,
            )
            all_new_traces.update(t)
            all_new_times.update(tm)
            print(f'  PAC: +{len(t)} features')

    # ── Save ────────────────────────────────────────────────────────────────
    if not all_new_traces:
        print('\nNo new features extracted. Cache unchanged.')
        return

    print(f'\nMerging {len(all_new_traces)} new features into cache …')
    merge_into_cache(cache_path, all_new_traces, all_new_times)

    print(f'\n{"="*64}')
    print(f'  Done. New features:')
    for name in sorted(all_new_traces):
        print(f'    {name:<42}  {len(all_new_traces[name])} samples')
    print(f'{"="*64}\n')


if __name__ == '__main__':
    main()
