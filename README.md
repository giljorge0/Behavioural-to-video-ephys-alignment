# Behavioural-to-Ephys Temporal Alignment

**Champalimaud Learning Lab — Internship Project**  
*Temporal synchronisation of behavioural state-machine logs to Neuropixels electrophysiology (SpikeGLX) and high-speed video*

---

## Overview

In head-fixed reaching tasks, three independent clocks run simultaneously:

| Clock | Source | Typical rate |
|---|---|---|
| **Behaviour** | State-machine (`GlobalLog*.csv`) | event-driven, ms resolution |
| **Video** | High-speed camera (`Camera*.avi`) | 100–400 fps |
| **Ephys** | Neuropixels / SpikeGLX (`.ap.bin` / `.lf.bin`) | 30 kHz (AP), 2.5 kHz (LF) |

These clocks drift relative to each other. A single 75-minute session can accumulate more than 8 seconds of drift between the behaviour log and the camera. Without correction, trial-by-trial neural and kinematic analyses are misaligned by hundreds of milliseconds.

This repository solves the problem by finding a **natural anchor event** visible in both the behaviour log and the electrophysiology — analogous to a clapperboard in film — and using it to fit a robust linear warp:

```
corrected_ephys_time = stretch × behavior_time + offset
```

---

## Repository contents

| File | Purpose |
|---|---|
| `ephys_alignment_first.py` | Main alignment pipeline |
| `plot_audit.py` | Standalone visual validation tool (works from cache, no bin files needed) |
| `video_alignment_run_ready.py` | Video ↔ behaviour alignment (separate pipeline) |

---

## Pipeline — Ephys ↔ Behaviour (`ephys_alignment_first.py`)

### How it works

Four candidate features are extracted from the SpikeGLX binary files and evaluated independently. Each feature is a 1-D z-scored time series that should produce a detectable peak near every reward event:

| Feature | Signal | Band | Best for |
|---|---|---|---|
| **Solenoid artifact** | Electromagnetic transient when reward valve fires | AP (30 kHz) | Hardware-level anchor; present on all channels simultaneously |
| **LFP deflection** | Reward-triggered slow cortical potential; extracted via PCA | LF (1–100 Hz) | Best biological accuracy (~144 ms RMSE in validation) |
| **MUA envelope** | Rising edge of multiunit spiking activity | AP (300–3000 Hz RMS) | Useful when LFP is absent; noisier |
| **Lick-band power** | Rhythmic 5–8 Hz burst coinciding with reward licking | LF (5–8 Hz) | Speculative; rig-dependent |

The pipeline runs in two phases:

1. **Audit phase** — evaluates each feature independently. Reports hit rate (fraction of reward trials with a detectable peak), median timing error, and false peak rate. Lets you select the best feature before committing to alignment.
2. **Alignment phase** — detects the best candidate event per trial, fits a Huber-robust linear warp on a training set, and evaluates on a held-out test set.

### Validated results (Milka session 1)

| Feature | RMSE | Notes |
|---|---|---|
| LFP deflection | **144 ms** | Best. Clear reward-evoked cortical potential. |
| Solenoid (derivative) | 173 ms | Recommended fallback if LFP fails. |
| Solenoid (global) | 176 ms | Reliable across all tested sessions. |
| Solenoid (block) | 182 ms | Worse than global; depth-localised mode adds noise. |

---

## Standard Operating Procedure (SOP) for overnight batch runs

### Step 0 — Copy data to a local drive (critical for large files)

Running the pipeline directly on `.bin` files stored in a live Dropbox Team Folder or deeply nested network drive **will crash** with `[Errno 22] Invalid argument`. There are two causes:

- **Windows MAX\_PATH limit**: Windows blocks file access for paths longer than 260 characters. Dropbox paths routinely exceed this.
- **Dropbox sync locks**: The Dropbox engine holds file locks on `.bin` files during sync, preventing `numpy.memmap` from opening them.

**The fix** is to mirror the data locally before running:

```powershell
# Mirror one session folder to local disk (fast, skips already-copied files)
robocopy "D:\Learning Lab Dropbox\..." "C:\data_temp\session1" /E /Z /MT:8
```

Alternatively, map the Dropbox root to a short drive letter to duck under the path limit (no restart needed, resets on reboot):

```powershell
subst Z: "D:\Learning Lab Dropbox\Learning Lab Team Folder"
# Now use Z:\Patlab protocols\... instead of the full path
```

---

### Step 1 — Extract all features once (the slow step)

This reads the binary files, extracts all feature traces, and saves them to a lightweight `.npz` cache. **Run this once per session.** All subsequent runs load from cache instantly.

```powershell
python ephys_alignment_first.py `
  --bhv  "C:\data_temp\session1\GlobalLogInt2023-08-15T11_14_10.csv" `
  --ap   "C:\data_temp\session1\imec0\session_t0.imec0.ap.bin" `
  --lf   "C:\data_temp\session1\imec0\session_t0.imec0.lf.bin" `
  --out  "C:\results\session1_imec0" `
  --feature all --solenoid-mode all --lfp-mode both `
  --chunk-sec 60 `
  --extract-only
```

`--extract-only` exits after saving the cache without running alignment. Cache files are named `<stem>_features_cache.npz` and saved to `--out`.

**Typical runtimes:**
- AP extraction (78 GB, 384 ch, 56 min): ~45–90 min
- LF extraction (7 GB, 384 ch, 56 min): ~10–20 min
- All subsequent runs: seconds

---

### Step 2 — Audit the features visually

```powershell
python plot_audit.py `
  --cache "C:\results\session1_imec0\session_t0.imec0.ap_features_cache.npz" `
  --bhv   "C:\data_temp\session1\GlobalLogInt2023-08-15T11_14_10.csv" `
  --out   "C:\results\session1_imec0\audit_plots" `
  --threshold 2.5
```

This produces three plots per feature (no bin files needed):

- **Full session overview** — feature trace with reward times (red lines), detected peaks (blue dots), and the largest inter-reward interval highlighted in gold
- **Zoom around largest ISI gap** — the most discriminative landmark for manual validation
- **Error histogram** — distribution of detection latency relative to reward time

Adjust `--threshold` to find where real biological peaks separate from noise. LFP features need threshold ≥ 2.5 after the z-scoring fix. Solenoid features typically need threshold ≥ 1.5.

---

### Step 3 — Run alignment with the best feature

```powershell
python ephys_alignment_first.py `
  --bhv  "C:\data_temp\session1\GlobalLogInt2023-08-15T11_14_10.csv" `
  --ap   "C:\data_temp\session1\imec0\session_t0.imec0.ap.bin" `
  --lf   "C:\data_temp\session1\imec0\session_t0.imec0.lf.bin" `
  --out  "C:\results\session1_imec0" `
  --feature lfp_deflection `
  --lfp-mode both `
  --chunk-sec 60 `
  --min-candidates 2 `
  --make-sync
```

The cache from Step 1 is detected automatically. The binary files are not re-read.

---

## Output files

| File | Contents |
|---|---|
| `<stem>_features_cache.npz` | 1-D feature traces + timestamps, saved after extraction |
| `sync_ephys_<stem>.json` | Warp parameters (`stretch`, `offset`, `rmse`) + per-trial aligned times |
| `syncfix_ephys_<stem>.mat` | MATLAB-compatible version of the sync JSON |
| `audit_plots/*.png` | Visual validation plots from `plot_audit.py` |

### JSON output schema

```json
{
  "warp": {
    "stretch": 0.9999968,
    "offset": 0.1649,
    "rmse": 0.1766
  },
  "events": [
    {
      "trial_idx": 0,
      "behavior_time": 1062.37,
      "ephys_time": 1062.54,
      "confidence": 0.91,
      "n_candidates": 4
    }
  ],
  "metadata": {
    "features_used": ["lfp_global", "lfp_block_0", "..."],
    "n_trials": 138,
    "n_detected": 131
  }
}
```

---

## CLI reference — `ephys_alignment_first.py`

### Inputs

| Argument | Required | Description |
|---|---|---|
| `--bhv` | yes | Behaviour log CSV (`GlobalLog*.csv`) |
| `--ap` | no | SpikeGLX AP band binary (`*.ap.bin` or `*.ap`) |
| `--lf` | no | SpikeGLX LF band binary (`*.lf.bin` or `*.lf`) |
| `--ks` | no | Kilosort output folder |
| `--gt` | no | Video-pipeline sync JSON for ground-truth comparison |
| `--out` | no | Output folder (default: same as `--bhv`) |

### Feature selection

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--feature` | `all` | `all`, `solenoid_artifact`, `lfp_deflection`, `mua_envelope`, `lick_band` | Which feature to extract and use |
| `--solenoid-mode` | `global` | `global`, `block`, `derivative`, `all` | Solenoid extraction variant. `all` extracts all three variants and saves each to cache |
| `--lfp-mode` | `both` | `random`, `global`, `block`, `both` | LFP extraction variant. `both` extracts global + all block traces |

**Solenoid variants:**
- `global` — median across all channels (common-mode rejection; good for widespread EM interference)
- `block` — independent median per 32-channel depth block (better spatial localisation)
- `derivative` — absolute first-difference of the global trace (isolates sharp onset transients; fewest false peaks)

**LFP variants:**
- `global` — PCA across all 384 channels simultaneously (one trace)
- `block` — independent PCA per 20-channel block (19 traces, one per region)
- `both` — saves both global and all block traces to cache

### Alignment

| Argument | Default | Description |
|---|---|---|
| `--min-conf` | `0.3` | Minimum confidence to accept an event detection |
| `--min-candidates` | `2` | Minimum candidates per trial to trust confidence score. Single-candidate trials are rejected (a single weak peak always gets confidence 1.0) |
| `--train-frac` | `0.8` | Fraction of trials used for warp fitting |
| `--make-sync` | off | Write `sync_ephys_*.json` and `syncfix_ephys_*.mat` |
| `--sweep` | off | Run hyperparameter grid search over feature weight combinations |

### Memory and caching

| Argument | Default | Description |
|---|---|---|
| `--chunk-sec` | `60.0` | Seconds per loading chunk. Both AP and LF are processed in overlapping windows; reduce to 30 if RAM is tight. 60 s × 384 ch ≈ 200–400 MB per chunk |
| `--ap-channels` | all neural | Comma-separated AP channel indices to use (e.g. `0,1,2,...,63`). Excludes last channel (sync) by default |
| `--lf-channels` | all neural | Comma-separated LF channel indices |
| `--no-cache` | off | Ignore and overwrite any existing feature cache |
| `--extract-only` | off | Extract features and save cache, then exit. Use for overnight extraction before analysis |
| `--analysis-only` | off | Skip binary reading entirely; load from cache and run alignment only |

---

## CLI reference — `plot_audit.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--cache` | yes | — | Feature cache `.npz` file |
| `--bhv` | yes | — | Behaviour log CSV |
| `--out` | no | `<cache_dir>/audit_plots` | Output folder for PNG plots |
| `--features` | no | `all` | Comma-separated feature names to plot, or `all` |
| `--threshold` | no | `1.5` | Peak detection threshold in z-scores. Use `2.5` for LFP after z-scoring fix; `1.5` for solenoid |
| `--min-distance` | no | `0.15` | Minimum distance between peaks (seconds) |
| `--context-sec` | no | `60.0` | Seconds of context shown around the largest ISI gap in the zoom plot |

---

## Technical notes

### Memory architecture

Both AP (30 kHz) and LF (2.5 kHz) files are processed in overlapping chunks using `numpy.memmap`. The overlap (0.5 s for AP, 2.0 s for LF) prevents filter-edge artefacts at chunk boundaries. After all chunks are processed, traces are concatenated and z-scored globally across the full session.

For a 56-minute session with 384 channels:
- AP file: ~78 GB on disk, ~200–400 MB per 60-second chunk in RAM
- LF file: ~7 GB on disk, ~20–40 MB per 60-second chunk in RAM

### Z-scoring and the "invisible ceiling" fix

Previous versions of the code applied `normalize01()` before z-scoring, clipping signals to the 5th–95th percentile range and bounding output to [0, 1]. This meant z-scores could never exceed ~2.0, causing the detector to miss genuine reward-evoked deflections.

The current code z-scores directly from the median-filtered signal:

```python
m = np.nanmedian(pc1_smooth)
s = np.nanstd(pc1_smooth)
z = (pc1_smooth - m) / s
```

Real reward-evoked LFP deflections now reach z = 3–5. Use `--threshold 2.5` for LFP features.

### LFP signal direction

The reward-evoked LFP signal is typically a **negative-going** deflection (trough, not peak). The pipeline inverts LFP traces internally before peak-finding so that the trough becomes a detectable positive peak. `plot_audit.py` applies the same inversion automatically for any feature whose name starts with `lfp_`.

### Sync channel (channel 384)

The last channel of every SpikeGLX AP and LF file is a digital TTL channel that toggles on every behaviour event recorded in the `GlobalLog`. The pipeline reads this channel automatically and uses it as ground truth to validate alignment quality when present. Sessions where the sync channel is flat or missing fall back to feature-based alignment only.

The ISI-based edge matching algorithm (not sequential index matching) is used to pair sync pulses to behaviour events, making it robust to double-pulses (valve-open + valve-close within 2 seconds are collapsed to one event) and to different absolute time bases.

### Warp model

The alignment model is a linear warp fitted with Huber robust regression (scipy `least_squares` with `loss='huber'`), which down-weights outlier trials caused by missed detections or motion artefacts:

```
ephys_time = stretch × behavior_time + offset
```

Typical values: `stretch` ≈ 1.000 ± 0.001, `offset` ≈ 0.1–0.5 s (hardware delay from state machine to physical reward delivery).

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `[Errno 22] Invalid argument` | Windows MAX\_PATH limit (>260 chars) or Dropbox lock | Copy data locally with `robocopy`, or map to short drive with `subst Z: "D:\..."` |
| `Unable to allocate X GiB` | LF/AP file loaded all at once instead of chunked | Should not happen with current code. If it does, reduce `--chunk-sec` to 30 |
| `need at least one array to concatenate` | AP file is empty (recording aborted) | Check file size. Pipeline now skips empty files gracefully |
| `vector x must be greater than padlen` | LF file is too short for the bandpass filter | Recording shorter than ~1 second. Nothing to do; session is unusable |
| `[SYNC] Sync channel present but poor fit` | Sync edges and behaviour events don't match in count or pattern | This session's sync signal is broken — normal, this is exactly the use case for feature-based alignment |
| `hit_rate=0.000` in plot_audit | Time base mismatch between behaviour (absolute clock) and feature trace (ephys seconds) | `plot_audit.py` subtracts the first timestamp to normalise. Check that bhv CSV is the correct session |
| Cache loaded but features missing | Previous run crashed before LF extraction finished | Re-run with `--no-cache` to rebuild from scratch |

---

## Folder structure

```
ephys_data/
└── 4_Milka/
    └── 15082023_Milka_StrCer_S1_g0/
        ├── 15082023_Milka_StrCer_S1_g0_imec0/
        │   ├── 15082023_Milka_StrCer_S1_g0_t0.imec0.ap.bin   # 78 GB
        │   ├── 15082023_Milka_StrCer_S1_g0_t0.imec0.ap.meta
        │   ├── 15082023_Milka_StrCer_S1_g0_t0.imec0.lf.bin   # 7 GB
        │   └── 15082023_Milka_StrCer_S1_g0_t0.imec0.lf.meta
        └── 15082023_Milka_StrCer_S1_g0_imec1/
            └── ...

behavior_data/
└── 4_Milka/
    └── R1/
        ├── GlobalLogInt2023-08-15T11_14_10.csv   # behaviour events + codes
        └── Camera1Timestamp2023-08-15T11_14_10.csv

results/
└── master_run/
    └── Milka_R1_imec0/
        ├── 15082023_..._features_cache.npz        # extracted traces (seconds to load)
        ├── sync_ephys_15082023_...json             # warp + aligned times
        ├── syncfix_ephys_15082023_...mat           # MATLAB version
        └── audit_plots/
            ├── lfp_global_full_session.png
            ├── lfp_global_zoom_largest_gap.png
            ├── lfp_global_error_histogram.png
            └── ...
```

---

## Dependencies

```
numpy scipy matplotlib pandas scikit-learn scipy.io
```

Install with:
```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

Python 3.8+ required. Tested on Windows 10/11 with Anaconda.
