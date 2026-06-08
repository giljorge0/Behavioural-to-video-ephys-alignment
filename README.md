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

These clocks drift relative to each other. A single 75-minute session can accumulate more than 8 seconds of drift. Without correction, trial-by-trial neural and kinematic analyses are misaligned by hundreds of milliseconds.

This repository solves the problem by finding a **natural anchor event** visible in both the behaviour log and the electrophysiology — analogous to a clapperboard in film — and fitting a robust linear warp:

```
ephys_time = stretch × behavior_time + offset
```

---

## Repository contents

| File | Purpose |
|---|---|
| `ephys_alignment_fusion.py` | Main alignment pipeline (extraction + warp fitting) |
| `extract_extra_features.py` | Extended features: LFP bands × PCs, AP bands × PCs, solenoid by depth, template matching |
| `extract_better_features.py` | Advanced features: reward modulation index, CAR-MUA, spike density, phase-amplitude coupling |
| `feature_explorer.py` | ERP-style visual audit of any cache across all event codes |
| `plot_audit.py` | Threshold-sweep peak detection audit (works from cache, no bin files needed) |
| `plot_individual_erps.py` | Per-event heatmap + mean trace, optionally warp-aligned via sync JSON |
| `video_alignment_run_ready.py` | Video ↔ behaviour alignment (separate pipeline) |

---

## Workflow overview

```
bin files + GlobalLog CSV
        │
        ▼
ephys_alignment_fusion.py   ─── extraction + warp ──► sync_ephys_*.json / .mat
        │
        ├── extract_extra_features.py   (LFP bands, AP bands, templates)
        └── extract_better_features.py  (RMI, CAR-MUA, SDF, PAC)
                        │
                        ▼
              features_cache.npz
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
     feature_explorer.py    plot_audit.py
    (ERP grid per event)   (peak audit)
              │
              ▼
     plot_individual_erps.py
     (heatmap + mean trace,
      warp-aligned)
```

---

## Standard Operating Procedure

### Step 0 — Copy data to a local drive

Running against `.bin` files on a live Dropbox Team Folder **will crash** — two causes:

- **Windows MAX\_PATH limit**: paths longer than 260 characters trigger `[Errno 22] Invalid argument`
- **Dropbox sync locks**: the sync engine holds file locks that block `numpy`'s file reader

Copy with robocopy (copies both `.bin` and `.meta`, which are both required):

```powershell
robocopy "D:\Learning Lab Dropbox\...\session_imec0" "C:\data_temp\session_imec0" "*.bin" "*.meta" /MT:8 /Z /W:5 /R:3
```

---

### Step 1 — Extract all features (the slow step, run once per session)

```powershell
python ephys_alignment_fusion.py `
  --bhv  "C:\data_temp\GlobalLogInt*.csv" `
  --ap   "C:\data_temp\session_imec0\*.ap.bin" `
  --lf   "C:\data_temp\session_imec0\*.lf.bin" `
  --out  "C:\results\session_imec0" `
  --feature all --solenoid-mode all --lfp-mode both `
  --chunk-sec 30 `
  --ap-channels 0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368 `
  --lf-channels 0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,336,344,352,360,368,376 `
  --min-candidates 2 --make-sync --no-cache
```

The 24-channel AP subset and 48-channel LF subset keep peak RAM at ~200 MB per chunk, preventing `[WinError 8] Not enough memory` on machines without a large page file. For solenoid (EM artifact, uniform across the probe) and LFP PCA, this subset is sufficient.

**Cache file** is saved as `<stem>_features_cache.npz` in `--out`. All subsequent scripts load from this instantly.

**Typical runtimes** (78 GB AP, 7 GB LF, 56 min session, 24/48 channels, chunk-sec 30):
- AP extraction: ~30–60 min
- LF extraction: ~5–15 min
- All subsequent runs: seconds

---

### Step 2 — Optionally add extended features

**`extract_extra_features.py`** adds per-band × per-PC traces and template matching:

```powershell
python extract_extra_features.py `
  --cache "C:\results\session_imec0\*_features_cache.npz" `
  --ap    "C:\data_temp\session_imec0\*.ap.bin" `
  --lf    "C:\data_temp\session_imec0\*.lf.bin" `
  --bhv   "C:\data_temp\GlobalLogInt*.csv" `
  --bands delta theta beta gamma `
  --ap-bands mua hfo gamma `
  --n-pcs 3 --block-size 20 --chunk-sec 60
```

New cache keys added: `lfp_block_N_<band>_pc{1,2,3}`, `ap_block_N_<band>_pc{1,2,3}`, `sol_depth_N`, `tmpl_<code>_<feature>`.

**`extract_better_features.py`** adds four supervised/advanced feature families:

```powershell
python extract_better_features.py `
  --cache "C:\results\session_imec0\*_features_cache.npz" `
  --ap    "C:\data_temp\session_imec0\*.ap.bin" `
  --lf    "C:\data_temp\session_imec0\*.lf.bin" `
  --bhv   "C:\data_temp\GlobalLogInt*.csv" `
  --block-size 20 --chunk-sec 60 --n-pcs 3 --top-k 5
```

New cache keys added: `rmi_block_N_<band>_topK`, `car_block_N_<band>_pc{1,2,3}`, `sdf_block_N_pc{1,2,3}`, `pac_block_N`.

---

### Step 3 — Explore features visually on a 500 s chunk

`feature_explorer.py` generates ERP-style plots for every feature in the cache. Each plot is a 2×2 grid (event codes 2, 31, 11, −11): individual trial traces overlaid in grey, mean ± SEM in black. A consistent shape = good feature. A flat mess = noise.

```powershell
python feature_explorer.py `
  --cache "C:\results\session_imec0\*_features_cache.npz" `
  --bhv   "C:\data_temp\GlobalLogInt*.csv" `
  --out   "C:\results\session_imec0\explorer_1500_2000" `
  --tmin 1500 --tmax 2000 `
  --pre 1.0 --post 2.0
```

Outputs per feature: `erp_<feature>.png` (2×2 grid), `overview_<feature>.png` (full 500 s trace with events), `summary_heatmap.png` (all features × all codes).

---

### Step 4 — Peak audit

`plot_audit.py` detects peaks above a threshold and reports hit rate, false peak rate, and median timing error relative to reward events:

```powershell
python plot_audit.py `
  --cache "C:\results\session_imec0\*_features_cache.npz" `
  --bhv   "C:\data_temp\GlobalLogInt*.csv" `
  --out   "C:\results\session_imec0\audit_plots" `
  --threshold 1.5
```

Threshold guidance: `1.5` for solenoid and AP features; `2.5` for LFP features (after the z-scoring fix that removed `normalize01()`).

---

### Step 5 — Aligned ERP heatmaps

`plot_individual_erps.py` cuts the cache trace into per-trial windows and plots a heatmap (trial × time) plus mean trace. Optionally applies the warp from a sync JSON so behaviour times are mapped to ephys clock before windowing:

```powershell
python plot_individual_erps.py `
  --cache  "C:\results\session_imec0\*_features_cache.npz" `
  --bhv    "C:\data_temp\GlobalLogInt*.csv" `
  --sync   "C:\results\session_imec0\sync_ephys_*.json" `
  --out    "C:\results\session_imec0\erp_plots" `
  --features lfp_block_5_delta,car_block_5_mua_pc1 `
  --pre 1.0 --post 1.5
```

Without `--sync`, behaviour times are plotted as-is (clock drift visible as diagonal smearing in the heatmap). With `--sync`, the warp-corrected times are used.

---

## Feature families reference

### Standard features (`ephys_alignment_fusion.py`)

| Cache key | Source | Description |
|---|---|---|
| `solenoid_global` | AP | Median across all channels; EM artifact |
| `solenoid_block` | AP | Median per 32-channel depth block |
| `solenoid_derivative` | AP | Absolute first-difference of global (sharpest onset) |
| `lfp_global` | LF | PC1 across all channels, broadband 1–100 Hz |
| `lfp_block_N` | LF | PC1 of 20-channel depth block N, broadband |
| `mua_envelope` | AP | RMS envelope 300–3000 Hz, global |
| `lick_band` | LF | Hilbert envelope of 5–8 Hz band |
| `sol_depth_N` | AP | Median of depth block N (solenoid by region) |

### Extended features (`extract_extra_features.py`)

| Cache key | Source | Description |
|---|---|---|
| `lfp_block_N_<band>_pc{1,2,3}` | LF | PC1/2/3 of block N in band (delta/theta/beta/gamma) |
| `ap_block_N_<band>_pc{1,2,3}` | AP | PC1/2/3 of block N in band (mua/hfo/gamma) |
| `tmpl_<code>_<feature>` | cache | Normalised cross-correlation of feature vs mean ERP for event code |

### Advanced features (`extract_better_features.py`)

| Cache key | Source | Description |
|---|---|---|
| `rmi_block_N_<band>_topK` | LF | Post/pre reward power ratio for top-K channels (supervised) |
| `car_block_N_<band>_pc{1,2,3}` | AP | MUA/HFO/gamma envelope after common-average referencing |
| `sdf_block_N_pc{1,2,3}` | AP | Spike density (threshold-crossing rate) per block |
| `pac_block_N` | LF | Delta-phase × gamma-amplitude coupling index |

**Frequency bands:**

| Name | Range | Typical signal |
|---|---|---|
| `delta` | 1–4 Hz | Slow oscillations, sleep-related activity |
| `theta` | 4–8 Hz | Licking rhythm, spatial navigation |
| `beta` | 8–30 Hz | Motor planning, reward anticipation |
| `gamma` | 30–80 Hz | Local computation, sensory processing |
| `mua` | 300–3000 Hz | Multiunit spiking activity |
| `hfo` | 80–200 Hz | High-frequency oscillations |

---

## Output files

| File | Contents |
|---|---|
| `*_features_cache.npz` | All extracted 1-D feature traces + time axes |
| `sync_ephys_*.json` | Warp parameters + per-trial aligned times |
| `syncfix_ephys_*.mat` | MATLAB-compatible version of the sync JSON |
| `erp_<feature>.png` | 2×2 ERP grid from `feature_explorer.py` |
| `overview_<feature>.png` | 500 s continuous trace with all event codes |
| `summary_heatmap.png` | All features × all event codes, ERP score |
| `audit_plots/*.png` | Peak detection audit from `plot_audit.py` |

### Sync JSON schema

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
    "features_used": ["lfp_block_5_delta_pc1"],
    "n_trials": 336,
    "n_detected": 310
  }
}
```

---

## CLI reference

### `ephys_alignment_fusion.py`

| Argument | Default | Description |
|---|---|---|
| `--bhv` | required | Behaviour log CSV |
| `--ap` | optional | AP binary (`.ap.bin`) |
| `--lf` | optional | LF binary (`.lf.bin`) |
| `--out` | required | Output folder |
| `--feature` | `all` | `all`, `solenoid_artifact`, `lfp_deflection`, `mua_envelope`, `lick_band` |
| `--solenoid-mode` | `global` | `global`, `block`, `derivative`, `all` |
| `--lfp-mode` | `both` | `random`, `global`, `block`, `both` |
| `--chunk-sec` | `60.0` | Seconds per loading chunk |
| `--ap-channels` | all neural | Comma-separated channel indices (use 24-channel subset to avoid `WinError 8`) |
| `--lf-channels` | all neural | Comma-separated LF channel indices |
| `--min-conf` | `0.3` | Minimum confidence to accept a detection |
| `--min-candidates` | `2` | Minimum candidates per trial (rejects single-candidate trials) |
| `--train-frac` | `0.8` | Fraction of trials for warp fitting |
| `--make-sync` | off | Write sync JSON + MAT |
| `--no-cache` | off | Ignore existing cache and re-extract |
| `--extract-only` | off | Extract and cache, then exit |
| `--analysis-only` | off | Load cache and run alignment only, skip binary reads |

### `extract_extra_features.py`

| Argument | Default | Description |
|---|---|---|
| `--cache` | required | Existing `.npz` cache to merge into |
| `--ap` | optional | AP binary |
| `--lf` | optional | LF binary |
| `--bhv` | optional | Behaviour CSV (required for templates) |
| `--bands` | `delta theta beta gamma` | LFP band names |
| `--ap-bands` | `mua hfo gamma` | AP band names |
| `--n-pcs` | `3` | PCs to extract per block |
| `--block-size` | `20` | Channels per depth block |
| `--chunk-sec` | `60.0` | Seconds per read chunk |
| `--out-sr` | `1000.0` | Output sample rate (Hz) |
| `--skip-lfp-bands` | off | Skip LFP band extraction |
| `--skip-ap-blocks` | off | Skip AP block extraction |
| `--skip-solenoid-depth` | off | Skip solenoid-by-depth extraction |
| `--skip-templates` | off | Skip template matching |

### `extract_better_features.py`

| Argument | Default | Description |
|---|---|---|
| `--cache` | required | Existing `.npz` cache |
| `--ap` | optional | AP binary |
| `--lf` | optional | LF binary |
| `--bhv` | optional | Behaviour CSV (required for RMI) |
| `--block-size` | `20` | Channels per depth block |
| `--chunk-sec` | `60.0` | Seconds per chunk |
| `--out-sr` | `1000.0` | Output sample rate |
| `--n-pcs` | `3` | PCs per block |
| `--top-k` | `5` | Top-K channels for reward modulation index |
| `--threshold-sd` | `4.0` | Spike detection threshold (SDF feature) |
| `--ap-bands` | `mua hfo gamma` | AP bands for CAR-MUA |
| `--lf-bands` | `delta theta beta gamma` | LFP bands for RMI |
| `--skip-rmi` | off | Skip reward modulation index |
| `--skip-car-mua` | off | Skip common-average-referenced MUA |
| `--skip-sdf` | off | Skip spike density |
| `--skip-pac` | off | Skip phase-amplitude coupling |

### `feature_explorer.py`

| Argument | Default | Description |
|---|---|---|
| `--cache` | required | Feature cache `.npz` |
| `--bhv` | required | Behaviour CSV |
| `--out` | `<cache_dir>/explorer_out` | Output folder |
| `--tmin` | `1500.0` | Start of analysis window (session seconds) |
| `--tmax` | `2000.0` | End of analysis window |
| `--pre` | `1.0` | Seconds before event for ERP window |
| `--post` | `2.0` | Seconds after event |
| `--features` | `all` | Comma-separated names or `all` |
| `--skip-overview` | off | Skip continuous overview plots (faster) |

### `plot_audit.py`

| Argument | Default | Description |
|---|---|---|
| `--cache` | required | Feature cache `.npz` |
| `--bhv` | required | Behaviour CSV |
| `--out` | `<cache_dir>/audit_plots` | Output folder |
| `--features` | `all` | Feature names or `all` |
| `--threshold` | `1.5` | Peak detection threshold in z-scores |
| `--min-distance` | `0.15` | Minimum distance between peaks (seconds) |
| `--context-sec` | `60.0` | Context around largest ISI gap in zoom plot |

### `plot_individual_erps.py`

| Argument | Default | Description |
|---|---|---|
| `--cache` | required | Feature cache `.npz` |
| `--bhv` | required | Behaviour CSV |
| `--sync` | optional | Sync JSON for warp-aligned event times |
| `--out` | `audit_plots_individual_erps` | Output folder |
| `--features` | `all` | Feature names, `all`, or `better` (only `car_`, `sdf_`, `rmi_`, `pac_` prefixes) |
| `--pre` | `1.0` | Seconds before event |
| `--post` | `1.5` | Seconds after event |

---

## Technical notes

### Memory architecture

All binary reads use direct file I/O (no `numpy.memmap`) in overlapping mini-batches of 5000 rows × n_channels × 2 bytes ≈ 3.8 MB per mini-batch. This eliminates `[WinError 8] Not enough memory` errors that occurred with `memmap` on machines with limited page file space, regardless of file size.

For a 78 GB AP file at 30 kHz, 384 channels, 60-second chunks:
- Old `memmap` approach: 693 MB virtual address reservation per chunk → `WinError 8`
- New direct-read approach: 3.8 MB peak per mini-batch → always succeeds
- Using the 24-channel subset: only the selected channels are kept → ~173 MB total per 60 s chunk

### Rolling pipeline pattern for overnight batch runs

Sessions are processed one at a time. Each session script: copies `.bin` + `.meta` files from Dropbox to a local temp folder, runs the Python pipeline, deletes the temp copy. This prevents RAM accumulation across sessions and sidesteps Dropbox file locks:

```powershell
# Spawns a fresh PowerShell process per session — memory fully reset between sessions
Start-Process powershell -ArgumentList "-File `".\Session_Script.ps1`"" -Wait -NoNewWindow
[System.GC]::Collect()
Start-Sleep -Seconds 30
```

### Z-scoring and the "invisible ceiling" fix

Previous code applied `normalize01()` before z-scoring, clipping signals to [0, 1] and capping z-scores at ≈2. The fix: z-score directly from the smoothed signal:

```python
m = np.nanmedian(pc1_smooth)
s = np.nanstd(pc1_smooth)
z = (pc1_smooth - m) / s
```

Reward-evoked LFP deflections now reach z = 3–5. Use `--threshold 2.5` for LFP features.

### LFP signal direction

Reward-evoked LFP is a negative-going deflection (trough). All scripts invert traces internally for any feature whose name starts with `lfp_`, so peak-finding works correctly without any extra configuration.

### Sync channel (channel 384)

The last channel of every SpikeGLX AP file is a digital TTL that toggles on every behaviour event. The pipeline reads this automatically and uses it as ground truth to validate the feature-based warp when present. ISI-based edge matching (not sequential index matching) is used — robust to double-pulses and to different absolute time bases between behaviour and ephys clocks.

### Warp model

```
ephys_time = stretch × behavior_time + offset
```

Fitted with Huber robust regression (`scipy.optimize.least_squares` with `loss='huber'`), which down-weights outlier trials from missed detections. Typical values: `stretch` ≈ 1.000 ± 0.001, `offset` ≈ 0.1–0.5 s.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `[Errno 22] Invalid argument` | Windows MAX\_PATH > 260 chars or Dropbox lock | Copy files locally with `robocopy "*.bin" "*.meta"` |
| `[WinError 8] Not enough memory` | `memmap` exhausted page file | Fixed in current code (direct file I/O). If still occurs, use `--chunk-sec 30` |
| `cannot reshape array of size 0` | `fh.read()` returned empty bytes | Old `\\?\` path prefix issue — current code uses plain `open()` |
| `need at least one array to concatenate` | AP/LF file is empty (aborted recording) | Pipeline skips gracefully with a warning |
| `vector x must be greater than padlen` | File too short for bandpass filter | Recording < 1 s; unusable |
| `[SYNC] Sync channel present but poor fit` | Sync edges don't match behaviour event count | Expected for corrupted sessions — feature-based alignment proceeds |
| `hit_rate=0.000` in plot_audit | Time base mismatch | `plot_audit.py` subtracts first timestamp automatically; check CSV is correct session |
| Cache missing features | Run crashed before LF finished | Re-run with `--no-cache` |
| `OverflowError: cannot convert float infinity to integer` | Solenoid time array has zero-diff entries → SR = inf | Fixed in `feature_explorer.py` (guards in `infer_sr` + downsampling for high-SR features) |

---

## Folder structure

```
ephys_data/
└── 4_Milka/
    └── 15082023_Milka_StrCer_S1_g0/
        ├── 15082023_Milka_StrCer_S1_g0_imec0/
        │   ├── *.ap.bin      # ~78 GB
        │   ├── *.ap.meta     # required alongside .bin
        │   ├── *.lf.bin      # ~7 GB
        │   └── *.lf.meta
        └── 15082023_Milka_StrCer_S1_g0_imec1/
            └── ...

behavior_data/
└── 4_Milka/R1/
    └── GlobalLogInt2023-08-15T11_14_10.csv

results/
└── master_run/
    └── Milka_R1_imec0/
        ├── *_features_cache.npz        # all extracted features
        ├── sync_ephys_*.json
        ├── syncfix_ephys_*.mat
        ├── audit_plots/
        ├── explorer_1500_2000/
        │   ├── erp_lfp_block_5_delta.png
        │   ├── overview_lfp_block_5_delta.png
        │   └── summary_heatmap.png
        └── erp_plots/
```

---

## Dependencies

```
numpy scipy matplotlib pandas scikit-learn
```

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

Python 3.8+ required. Tested on Windows 10/11 with Anaconda.
