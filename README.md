# Behavioural-to-Video & Ephys Temporal Alignment

**Champalimaud Learning Lab — Internship Project**
*Temporal synchronisation of behavioural state-machine logs to high-speed video recordings and Neuropixels electrophysiology (SpikeGLX)*

---

## Overview

In head-fixed reaching tasks, three independent clocks run simultaneously:

| Clock | Source | Typical rate |
|---|---|---|
| **Behaviour** | State-machine (`GlobalLog*.csv`) | event-driven (ms resolution) |
| **Video** | High-speed camera (`Camera*.avi`) | 100–400 fps |
| **Ephys** | Neuropixels / SpikeGLX (`.ap.bin` / `.lf.bin`) | 30 kHz (AP), 2.5 kHz (LF) |

These clocks drift relative to each other. A single 75-minute session can accumulate more than 8 seconds of drift between the behaviour log and the camera. Without correction, trial-by-trial neural and kinematic analyses are misaligned by hundreds of milliseconds.

This repository contains two pipelines that solve this problem by finding a **natural anchor event** visible in each modality — analogous to a "clapperboard" in film — and using it to fit a robust linear warp:

```
corrected_time = stretch × behavior_time + offset
```

### Pipeline 1 — Video ↔ Behaviour (`video_alignment_run_ready.py`)

The anchor event is the **water droplet** that appears on the spout at reward delivery. It is detected optically using template matching + PCA across regions of interest (ROIs) on the mirror reflections visible in the camera frame.

### Pipeline 2 — Ephys ↔ Behaviour (`ephys_alignment_first.py`)

The anchor event is an **electrical or neural signature** time-locked to reward delivery, extracted directly from the SpikeGLX binary files. Four candidate features are evaluated:

1. **Solenoid artifact** — electromagnetic transient when the reward valve fires (gold standard; hardware-level timestamp)
2. **LFP deflection** — reward-triggered slow cortical potential (1–100 Hz)
3. **MUA envelope** — rising edge of multiunit activity (300–3000 Hz RMS)
4. **Lick-band power** — rhythmic 5–8 Hz burst coinciding with licking

---


```

### Key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--bhv` | required | Behaviour log CSV |
| `--video` | required | Full-frame camera AVI (not spatially cropped clips) |
| `--sync` | required | Camera timestamp CSV (space-separated, no header) |
| `--out` | bhv parent | Output folder |
| `--fps` | auto | Expected camera FPS (400 for this dataset) |
| `--frame-step` | 10 | Process 1 in every N frames |
| `--search-margin` | 50 | Template search radius (px) around ROI |
| `--resize` | 1.0 | Frame downscale factor (<1 for extra speed) |
| `--make-sync` | off | Write `sync_*.json` + `syncfix_*.mat` |
| `--sweep` | off | Run hyperparameter grid search |
| `--min-conf` | 0.35 | Minimum detection confidence |
| `--train-frac` | 0.8 | Fraction of trials used for warp fitting |
| `--roi` | none | Hard-coded ROI as `x0,y0,x1,y1` |


### Usage

```bash
# Single session, try all available features
python ephys_alignment_first.py \
  --bhv "/data/R1/GlobalLog.csv" \
  --ap  "/data/R1/probe0/R1.imec0.ap.bin" \
  --lf  "/data/R1/probe0/R1.imec0.lf.bin" \
  --out "/data/R1/ephys_out/" \
  --feature all

# Compare against video sync as ground truth
python ephys_alignment_first.py \
  --bhv "/data/R1/GlobalLog.csv" \
  --ap  "/data/R1/probe0/R1.imec0.ap.bin" \
  --lf  "/data/R1/probe0/R1.imec0.lf.bin" \
  --gt  "/data/R1/out_dryrun/sync_Camera1.json" \
  --out "/data/R1/ephys_out/" \
  --feature all --sweep

# Single feature, write canonical sync for a lost session
python ephys_alignment_first.py \
  --bhv "/data/R_lost/GlobalLog.csv" \
  --ap  "/data/R_lost/probe0/R_lost.imec0.ap.bin" \
  --out "/data/R_lost/ephys_out/" \
  --feature solenoid_artifact --make-sync

# Use specific channel subsets (faster for large probes)
python ephys_alignment_first.py \
  --bhv ... --ap ... --lf ... \
  --ap-channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --lf-channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --feature all --sweep
```

### Key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--bhv` | required | Behaviour log CSV |
| `--ap` | optional | SpikeGLX AP band binary (`*.ap.bin`) |
| `--lf` | optional | SpikeGLX LF band binary (`*.lf.bin`) |
| `--ks` | optional | Kilosort output folder |
| `--gt` | optional | Video sync JSON used as ground truth |
| `--feature` | `all` | Feature to extract: `all`, `solenoid_artifact`, `lfp_deflection`, `mua_envelope`, `lick_band` |
| `--min-conf` | 0.3 | Minimum confidence to accept a detection |
| `--train-frac` | 0.8 | Training fraction for warp fitting |
| `--make-sync` | off | Write `sync_ephys_*.json` + `syncfix_ephys_*.mat` |
| `--sweep` | off | Sweep feature weight combinations |
| `--ap-channels` | auto (first 64) | Comma-separated AP channel indices |
| `--lf-channels` | auto (first 64) | Comma-separated LF channel indices |

---


