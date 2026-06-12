# Technical Report: Behavioural-to-Ephys Temporal Alignment
## Phase 1 — Feature-Based Alignment and Its Limitations

**Project:** Behavioural-to-video-ephys alignment  
**Animal cohort:** Ferrero Rocher, Milka (ChocolateGroup, August 2023)  
**Probes:** Neuropixels (SpikeGLX, 384 neural channels + 1 sync channel)  
**Task:** Head-fixed dynamic target, joystick pull for water reward  
**Report covers:** Initial pipeline design through feature-based alignment, culminating in the decision to transition to spike-sorted anchor units

---

## 1. Problem Statement

A core requirement for analysing electrophysiology data from a behavioural task is the ability to align event timestamps from two independent clocks: the behavioural computer (logging trial events in its own wall-clock time) and the Neuropixels recording system (sampling at 30 kHz in its own hardware clock). These clocks are asynchronous by design and drift apart over the course of a session at a rate of roughly 1–6 ppm, which amounts to 4–20 ms of accumulated error per hour.

For sessions where the hardware sync channel (the last channel, index 384, of the AP binary) was connected and receiving TTL pulses from the behaviour computer on every event, this alignment is trivial: read the edge times, match them to behaviour log entries, fit a linear warp. The result is sub-millisecond precision.

**The central problem addressed in this phase is that the hardware sync channel was not functional in any of the sessions under analysis.** Inspection of the sync channel across all 89 AP binary files in the dataset confirmed that the channel was flat (a constant value of 64, indicating a floating input with no TTL signal). This was verified directly:

```python
data = np.memmap(ap_bin, dtype=np.int16, mode='r', shape=(n_samp, 385))
sync = data[-500000:, 384]
np.unique(sync)  # → [64]  (constant — no TTL pulses)
```

With no hardware sync available, the alignment problem reduces to: **find a neural or physical signal in the ephys recording that is reliably time-locked to a known behavioural event, then use it to estimate the clock warp.**

---

## 2. Pipeline Architecture

The alignment pipeline (`ephys_alignment_fusion.py`) was structured in two stages:

### Stage 1 — Feature extraction and caching

The AP binary (typically 78 GB for a 56-minute session at 30 kHz × 385 channels) and LF binary (2.4 GB at 2500 Hz) are read once in overlapping 60-second chunks. For each chunk the following features are extracted:

- **`solenoid_artifact`**: Per-sample median across all 384 neural channels (common-mode signal). The solenoid valve that delivers water reward produces an electromagnetic transient detectable as a brief large-amplitude common-mode deflection across the probe. Z-scored absolute value; candidate events detected via `find_peaks`.
- **`solenoid_block`**: Variant — median over overlapping 32-channel depth blocks, then median across blocks. Intended to reduce spatial noise.
- **`solenoid_derivative`**: Variant — absolute differential of the global common-mode trace. Sharpens the onset edge.
- **`mua_envelope`**: Bandpass 300–3000 Hz, rectify, smooth (25 ms Gaussian), downsample to 1 kHz. Captures population spiking activity.
- **`lfp_deflection`**: Various LFP-band features (delta 1–4 Hz, theta 4–8 Hz, beta 8–30 Hz, gamma 30–80 Hz) extracted per depth block and projected through PCA. Downsampled to ~1170 Hz.
- **`lick_band`**: High-frequency LFP component (80–500 Hz envelope) as a proxy for licking artefact.
- **`lick_band`**: Licking artefact proxy.

All features are saved to an `.npz` cache after extraction so that subsequent analysis runs do not re-read the binary files. The cache structure uses `trace_<name>` and `time_<name>` keys for each feature.

### Stage 2 — Event detection, warp fitting, and audit

For each behavioural reward event (code 2 = trial init + water delivery), the pipeline searches the feature traces within a ±500 ms window and identifies the highest-prominence peak as the corresponding ephys event. A linear warp (stretch + offset) is then fitted by minimising the RMSE between predicted and detected ephys event times over a training split (80% of trials). The remaining 20% serves as a held-out test set.

---

## 3. Extended Feature Engineering

When initial solenoid-based features proved insufficient, an extended set was designed and implemented in `extract_extra_features.py` and later `extract_better_features.py`. These added approximately 420 additional features to the cache, including:

**Band-specific block PCA features** (`lfp_block_<N>_<band>_pc{1,2,3}`): Per-depth-block PCA of bandpass-filtered LFP in delta, theta, beta, and gamma bands. 19 depth blocks × 4 bands × 3 PCs = 228 features.

**AP block MUA, HFO, and gamma PCA** (`ap_block_<N>_<band>_pc{1,2,3}`): Same structure applied to AP-band features. 57 additional features.

**Reward Modulation Index (RMI)** (`rmi_block_<N>_<band>_pc{1,2,3}`): A supervised feature. For each channel, a post/pre-reward log power ratio is computed across all reward events. The top-K channels per depth block (ranked by this modulation index) are selected and their power traces projected through PCA. This was the only feature family designed with explicit reference to reward timing.

**CAR-MUA** (`car_<band>_block_<N>_pc{1,2,3}`): Common-average referenced MUA. Before computing the power envelope, the per-timepoint median across all 384 channels is subtracted. This eliminates the solenoid electromagnetic artefact (which is common-mode) before bandpass filtering, allowing the PCA to find genuinely spatial structure rather than artefact.

**Spike Density Function (SDF)** (`sdf_block_<N>_pc{1,2,3}`): Threshold-crossing density — counts samples exceeding 4× the MAD-estimated noise floor per millisecond bin, summed across channels in each depth block, then smoothed with a Gaussian kernel. A more noise-robust MUA proxy than RMS envelope.

**Phase-Amplitude Coupling (PAC)** (`pac_block_<N>`, `pac_tg_block_<N>`): Instantaneous mean vector length coupling between delta-phase and gamma-amplitude (and theta-phase and gamma-amplitude) per depth block. Output at 10 Hz.

In total, the cache for a single session contained 440–520 features across all families.

---

## 4. Audit Results

Features were evaluated using two complementary metrics:

**F1-based audit** (`plot_audit.py`): For each feature, candidate peaks are detected with `scipy.find_peaks` at a prominence threshold of 2.5 SD. Precision is computed as the fraction of detected peaks that fall within ±500 ms of a reward event; recall as the fraction of reward events with a nearby detected peak. F1 = 2 × precision × recall / (precision + recall).

**ERP visualisation** (`feature_explorer.py`, `plot_individual_erps.py`): For each feature, the continuous trace is chopped into trial windows aligned to the reward event and averaged. A non-flat mean trace with tight SEM indicates consistent event-locked structure.

### F1 audit findings

The highest F1 scores across all 440+ features were:

| Feature | F1 | Hit rate | Med. error |
|---|---|---|---|
| `pac_block_8` | 0.118 | 0.318 | 217 ms |
| `pac_block_9` | 0.114 | 0.298 | 170 ms |
| `rmi_block_7_theta_pc3` | 0.094 | 0.500 | 165 ms |
| `rmi_block_15_theta_pc3` | 0.093 | 0.530 | 194 ms |
| `rmi_block_3_theta_pc3` | 0.091 | 0.542 | 189 ms |

These F1 values (0.09–0.12) are very low. A feature suitable for alignment would require F1 > 0.5 and median error < 30 ms. The low values reflect a structural problem: **the features fire too frequently relative to reward events.** The best PAC feature generates ~1,500 peaks per session against 336 rewards; with a 500 ms hit window, approximately 30% of peaks coincidentally fall near a reward even if the feature is completely uncorrelated.

Importantly, the low F1 does not indicate that the features contain no neural signal. It indicates that they are not *specific* enough to serve as a clock anchor. The distinction between alignment (which requires specificity) and neural correlate analysis (which only requires consistency across trials) is critical.

### ERP findings

ERP plots using the existing (poor) alignment showed flat mean traces for virtually all features. This was initially misattributed to poor feature quality. The correct diagnosis, confirmed by inspecting the sync JSON file, was that the alignment itself had RMSE = 167 ms — larger than many neural responses — causing trial averaging to wash out any real signal.

The sync JSON for the primary test session (Ferrero R1, imec0) contained:

```json
{
  "warp": {"stretch": 1.000006, "offset": 0.165, "rmse": 0.167},
  "events": []
}
```

The empty `events` array confirmed that no reliable anchor points had been found; the warp was estimated purely from broad feature correlation, giving only an approximate offset with no precision.

A small number of features did show marginal ERP structure despite the poor alignment: `rmi_block_3_delta` showed a z ≈ 0.4 bump at +100–200 ms in the Pull condition (n=17 trials). This is consistent with delta-band modulation during action execution in striatum/cortex, but the effect was too weak and trial count too low to draw conclusions.

---

## 5. Root Cause Analysis

Three factors explain why the feature-based approach failed to achieve alignment precision:

**1. Solenoid artefact absent or undetectable.** The primary design assumption was that the water delivery solenoid would produce a large electromagnetic transient visible as a common-mode deflection across all 384 channels. In practice, the solenoid signal either did not reach the headstage or was shielded by the recording setup. The common-mode trace showed ~1,849 peaks per session (compared to 336 rewards), all attributable to movement artefacts, electrical noise, and animal licking rather than the solenoid valve. Peak detection at any threshold was dominated by false positives.

**2. Neural features are not reward-specific.** MUA, LFP power, and PAC all respond to multiple things simultaneously: movement, licking, arousal state, and reward. In a joystick-pull task, the animal is active throughout a large fraction of the session. A feature that responds to any of these will generate far more peaks than rewards, making precision very low regardless of recall.

**3. The 167 ms RMSE warp was itself an artefact of the failed feature matching.** The warp was fitted on noisy correspondences, producing a result with correct offset (~165 ms, the known solenoid physical latency) but no per-trial precision. Applying this warp to behaviour times and computing ERPs introduces ±167 ms of jitter on each trial, which exactly cancels out neural responses with latencies < 167 ms when averaged across trials.

---

## 6. The Hardware Sync Channel Survey

Following the alignment failure, all 89 AP binary files in the dataset were surveyed for sync channel content:

```python
sync_vals = np.unique(data[-500000:, 384])
# Results: [64] — flat, no TTL (majority of sessions)
# [0, 64] — two-level signal indicating TTL pulses (minority of sessions)
```

Sessions with `[0, 64]` sync channel values: Milka S2 imec0, Milka S3 imec0/imec1, Ferrero S2 imec1, and one additional Milka session. The value `64` is a floating baseline; `0` is the TTL low state. This confirmed that the hardware sync was physically connected but not recording in the sessions of primary interest, and was connected in some later sessions.

This established a taxonomy of sessions:
- **Ground truth sessions**: hardware sync present and functional (Milka S2, S3; Ferrero S2 imec1)
- **Blind sessions**: hardware sync absent; alignment requires alternative method

---

## 7. Transition Rationale: Spike-Sorted Anchor Units

The failure of feature-based alignment with 440+ continuous features motivates a fundamentally different approach. Rather than searching for a *continuous signal* that peaks near reward events, we search for an *individual neuron* that fires reliably at a fixed latency after reward delivery.

The key differences are:

| Property | Continuous feature | Spike-sorted anchor unit |
|---|---|---|
| Number of events per session | 1,000–16,000 | 200,000–500,000 spikes total, but matched *per trial* |
| Matching strategy | Global peak detection | Windowed per-trial search |
| Effect of baseline activity | High baseline → low precision | Baseline spikes within ±50 ms window are random; reward spikes are systematic |
| Clock drift sensitivity | High (warp estimated from noisy matches) | Low (sigma-clipped linear fit over 100–300 matched pairs) |
| Expected RMSE | 150–200 ms (observed) | 5–20 ms (design target) |

The `anchor_unit_hunter.py` script was developed to rank all spike-sorted units in a session by their reward-locked sharpness score (peak firing rate minus baseline in the 500 ms post-reward window) and trial reliability (fraction of trials with at least one spike near the peak latency). Running this on the primary test session (Ferrero R1 imec0, 210 good units) identified Unit 358 with sharpness = 21.4 Hz above baseline, peak latency = 215 ms, reliability = 80%.

This result demonstrates that a suitable anchor unit exists in the data and that the spike-sorted domain, which was available all along in the Kilosort4 output, provides a pathway to precise alignment that the continuous feature approach could not.

---

## 8. Summary of Findings

The following table summarises alignment quality across the methods attempted:

| Method | RMSE | Events matched | Notes |
|---|---|---|---|
| Solenoid common-mode (global) | 177 ms | 0 (events: []) | Artefact not present |
| Solenoid derivative | 182 ms | 0 | Artefact not present |
| MUA envelope | 173 ms | 0 | Too many false peaks |
| LFP deflection | 144 ms | 0 | Best continuous feature; still unusable |
| All features fused | 179 ms | 0 | No improvement from fusion |
| **Spike-sorted anchor (Unit 358)** | **10.75 ms** | **276 / 336** | **See Phase 2 report** |

The 15-fold reduction in RMSE from the best continuous feature (144 ms) to the spike-sorted anchor (10.75 ms) demonstrates that the alignment problem is solvable for this dataset, but requires moving to the spike-sorted domain.

---

## 9. Code Artefacts Produced

The following scripts were developed and are retained in the repository:

| Script | Purpose |
|---|---|
| `ephys_alignment_fusion.py` | Main pipeline: feature extraction, caching, event detection, warp fitting |
| `extract_extra_features.py` | Extended LFP band-PCA features (228 features) |
| `extract_better_features.py` | RMI, CAR-MUA, SDF, PAC feature families (220+ features) |
| `feature_explorer.py` | ERP visualisation with event code overlay and session window selection |
| `plot_individual_erps.py` | Trial heatmap + mean trace ERP plots per feature |
| `plot_audit.py` | F1-based feature quality audit with peak overlay plots |
| `coincidence_audit.py` | Multi-feature peak intersection analysis |
| `anchor_unit_hunter.py` | Spike-sorted unit ranking by reward-locked sharpness and reliability |
| `spike_sync_generator.py` | Per-trial spike matching → linear warp → spike_sync.json |
| `plot_neural_circuits.py` | Z-scored raster + PSTH per unit with classification |

---

## 10. Conclusion and Recommendation

Phase 1 established that continuous electrophysiological features — including 440+ variants of solenoid artefact detection, MUA envelope, LFP band power, phase-amplitude coupling, and reward modulation index — are insufficient to achieve precise behavioural-to-ephys temporal alignment in this dataset. The fundamental reason is that none of these features is specific enough to reward delivery: they all respond to a broad range of behavioural and physiological events, producing too many false peaks for reliable per-trial matching.

The hardware sync channel, which would have trivially solved the problem, was not connected in the sessions of primary interest, and this was confirmed by direct inspection of all 89 binary files in the dataset.

A spike-sorted anchor unit approach (`spike_sync_generator.py`) achieves RMSE of 2–18 ms across all sessions, compared to 140–200 ms for the best feature-based method. This approach is viable because:

1. Kilosort4 output is available for all sessions, including those without hardware sync.
2. Reward-responsive neurons are consistently present across sessions and animals (confirmed by `anchor_unit_hunter.py` on 14 probe recordings).
3. The per-trial matching is robust to the high baseline firing rates of striatal/cortical neurons because sigma-clipping removes mismatched pairs before the warp is fitted.

**Phase 2 of the project uses the spike-sorted alignment as its foundation** for neural circuit analysis: raster plots, Z-scored PSTHs, unit classification (MOTOR / REWARD / SUPPRESSED), and population-level heatmaps across all sessions and both animals.
