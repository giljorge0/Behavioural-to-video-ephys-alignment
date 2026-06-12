import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d

def extract_event_windows(z_trace, times, event_times, pre_sec=1.0, post_sec=1.5):
    """Chops the continuous trace into windows matching exactly to ephys timestamps."""
    windows = []
    if len(times) < 2: return np.array([])
    
    fs = 1.0 / np.median(np.diff(times[:10000]))
    pre_samps = int(pre_sec * fs)
    post_samps = int(post_sec * fs)
    expected_len = pre_samps + post_samps
    
    for t in event_times:
        idx = np.searchsorted(times, t)
        start_idx = idx - pre_samps
        end_idx = idx + post_samps
        
        if start_idx >= 0 and end_idx < len(z_trace):
            win = z_trace[start_idx:end_idx]
            if len(win) == expected_len:
                windows.append(win)
            elif len(win) > expected_len:
                windows.append(win[:expected_len])
                
    if len(windows) == 0:
        return np.array([])
    return np.vstack(windows)

def plot_single_erp(windows, feat_name, code_title, file_suffix, out_folder, pre_sec, post_sec):
    n_trials, n_samps = windows.shape
    time_axis = np.linspace(-pre_sec, post_sec, n_samps)
    
    fig, (ax_heat, ax_mean) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    vmin, vmax = np.percentile(windows, [5, 95])
    im = ax_heat.imshow(windows, aspect='auto', origin='lower', cmap='RdBu_r', 
                        extent=[-pre_sec, post_sec, 0, n_trials], vmin=-vmax, vmax=vmax)
    
    ax_heat.set_title(f"{feat_name}\nEvent: {code_title} (n={n_trials})", fontsize=14)
    ax_heat.set_ylabel("Trial Number", fontsize=12)
    ax_heat.axvline(0, color='black', linestyle='--', linewidth=1.5)
    
    if "Code 2" in code_title:
        ax_heat.axvline(0.221, color='blue', linestyle=':', linewidth=2, label='+221ms (Water)')
        ax_heat.legend(loc='upper right')
        
    plt.colorbar(im, ax=ax_heat, label="Z-Score")
    
    mean_trace = np.mean(windows, axis=0)
    sem_trace = np.std(windows, axis=0) / np.sqrt(n_trials)
    
    ax_mean.plot(time_axis, mean_trace, color='black', linewidth=2)
    ax_mean.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color='gray', alpha=0.3)
    
    ax_mean.axvline(0, color='black', linestyle='--', linewidth=1.5)
    if "Code 2" in code_title:
        ax_mean.axvline(0.221, color='blue', linestyle=':', linewidth=2)
        
    ax_mean.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax_mean.set_xlim(-pre_sec, post_sec)
    ax_mean.set_xlabel("Time from Event (s)", fontsize=12)
    ax_mean.set_ylabel("Average Z-Score", fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(out_folder, f"{feat_name}_{file_suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--bhv', required=True)
    parser.add_argument('--sync', default=None, help='Path to sync.json')
    parser.add_argument('--out', default='audit_plots_individual_erps')
    parser.add_argument('--features', type=str, default='all')
    parser.add_argument('--pre', type=float, default=1.0, help='Seconds before event')
    parser.add_argument('--post', type=float, default=1.5, help='Seconds after event')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Loading cache: {os.path.basename(args.cache)}")
    cache = np.load(args.cache, allow_pickle=False)
    
    traces, times = {}, {}
    for key in cache.files:
        if key.startswith('trace_'):
            traces[key[6:]] = cache[key]
        elif key.startswith('time_'):
            times[key[5:]]  = cache[key]

    for key in cache.files:
        if key not in traces and key not in times and not key.startswith('time_') and key not in ['sync_edge_times', 'sr_ap', 'sr_lf']:
            traces[key] = cache[key]
            times[key] = np.arange(len(cache[key])) / 1000.0
    
    bhv_df = pd.read_csv(args.bhv, header=None, names=["time", "code"])
    bhv_df['time'] = pd.to_numeric(bhv_df['time'], errors='coerce')
    bhv_df['code'] = pd.to_numeric(bhv_df['code'], errors='coerce')
    bhv_df = bhv_df.dropna()
    
    sr_mask = bhv_df['code'] == 1000
    t0 = float(bhv_df['time'][sr_mask].iloc[0]) if sr_mask.any() else float(bhv_df['time'].iloc[0])
    bhv_df['time'] = bhv_df['time'] - t0
    
    if args.sync and os.path.exists(args.sync):
        print(f"Aligning behavior using: {os.path.basename(args.sync)}")
        with open(args.sync, 'r') as f:
            sync_data = json.load(f)
        interp_func = interp1d(sync_data['behavior_sync_times'], sync_data['ephys_sync_times'], bounds_error=False, fill_value="extrapolate")
        bhv_df['time'] = interp_func(bhv_df['time'])
    else:
        print("[WARN] No sync.json provided! Plotting with raw, unaligned drifting clocks.")
    
    events_to_plot = [
        ("Code 2 (Init & Water)", "Code_2", bhv_df[bhv_df["code"] == 2]["time"].values),
        ("Code 31 (Reach)", "Code_31", bhv_df[bhv_df["code"] == 31]["time"].values),
        ("Code 11 (Pull)", "Code_11", bhv_df[bhv_df["code"] == 11]["time"].values),
        ("Code -11 (Inv Pull)", "Code_Minus11", bhv_df[bhv_df["code"] == -11]["time"].values)
    ]
    
    if args.features == 'all':
        feat_list = list(traces.keys())
    elif args.features == 'better':
        feat_list = [k for k in traces.keys() if k.startswith(('car_', 'sdf_', 'rmi_', 'pac_'))]
    else:
        feat_list = [f.strip() for f in args.features.split(',')]
    
    for feat_name in feat_list:
        if feat_name not in traces:
            continue
            
        print(f"Processing {feat_name}...")
        z_trace = traces[feat_name]
        t_trace = times[feat_name]
        
        for code_title, file_suffix, event_times in events_to_plot:
            if len(event_times) == 0:
                continue
                
            windows = extract_event_windows(z_trace, t_trace, event_times, pre_sec=args.pre, post_sec=args.post)
            if len(windows) == 0:
                continue
                
            plot_single_erp(windows, feat_name, code_title, file_suffix, args.out, args.pre, args.post)

    print(f"\nAll ERP plots saved to: {args.out}")

if __name__ == '__main__':
    main()
