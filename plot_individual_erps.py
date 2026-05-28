import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def extract_event_windows(z_trace, event_times, fs=1000.0, pre_sec=1.0, post_sec=1.5):
    """Chops the continuous trace into consistent windows around each event."""
    windows = []
    
    pre_samps = int(pre_sec * fs)
    post_samps = int(post_sec * fs)
    
    for t in event_times:
        idx = int(t * fs)
        start_idx = idx - pre_samps
        end_idx = idx + post_samps
        
        if start_idx >= 0 and end_idx < len(z_trace):
            windows.append(z_trace[start_idx:end_idx])
            
    if len(windows) == 0:
        return np.array([])
        
    return np.vstack(windows)

def plot_single_erp(windows, feat_name, code_title, file_suffix, out_folder, pre_sec, post_sec, fs=1000.0):
    """Generates a dedicated Heatmap + Mean Trace image for a single feature/code combo."""
    n_trials = windows.shape[0]
    time_axis = np.linspace(-pre_sec, post_sec, int((pre_sec + post_sec) * fs))
    
    fig, (ax_heat, ax_mean) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 1. HEATMAP
    vmin, vmax = np.percentile(windows, [5, 95])
    im = ax_heat.imshow(windows, aspect='auto', origin='lower', cmap='RdBu_r', 
                        extent=[-pre_sec, post_sec, 0, n_trials], vmin=-vmax, vmax=vmax)
    
    ax_heat.set_title(f"{feat_name}\nEvent: {code_title} (n={n_trials})", fontsize=14)
    ax_heat.set_ylabel("Trial Number", fontsize=12)
    ax_heat.axvline(0, color='black', linestyle='--', linewidth=1.5)
    
    # Add Code 3 reference line if this is Code 2
    if "Code 2" in code_title:
        ax_heat.axvline(0.221, color='blue', linestyle=':', linewidth=2, label='+221ms (Water)')
        ax_heat.legend(loc='upper right')
        
    plt.colorbar(im, ax=ax_heat, label="Z-Score")
    
    # 2. MEAN TRACE
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
    parser.add_argument('--out', default='audit_plots_individual_erps')
    parser.add_argument('--features', type=str, default='solenoid_derivative,lfp_global')
    parser.add_argument('--pre', type=float, default=1.0, help='Seconds before event')
    parser.add_argument('--post', type=float, default=1.5, help='Seconds after event')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Loading cache: {os.path.basename(args.cache)}")
    cache = np.load(args.cache)
    bhv_df = pd.read_csv(args.bhv, header=None, names=["time", "code"])
    
    # The 4 Codes to generate images for
    events_to_plot = [
        ("Code 2 (Init & Water)", "Code_2", bhv_df[bhv_df["code"] == 2]["time"].values),
        ("Code 31 (Reach)", "Code_31", bhv_df[bhv_df["code"] == 31]["time"].values),
        ("Code 11 (Pull)", "Code_11", bhv_df[bhv_df["code"] == 11]["time"].values),
        ("Code -11 (Inv Pull)", "Code_Minus11", bhv_df[bhv_df["code"] == -11]["time"].values)
    ]
    
    feat_list = [f.strip() for f in args.features.split(',')]
    
    for feat_name in feat_list:
        if feat_name not in cache:
            print(f"[WARN] {feat_name} not in cache, skipping.")
            continue
            
        print(f"\nProcessing {feat_name}...")
        z_trace = cache[feat_name]
        
        for code_title, file_suffix, event_times in events_to_plot:
            if len(event_times) == 0:
                print(f"  Skipping {code_title}: No events found.")
                continue
                
            windows = extract_event_windows(z_trace, event_times, pre_sec=args.pre, post_sec=args.post)
            if len(windows) == 0:
                continue
                
            plot_single_erp(windows, feat_name, code_title, file_suffix, args.out, args.pre, args.post)

    print(f"\nAll ERP plots saved to: {args.out}")

if __name__ == '__main__':
    main()