import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def plot_10min_code(z_trace, time_axis, event_times, code_name, feat_name, out_folder, color, secondary_times=None, secondary_name=""):
    """Generates a 10-minute plot highlighting ONLY the specified event code."""
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # Plot the continuous neural trace
    ax.plot(time_axis, z_trace, color='gray', linewidth=0.8, label=feat_name)
    
    # Add a zero-line for reference (helps see dips vs swells)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    
    # Plot the primary event codes
    for i, t in enumerate(event_times):
        ax.axvline(t, color=color, linestyle='-', alpha=0.9, linewidth=1.5, label=f'Code {code_name}' if i==0 else "")
        
    # Plot secondary event codes (used for Code 3 overlaying Code 2)
    if secondary_times is not None:
        for i, t in enumerate(secondary_times):
            ax.axvline(t, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Code {secondary_name}' if i==0 else "")

    # Formatting
    tmin, tmax = time_axis[0], time_axis[-1]
    ax.set_title(f"{feat_name} | {tmin}-{tmax}s | Event: {code_name}")
    ax.set_xlabel('Session Time (s)')
    ax.set_ylabel('Z-Score')
    ax.set_xlim(tmin, tmax)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    filename = f"{feat_name}_{code_name}.png".replace(" ", "_")
    plt.savefig(os.path.join(out_folder, filename), dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--bhv', required=True)
    parser.add_argument('--out', default='audit_plots_shapes')
    parser.add_argument('--features', type=str, default='solenoid_derivative,lfp_global')
    parser.add_argument('--tmin', type=float, default=1500.0)
    parser.add_argument('--tmax', type=float, default=2000.0)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Loading cache: {os.path.basename(args.cache)}")
    cache = np.load(args.cache)
    
    # Load and filter behavior
    bhv_df = pd.read_csv(args.bhv, header=None, names=["time", "code"])
    bhv_df = bhv_df[(bhv_df["time"] >= args.tmin) & (bhv_df["time"] <= args.tmax)]
    
    code2  = bhv_df[bhv_df["code"] == 2]["time"].values
    code3  = bhv_df[bhv_df["code"] == 3]["time"].values
    code31 = bhv_df[bhv_df["code"] == 31]["time"].values
    code11 = bhv_df[bhv_df["code"] == 11]["time"].values
    codem11= bhv_df[bhv_df["code"] == -11]["time"].values

    fs = 1000.0
    start_idx = int(args.tmin * fs)
    end_idx = int(args.tmax * fs)
    
    feat_list = [f.strip() for f in args.features.split(',')]
    
    for feat_name in feat_list:
        if feat_name not in cache:
            print(f"[WARN] {feat_name} not in cache, skipping.")
            continue
            
        print(f"Plotting separated codes for {feat_name}...")
        z_window = cache[feat_name][start_idx:end_idx]
        time_axis = np.arange(len(z_window)) / fs + args.tmin
        
        # 1. Code 2 (Solid Red) and Code 3 (Dashed Blue)
        plot_10min_code(z_window, time_axis, code2, "2_and_3", feat_name, args.out, color='red', secondary_times=code3, secondary_name="3")
        
        # 2. Code 31 (Cyan)
        plot_10min_code(z_window, time_axis, code31, "31_Reach", feat_name, args.out, color='cyan')
        
        # 3. Code 11 (Orange)
        plot_10min_code(z_window, time_axis, code11, "11_Pull", feat_name, args.out, color='orange')
        
        # 4. Code -11 (Magenta)
        plot_10min_code(z_window, time_axis, codem11, "Minus11_InvPull", feat_name, args.out, color='magenta')

    print(f"\nAll shape plots saved to: {args.out}")

if __name__ == '__main__':
    main()
