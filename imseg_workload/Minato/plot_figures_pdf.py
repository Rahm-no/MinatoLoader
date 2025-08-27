import pandas as pd
import matplotlib.pyplot as plt

# Set global matplotlib styles (for font scaling, etc.)
plt.rcParams.update({
    'font.size': 40,
    'axes.titlesize': 40,
    'axes.labelsize': 40,
    'xtick.labelsize': 40,
    'ytick.labelsize': 40,
    'legend.fontsize': 38,
    'lines.markersize': 6,
    'lines.linewidth': 4,
    'legend.loc': 'upper left',
    'figure.titlesize': 26,
})

colors = {
    'PyTorch': '#1f77b4',
    'DALI': '#ff7f0e',
    'Speedy': '#2ca02c',
    'PECAN': '#d62728'
}

# ---------- Load and Preprocess Data ----------

# SpeedyLoader
df_speedy = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/speedy_v100_b4_50epochs_offandon.csv")
df_speedy['throughput(MBs)'] = (df_speedy['throughput(MBs)'] * 6) / df_speedy['time_diff']
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_speedy[col] = pd.to_numeric(df_speedy[col], errors='coerce')
df_avg_speedy = df_speedy.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_speedy = df_avg_speedy[df_avg_speedy['epoch'] <= 25]

# PyTorch
df_pytorch = pd.read_csv("/dl-bench/rnouaj/mlcomns_imseg/pytorch_v100_b4_4GPUs_pytorch_50ep.csv")
df_pytorch['throughput(MBs)'] = (df_pytorch['throughput(MBs)'] * 4) / df_pytorch['time_diff']
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_pytorch[col] = pd.to_numeric(df_pytorch[col], errors='coerce')
df_avg_pytorch = df_pytorch.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()

# DALI
df_dali = pd.read_csv("/home/2023/rnouaj/Baselines_Speedy/mlcomns_imseg_with_dali/pytorch_v100_b4_4GPUs_dali_50epochs.csv")
df_dali['throughput(MBs)'] = (df_dali['throughput(MBs)'] * 5) / df_dali['time_diff']
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_dali[col] = pd.to_numeric(df_dali[col], errors='coerce')
df_avg_dali = df_dali.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_dali = df_avg_dali[df_avg_dali['epoch'] <= 40]

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(20, 14), dpi=100)

# Plot each system
ax.plot(df_avg_pytorch['iteration_time'], df_avg_pytorch['throughput(MBs)'],
        marker='o', linestyle='-', label='PyTorch', color=colors['PyTorch'])

ax.plot(df_avg_dali['iteration_time'], df_avg_dali['throughput(MBs)'],
        marker='o', linestyle='-', label='DALI', color=colors['DALI'])

ax.plot(df_avg_speedy['iteration_time'], df_avg_speedy['throughput(MBs)'],
        marker='o', linestyle='-', label='SpeedyLoader', color=colors['Speedy'])

# Add vertical lines at end of each run
x_speedy = df_avg_speedy['iteration_time'].max()
x_pytorch = df_avg_pytorch['iteration_time'].max()
x_dali = df_avg_dali['iteration_time'].max()
for xpos in [x_speedy, x_pytorch, x_dali]:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=0.15)

# Labels and styling
ax.set_xlabel("Time (s)")
ax.set_ylabel("Throughput (MB/s)")
ax.set_ylim(bottom=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
# ax.legend(loc='best')

plt.tight_layout()
plt.savefig("imseg_V100_1_unified.pdf", bbox_inches='tight', dpi=100)
plt.show()
