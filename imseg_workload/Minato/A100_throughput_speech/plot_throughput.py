import pandas as pd
import matplotlib.pyplot as plt

# Set global matplotlib styles (for font scaling, etc.)
plt.rcParams.update({
    'font.size': 62,
    'axes.titlesize': 62,
    'axes.labelsize': 62,
    'xtick.labelsize': 62,
    'ytick.labelsize': 62,
    'legend.fontsize': 64,
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
df_speedy = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/results_speech_onA100/speech_speedyA100_10s/speedy10s_test2.csv")
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_speedy[col] = pd.to_numeric(df_speedy[col], errors='coerce')
df_avg_speedy = df_speedy.groupby(['iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_speedy = df_avg_speedy[df_avg_speedy['iteration'] < 900]
# PyTorch
df_pytorch = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/results_speech_onA100/speech_pytorchA100_10s/speech_pytorchA100_10s_test2.csv")
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_pytorch[col] = pd.to_numeric(df_pytorch[col], errors='coerce')
df_avg_pytorch = df_pytorch.groupby(['iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_pytorch = df_avg_pytorch[df_avg_pytorch['iteration'] < 800]
# DALI
df_dali = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/results_speech_onA100/speech_daliA100_10s/speech_daliA100_10s_good.csv")
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_dali[col] = pd.to_numeric(df_dali[col], errors='coerce')
df_avg_dali = df_dali.groupby([ 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_dali["throughput(MBs)"] = df_avg_dali["throughput(MBs)"]*3

df_avg_dali = df_avg_dali[df_avg_dali['iteration'] < 750]
df_pecan = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/results_speech_onA100/speech_pytorchA100_10s/speech_pecantestA100_10s.csv")

df_avg_pecan= df_pecan.groupby([ 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()

df_avg_pecan = df_avg_pecan[df_avg_pecan['iteration'] < 800]
# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(20, 15), dpi=300)

# Plot each system
ax.plot(df_avg_pytorch['iteration_time'], df_avg_pytorch['throughput(MBs)'],
        marker='o', linestyle='-', label='PyTorch', color=colors['PyTorch'])
ax.plot(df_avg_pecan['iteration_time'], df_avg_pecan['throughput(MBs)'],
        marker='o', linestyle='-', label='PECAN', color=colors['PECAN'])

ax.plot(df_avg_dali['iteration_time'], df_avg_dali['throughput(MBs)'],
        marker='o', linestyle='-', label='DALI', color=colors['DALI'])

ax.plot(df_avg_speedy['iteration_time'], df_avg_speedy['throughput(MBs)'],
        marker='o', linestyle='-', label='Speedy', color=colors['Speedy'])


# Add vertical lines at end of each run
x_speedy = df_avg_speedy['iteration_time'].max()
x_pytorch = df_avg_pytorch['iteration_time'].max()
x_dali = df_avg_dali['iteration_time'].max()
x_pecan = df_avg_pecan['iteration_time'].max()
for xpos in [x_speedy, x_pytorch, x_dali, x_pecan]:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=0.15)


# Labels and styling
ax.set_xlabel("Time (s)")
# ax.set_ylabel("Throughput (MB/s)")
ax.set_ylim(bottom=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='best')
ax.tick_params(axis='both', pad =10, length=10, width=2)
plt.tight_layout()
plt.savefig("Speech10s_A100_.pdf", bbox_inches='tight', dpi=300)
plt.show()
