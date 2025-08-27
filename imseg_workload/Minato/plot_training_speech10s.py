import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data
gpu_counts_a100 = [1, 2, 3, 4]
gpu_counts_v100 = [2, 4, 6, 8]

training_data_v100 = {
    'PyTorch DataLoader': [5180.6, 2565.3, 1380.7, 1210],
    'Pecan': [5188.58, 2560.9, 1349.7, 1216],
    'DALI': [2254.59, 1144.7, 680.9, 595.8],
    'SpeedyLoader': [1071.69, 510, 350, 257.5],
}

# training_data_a100 = {
#     'PyTorch DataLoader': [1663.67, 1450.166, 1220.7, 1050.1],
#     'Pecan': [1658.98, 1341.10, 1215.320 ,1056.72],
#     'DALI': [1340.59, 1104.7, 980.9, 789.78],
#     'SpeedyLoader': [856.940, 684.44, 440.17, 276.294],
# }


training_data_a100 = {
    'PyTorch DataLoader': [3006.49, 2131.166, 1286.6, 989.438],
    'Pecan': [3058.98, 1881.10, 1256.320 ,995.20],
    'DALI': [2567.59,1515.1, 1029.9, 783.78],
   'SpeedyLoader': [856.940, 684.44, 440.17, 256.294],

}

# Plotting setup
hatch_patterns = ['//', 'xx', 'oo', '###']  # Match labels order
highlight_label = 'SpeedyLoader'
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
labels = ['PyTorch', 'Pecan', 'DALI', 'SpeedyLoader']
bar_width = 0.18
fontsize = 48
tick_pad = 10
yticks_a100 = [0, 1000, 2000, 3000,4000]
yticks_v100 = [0, 1000, 2000, 3000, 4000, 5000,6000]
# -------- V100 Figure --------
fig_v100, ax_v100 = plt.subplots(figsize=(12, 9), dpi=300)
x_v100 = np.arange(len(gpu_counts_v100))

for i, label in enumerate(labels):
    ax_v100.bar(
        x_v100 + i * bar_width,
        training_data_v100[label if label != 'PyTorch' else 'PyTorch DataLoader'],
        width=bar_width,
        color=colors[i],
        # hatch=hatch_patterns[i],
        # edgecolor='black',
        linewidth=1.5
    )

# Axis setup
ax_v100.set_xticks(x_v100 + 1.5 * bar_width)
ax_v100.set_xticklabels(gpu_counts_v100, fontsize=fontsize)
ax_v100.set_yticks(yticks_v100)
ax_v100.set_yticklabels(yticks_v100, fontsize=fontsize)
ax_v100.tick_params(axis='x', pad=tick_pad)
ax_v100.tick_params(axis='y', pad=tick_pad)
ax_v100.set_xlabel("Number of GPUs", fontsize=fontsize)
# ax_v100.set_ylabel("Training time (s)", fontsize=fontsize)
ax_v100.grid(axis='y', linestyle='--', alpha=0.6)

fig_v100.tight_layout()
fig_v100.savefig("speech10s_training_time_v100_only.pdf", bbox_inches='tight')

# -------- A100 Figure --------
fig_a100, ax_a100 = plt.subplots(figsize=(12, 9), dpi=300)
x_a100 = np.arange(len(gpu_counts_a100))

for i, label in enumerate(labels):
    ax_a100.bar(
            x_a100 + i * bar_width,
            training_data_a100[label if label != 'PyTorch' else 'PyTorch DataLoader'],
            width=bar_width,
            color=colors[i],
            # hatch=hatch_patterns[i],
            # edgecolor='black',
            linewidth=0.5
        )

# Axis setup
ax_a100.set_xticks(x_a100 + 1.5 * bar_width)
ax_a100.set_xticklabels(gpu_counts_a100, fontsize=fontsize)
ax_a100.set_yticks(yticks_a100)
ax_a100.set_yticklabels(yticks_a100, fontsize=fontsize)
ax_a100.tick_params(axis='x', pad=tick_pad)
ax_a100.tick_params(axis='y', pad=tick_pad)
ax_a100.set_xlabel("Number of GPUs", fontsize=fontsize)
# ax_a100.set_ylabel("Training time (s)", fontsize=fontsize)
ax_a100.grid(axis='y', linestyle='--', alpha=0.6)
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor=colors[0], label='PyTorch'),
    Patch(facecolor=colors[1], label='Pecan'),
    Patch(facecolor=colors[2],label='DALI'),
    Patch(facecolor=colors[3], label='Minato')  # No hatch
]

ax_a100.legend(
    handles=legend_handles,
    loc='best',
    borderaxespad=0,
    fontsize=48,
    ncol=1,
    frameon=True
)

fig_a100.tight_layout()
fig_a100.savefig("speech10s_training_time_a100_only.pdf", bbox_inches='tight')
