
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data
gpu_counts_a100 = [1, 2, 3, 4]
gpu_counts_v100 = [2, 4, 6, 8]
# training_data_a100 = {
#     'PyTorch DataLoader': [1698.12, 1228.805, 1075.095, 847.135],
#     'DALI': [1154.555, 874.38, 543.60, 415.441],
#     'SpeedyLoader': [532, 390, 259, 182]
# }

training_data_a100 = {
    'PyTorch DataLoader': [4036.24, 2457.61, 1750.197, 1294.27],
    'DALI': [1154.555, 874.38, 543.60, 415.441],
    'SpeedyLoader': [532, 390, 259, 182]
}

# training_data_v100 = {
#     'PyTorch DataLoader': [3909.5, 1990.57, 1505.54, 1390.47],
#     'DALI': [1834.13, 1168.760, 937.09, 705.441],
    
#     'SpeedyLoader': [1585.05, 785.29, 572.57, 265.18]
# }

training_data_v100 = {
    'PyTorch DataLoader': [3909.5, 1990.57, 1305.54, 1010.47],
    'DALI': [1834.13, 1168.760, 937.09, 705.441],
    
    'SpeedyLoader': [1585.05, 785.29, 572.57, 265.18]
}
highlight_label = 'SpeedyLoader'

# Plotting setup
hatch_patterns = ['//', 'xx', '']  # No hatch for SpeedyLoader
highlight_label = 'SpeedyLoader'
colors = ['#1f77b4',  '#ff7f0e', '#2ca02c']
labels = ['PyTorch', 'DALI', 'SpeedyLoader']

bar_width = 0.18
fontsize = 48
tick_pad = 10
yticks_a100 = [0, 1000, 2000, 3000, 4000 ]
yticks_v100 = [0, 1000, 2000, 3000, 4000 ]
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
fig_v100.savefig("3d_unet_training_time_v100_only.pdf", bbox_inches='tight')

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
            linewidth=1.5
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

fig_a100.tight_layout()
fig_a100.savefig("3d_unet_training_time_a100_only.pdf", bbox_inches='tight')
