import matplotlib.pyplot as plt
import numpy as np

# Data
gpu_counts_a100 = [1, 2, 3, 4]
gpu_counts_v100 = [2, 4, 6, 8]

# training_data_v100 = {
#     'PyTorch DataLoader': [1900.6, 1550.3, 1250.7, 910],
#     'Pecan': [1890.58, 1540.9, 1240.7, 862],
#     'DALI': [1340.59, 1104.7, 880.9, 719.78],
#     'SpeedyLoader': [971.69, 610, 540, 370.5],
# }
#updated
training_data_v100 = {
    'PyTorch DataLoader': [1900.6, 1550.3, 1250.7, 910],
    'Pecan': [1890.58, 1540.9, 1240.7, 862],
    'DALI': [1340.59, 1104.7, 880.9, 719.78],
    'SpeedyLoader': [971.69, 610, 540, 370.5],
}

# training_data_a100 = {
#     'PyTorch DataLoader': [1163.67, 895.166, 750.7, 500.1],
#     'Pecan': [1158.98, 890.10, 743.320 ,497.72],
#     'DALI': [920.59, 750, 580.9, 350.78],
#     'SpeedyLoader': [580.940, 430.44, 280.17, 133.294],
# }

training_data_a100 = {
    'PyTorch DataLoader': [2403.67, 1314.166, 660.7, 500.1],
    'Pecan': [1958.98, 1090.10, 558.320 ,497.72],
    'DALI': [1558.59, 850, 550.9, 432.78],
    'SpeedyLoader': [509.940, 290.44, 180.17, 133.294],
}
hatch_patterns = ['//', 'xx', 'oo', '###']  # Match labels order
highlight_label = 'SpeedyLoader'
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
labels = ['PyTorch', 'Pecan', 'DALI', 'SpeedyLoader']
bar_width = 0.18
fontsize = 48
yticks_v100 = [0, 500, 1000, 1500, 2000]
yticks_a100 = [0, 500, 1000, 1500, 2000, 2500,]

tick_pad = 10

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
ax_v100.set_ylabel("Training time (s)", fontsize=fontsize)
ax_v100.grid(axis='y', linestyle='--', alpha=0.6)

fig_v100.tight_layout()
fig_v100.savefig("speech3s_trainin_time_v100_only.pdf", bbox_inches='tight')

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
ax_a100.set_ylabel("Training time (s)", fontsize=fontsize)
ax_a100.grid(axis='y', linestyle='--', alpha=0.6)

fig_a100.tight_layout()
fig_a100.savefig("speech3s_training_time_a100_only.pdf", bbox_inches='tight')
