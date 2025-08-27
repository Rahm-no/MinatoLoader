import matplotlib.pyplot as plt
import numpy as np

# Data
gpu_counts_a100 = [1, 2, 3, 4]
gpu_counts_v100 = [2, 4, 6, 8]
hatch_patterns = ['//', 'xx', 'oo', '###']  # Match labels order
highlight_label = 'SpeedyLoader'
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
labels = ['PyTorch', 'Pecan', 'DALI', 'SpeedyLoader']
training_data_v100 = {
    'PyTorch DataLoader': [2690.6, 2390.3, 1950.7, 1710],
    'Pecan': [2688.58, 2388.9, 1949.7, 1706],
    'DALI': [1894.59, 1504.7, 1290.9, 995.8],
    'SpeedyLoader': [1471.69, 1201, 850, 640.5],
}

# training_data_a100 = {
#     'PyTorch DataLoader': [2063.67, 1850.166, 1650.7, 1390.1],
#     'Pecan': [2058.98, 1859.10, 1645.320 ,1406.72],
#     'DALI': [1410.59, 1114.7, 950.9, 709.78],
#     'SpeedyLoader': [1056.940, 884.44, 620.17, 406.294],
# }
## rhese are correct values
training_data_a100 = {
    'PyTorch DataLoader': [2063.67, 1850.166, 1650.7, 1390.1],
    'Pecan': [2058.98, 1859.10, 1645.320 ,1406.72],
    'DALI': [1410.59, 1114.7, 950.9, 709.78],
    'SpeedyLoader': [1056.940, 854.44, 620.17, 406.294],
}
# Grayscale shades from light to dark


bar_width = 0.18
fontsize = 48
yticks_v100 = [0, 500, 1000, 1500, 2000, 2500, 3000,3500]
yticks_a100 = [0, 500, 1000, 1500, 2000, 2500]
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

ax_v100.set_xticks(x_v100 + 1.5 * bar_width)
ax_v100.set_xticklabels(gpu_counts_v100, fontsize=fontsize)
ax_v100.set_yticks(yticks_v100)
ax_v100.set_yticklabels(yticks_v100, fontsize=fontsize)

# Padding for x/y tick labels
ax_v100.tick_params(axis='x', pad=tick_pad)
ax_v100.tick_params(axis='y', pad=tick_pad)

ax_v100.set_xlabel("Number of GPUs", fontsize=fontsize)
# ax_v100.set_ylabel("Training time (s)", fontsize=fontsize)
ax_v100.grid(axis='y', linestyle='--', alpha=0.6)

fig_v100.tight_layout()
fig_v100.savefig("Object_training_time_v100_only.pdf", bbox_inches='tight')

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

ax_a100.set_xticks(x_a100 + 1.5 * bar_width)
ax_a100.set_xticklabels(gpu_counts_a100, fontsize=fontsize)
ax_a100.set_yticks(yticks_a100)
ax_a100.set_yticklabels(yticks_a100, fontsize=fontsize)

# Padding for x/y tick labels
ax_a100.tick_params(axis='x', pad=tick_pad)
ax_a100.tick_params(axis='y', pad=tick_pad)

ax_a100.set_xlabel("Number of GPUs", fontsize=fontsize)
# ax_a100.set_ylabel("Training time (s)", fontsize=fontsize)
ax_a100.grid(axis='y', linestyle='--', alpha=0.6)

fig_a100.tight_layout()
fig_a100.savefig("Object_training_time_a100_only.pdf", bbox_inches='tight')
