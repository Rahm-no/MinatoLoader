#read two csv files and do plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.figsize': (10, 6),     # Bigger figure
    'font.size': 64,                # Base font size
    'axes.titlesize': 64,           # Title font
    'axes.labelsize': 64,           # Axis labels
    'xtick.labelsize': 64,          # X tick labels
    'ytick.labelsize': 64,          # Y tick labels
    'legend.fontsize': 42,          # Legend text
    'lines.markersize': 9,         # Bigger markers
    'lines.linewidth': 6,           # Thicker lines
    'legend.loc': 'best',     # Legend location (optional here)
    'figure.titlesize': 26,         # Global figure title
})
# Read the CSV files
df1 = pd.read_csv("dali_a100.csv", sep='\t')
new_time = np.linspace(5, 400, num=len(df1["Time(s)"])).astype(int)
df1['Time(s)'] = new_time

print(df1.head())
print(df1.columns)
# plt.xlabel('Time (s)')
mean_cpu = df1['CPU'].mean()
mean_gpu = df1['GPU'].mean()

plt.ylabel('Usage (%)')


# Get last x value (for placement)
last_x = df1['Time(s)'].max()
# Annotate the means on the plot
plt.text(last_x - 450, mean_cpu + 2, f"Avg CPU: {mean_cpu:.1f}%", color='black', fontsize=64, weight='bold')
plt.text(last_x - 450, mean_gpu - 20, f"Avg GPU: {mean_gpu:.1f}%", color='black', fontsize=64, weight='bold')
plt.grid(True)
plt.ylabel('Usage (%)')
plt.xticks([0, 200,400])
# Assign the new time values
# Plot both datasets
plt.plot(df1['Time(s)'], df1['CPU'], label='CPU (%)')
plt.plot(df1['Time(s)'], df1['GPU'], label='GPU (%)')

plt.savefig('cpu_gpu_dali_imseg.svg', dpi=300, bbox_inches='tight')
plt.show()





