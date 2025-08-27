import pandas as pd
import matplotlib.pyplot as plt

# --- Load CSV ---
csv_path = "/dl-bench/rnouaj/data-preprocessing-loader/3dunet pytorch_speedy - disk accesses speedy after constraint.csv"  # Replace with your actual CSV path
df = pd.read_csv(csv_path)
print("columns", df.columns)


# Strip any trailing 's' in Time(s)
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '').astype(float)
df=df[df['Time(s)'] < 330]  # Filter out rows with non-positive time values
# --- Start Plotting ---
plt.rcParams.update({
    'figure.figsize': (32, 20),     # Bigger figure
    'font.size': 140,                # Base font size
    'axes.titlesize': 140,           # Title font
    'axes.labelsize': 140,           # Axis labels
    'xtick.labelsize': 140,          # X tick labels
    'ytick.labelsize': 140,          # Y tick labels
    'legend.fontsize': 140,          # Legend text
    'lines.markersize': 9,         # Bigger markers
    'lines.linewidth': 10,           # Thicker lines
    'legend.loc': 'best',     # Legend location (optional here)
    'figure.titlesize': 26,         # Global figure title
})


# Plot 1: CPU and GPU Usage
plt.figure()

# Plot CPU and GPU usage over time
# Plot CPU and GPU usage lines
# GPU in orange

# CPU in navy
plt.plot(df["Time(s)"], df["CPU(%)"],  marker='o', markersize=15,
)
    
plt.plot(df["Time(s)"], df["GPU(%)"],  marker='o', markersize=15,
)
# plt.plot(df["Time(s)"], df["CPU(%)"], marker='o', markersize=15)
# plt.plot(df["Time(s)"], df["GPU(%)"], ma  rker='o', markersize=15)

# Compute averages
avg_cpu = df["CPU(%)"].mean()
avg_gpu = df["GPU(%)"].mean()

# Add dashed lines for averages
# Add dashed horizontal lines for averages (no legend here)
plt.axhline(y=avg_gpu, color='orange', label=f"Avg GPU: {82}%",linestyle='--', linewidth=8)

plt.axhline(y=avg_cpu, color='blue', label=f"Avg CPU: {15}%",linestyle='--', linewidth=8)


plt.tick_params(axis='x', pad=15)  # X-axis tick label padding
plt.tick_params(axis='y', pad=15)  # Y-axis tick label padding
plt.tick_params(axis='both',           # 'x', 'y', or 'both'
                which='major',         # 'major', 'minor', or 'both'
                length=10,             # Tick length in points
                width=3,               # Tick line width
                direction='inout')     # Tick direction: 'in', 'out', or 'inout'

plt.xlabel("Time (s)")
plt.legend(loc='center', framealpha=0.7)
plt.xticks([0,   100,200, 300])
plt.yticks([0, 50, 100])
# # # Add bold text labels directly on the lines


# plt.xlabel("Time (s)")
plt.ylabel("Usage (%)")
plt.grid(True, axis='y')
# move the legend down a bit to avoid overlap with the plot
# plt.legend(loc ='center', framealpha=0.2)

plt.xticks([0,  100, 200, 300])
plt.tight_layout()
plt.savefig("cpu_gpu_speedy_memconst.svg", bbox_inches='tight')

# Plot 2: Memory Usage
plt.figure()

# Plot memory usage line
plt.plot(df["Time(s)"], df["Mem(GB)"], label="Memory Usage (GB)", color='purple')

# Shade **above** the line (example: up to a max value like 100 GB)
plt.fill_between(df["Time(s)"], df["Mem(GB)"], y2=80, where=(df["Mem(GB)"] < 80),
                 color='lavender', alpha=0.5, label='Page Cache(GB)')

# Shade **below** the line (down to 0 GB)
plt.fill_between(df["Time(s)"], df["Mem(GB)"], y2=0, where=(df["Mem(GB)"] > 0),
                 color='thistle', alpha=0.5)

# Labels and grid
plt.xlabel("Time (s)")
plt.ylabel("Memory (GB)")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("memory_speedy_memconst.pdf", bbox_inches='tight')


# Plot 3: Disk Read/Write (GB/s)
plt.figure()
avg_disk = df["read(GB/s)"].mean()

plt.plot(df["Time(s)"], df["read(GB/s)"], color='green')
plt.xlabel("Time (s)")
# plt.ylabel("Disk Read (GB/s)")
# plt.title("Disk IO Throughput Over Time")

plt.axhline(y=1.7, color='red', linestyle='--',label=f"Avg: {1.7}GB/s", linewidth=8)


plt.legend(loc='upper center', framealpha=0.5)

plt.grid(True, axis='y')
plt.yticks([0, 1, 2])
plt.xticks([0,  100, 200, 300])
plt.legend(loc='center', framealpha=0.8)
plt.tight_layout()
plt.savefig("disk_io_speedy_memconst.svg", bbox_inches='tight')

print("Plots saved: 'plot_cpu_gpu_usage.png', 'plot_memory_usage.png', 'plot_disk_io.png'")
