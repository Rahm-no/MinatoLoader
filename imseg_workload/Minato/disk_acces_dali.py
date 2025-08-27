import pandas as pd
import matplotlib.pyplot as plt

# --- Load CSV ---
csv_path = "/dl-bench/rnouaj/data-preprocessing-loader/3dunet pytorch_speedy - disk accesses dali after constraint.csv"  # Replace with your actual CSV path
df = pd.read_csv(csv_path)
print("columns", df.columns)


# Strip any trailing 's' in Time(s)
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '').astype(float)
df=df[df['Time(s)'] < 514]  # Filter out rows with non-positive time values
# --- Start Plotting ---
plt.rcParams.update({
    'figure.figsize': (28, 18),     # Bigger figure
    'font.size': 110,                # Base font size
    'axes.titlesize': 110,           # Title font
    'axes.labelsize': 110,           # Axis labels
    'xtick.labelsize': 110,          # X tick labels
    'ytick.labelsize': 110,          # Y tick labels
    'legend.fontsize': 110,          # Legend text
    'lines.markersize': 9,         # Bigger markers
    'lines.linewidth': 10,           # Thicker lines
    'legend.loc': 'best',     # Legend location (optional here)
    'figure.titlesize': 26,         # Global figure title
})


# Plot 1: CPU and GPU Usage
plt.figure()

# Plot CPU and GPU usage over time
plt.plot(df["Time(s)"], df["CPU(%)"], marker='o', markersize=15)
plt.plot(df["Time(s)"], df["GPU(%)"],marker='o', markersize=15)

# Compute averages
avg_cpu = df["CPU(%)"].mean()
avg_gpu = df["GPU(%)"].mean()
max_x = df["Time(s)"].max()
plt.xticks([0,  200,400, 600])
# plt.legend(loc = 'center')
plt.yticks([0, 50, 100])

# Add horizontal lines for average CPU and GPU usage
plt.axhline(y=avg_cpu, color='blue', linestyle='--', linewidth=8, label=f"Avg CPU :{avg_cpu:.1f}%")
plt.axhline(y=avg_gpu, color='orange', linestyle='--', linewidth=8, label=f"Avg GPU: {avg_gpu:.1f}%")
plt.tick_params(axis='x', pad=15)  # X-axis tick label padding
plt.tick_params(axis='y', pad=15)  # Y-axis tick label padding
plt.tick_params(axis='both',           # 'x', 'y', or 'both'
                which='major',         # 'major', 'minor', or 'both'
                length=10,             # Tick length in points
                width=3,               # Tick line width
                direction='inout')     # Tick direction: 'in', 'out', or 'inout'
plt.legend(loc='center', framealpha=0.2)
# plt.text(
#     x=df["Time(s)"].iloc[-1] * 0.05,
#     y=avg_cpu +5,
#     s=f"Avg CPU: {avg_cpu:.1f}%",
#     fontsize=98,
#     # weight='bold',
#     color='black'
# )

# plt.text(
#     x=df["Time(s)"].iloc[-1] * 0.0005,
#     y=avg_gpu -14,
#     s=f"Avg GPU: {avg_gpu:.1f}%",
#     fontsize=98,
#     # weight='bold',
#     color='black'
# )
plt.xlabel("Time (s)")
plt.yticks([0, 50, 100])
plt.ylabel("Usage (%)")
plt.grid(True, axis='y')
# plt.legend(loc = 'center', ncol =1) 
plt.tight_layout()
plt.savefig("cpu_gpu_dali_memconst.svg", bbox_inches='tight')

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
plt.savefig("memory_dali_memconst.pdf", bbox_inches='tight')




# Plot 3: Disk Read/Write (GB/s)
plt.figure()
plt.plot(df["Time(s)"], df["DRd(GB/s)"], color='green')
plt.ylabel("Disk Read (GB/s)")
# plt.title("Disk IO Throughput Over Time")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("disk_io_dali_memconst.svg", bbox_inches='tight')

