import pandas as pd
import matplotlib.pyplot as plt

# --- Load CSV ---
csv_path = "/dl-bench/rnouaj/data-preprocessing-loader/3dunet pytorch_speedy - disk accesses pytorch after constraint.csv"  # Replace with your actual CSV path
df = pd.read_csv(csv_path)
print("columns", df.columns)


# Strip any trailing 's' in Time(s)
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '').astype(float)
df=df[df['Time(s)'] < 650]  # Filter out rows with non-positive time values
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

# Plot 3: Disk Read/Write (GB/s)
plt.figure()
plt.plot(df["Time(s)"], df["Read(GB/s)"], color='green')
# plt.plot(df["Time(s)"], df["Write(GB/s)"], color='red')
# plt.xlabel("Time (s)")
# plt.ylabel("Disk Read (GB/s)")
avg_disk = df["Read(GB/s)"].mean()
plt.tick_params(axis='x', pad=15)  # X-axis tick label padding
plt.tick_params(axis='y', pad=15)  # Y-axis tick label padding
plt.tick_params(axis='both',           # 'x', 'y', or 'both'
                which='major',         # 'major', 'minor', or 'both'
                length=10,             # Tick length in points
                width=3,               # Tick line width
                direction='inout')     # Tick direction: 'in', 'out', or 'inout'
plt.axhline(y=avg_disk, color='red', label=f"Avg: {1.1}GB/s", linestyle='--', linewidth=8)
plt.legend(loc = 'lower center', framealpha=0.7)


# plt.title("Disk IO Throughput Over Time")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("disk_io_pytorch_memconst.svg", bbox_inches='tight')


# Plot 1: CPU and GPU Usage
plt.figure()


avg_cpu = df["CPU(%)"].mean()
avg_gpu = df["GPU(%)"].mean()

# Plot CPU and GPU usage lines



plt.plot(df["Time(s)"], df["CPU(%)"], marker='o', markersize=15,
)
plt.plot(df["Time(s)"], df["GPU(%)"],  marker='o', markersize=15,
)
# Add dashed horizontal lines for averages (no legend here)
plt.axhline(y=avg_gpu, color='orange', label=f"Avg GPU: {57}%",linestyle='--', linewidth=8)

plt.axhline(y=avg_cpu, color='blue', label=f"Avg CPU: {30}%",linestyle='--', linewidth=8)

plt.yticks([0,50,100])
plt.xticks([0,  200,400, 600])
plt.tick_params(axis='x', pad=15)  # X-axis tick label padding
plt.tick_params(axis='y', pad=15)  # Y-axis tick label padding
plt.tick_params(axis='both',           # 'x', 'y', or 'both'
                which='major',         # 'major', 'minor', or 'both'
                length=10,             # Tick length in points
                width=3,               # Tick line width
                direction='inout')     # Tick direction: 'in', 'out', or 'inout'
# plt.legend(loc='upper right')  # Legend location
# plt.xlabel("Time (s)")
plt.ylabel("Usage (%)")
plt.grid(True, axis='y')
# plt.text(
#     x=df["Time(s)"].iloc[-1] * 0.0001,
#     y=avg_cpu +5,
#     s=f"Avg CPU: {avg_cpu:.1f}%",
#     fontsize=130,
#     # weight='bold',
#     color='black'
# )

# plt.text(
#     x=df["Time(s)"].iloc[-1] * 0.0001,
#     y=avg_gpu +5,
#     s=f"Avg GPU: {avg_gpu:.1f}%",
#     fontsize=130,
#     # weight='bold',
#     color='black'
# )
# plt.legend(loc = 'upper center', framealpha=0.3) 
# move the legend down a bit to avoid overlap with the plot

plt.legend(loc ='upper center', framealpha=0.7)

plt.tight_layout()
plt.savefig("cpu_gpu_pytorch_memconst.svg", bbox_inches='tight')

#