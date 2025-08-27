import pandas as pd
import matplotlib.pyplot as plt

# --- Load CSV ---
# --- Load CSV ---
csv_path = "/dl-bench/rnouaj/data-preprocessing-loader/3dunet pytorch_speedy - disk accesses dali after constraint.csv"  # Replace with your actual CSV path
df = pd.read_csv(csv_path)
print("columns", df.columns)


# Strip any trailing 's' in Time(s)
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '').astype(float)
df=df[df['Time(s)'] < 514]  # Filter out rows with non-positive time values
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


# plt.plot(df["Time(s)"], df["CPU(%)"], marker='o', markersize=15)
# plt.plot(df["Time(s)"], df["GPU(%)"], ma  rker='o', markersize=15)

# Compute averages
avg_cpu = df["CPU(%)"].mean()
avg_gpu = df["GPU(%)"].mean()
plt.plot(df["Time(s)"], df["CPU(%)"],  marker='o', markersize=15,
          )
    
plt.plot(df["Time(s)"], df["GPU(%)"],  marker='o', markersize=15,
 )

# Add dashed lines for averages
# Add dashed horizontal lines for averages (no legend here)

plt.axhline(y=avg_gpu, color='orange', label=f"Avg GPU: {81}%",linestyle='--', linewidth=8)

plt.axhline(y=avg_cpu, color='blue', label=f"Avg CPU: {9}%",linestyle='--', linewidth=8)



plt.tick_params(axis='x', pad=15)  # X-axis tick label padding
plt.tick_params(axis='y', pad=15)  # Y-axis tick label padding
plt.tick_params(axis='both',           # 'x', 'y', or 'both'
                which='major',         # 'major', 'minor', or 'both'
                length=10,             # Tick length in points
                width=3,               # Tick line width
                direction='inout')     # Tick direction: 'in', 'out', or 'inout'
plt.xticks([0, 200, 400])

plt.yticks([0, 50, 100])
# # # Add bold text labels directly on the lines
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
#     y=avg_gpu -14,
#     s=f"Avg GPU: {avg_gpu:.1f}%",
#     fontsize=130,
#     # weight='bold',
#     color='black'
# )


# plt.xlabel("Time (s)")
plt.ylabel("Usage (%)")
plt.grid(True, axis='y')
# move the legend down a bit to avoid overlap with the plot
plt.legend(loc ='center', framealpha=0.5)

plt.tight_layout()
plt.savefig("cpu_gpu_dali_memconst.svg", bbox_inches='tight')


# Plot 3: Disk Read/Write (GB/s)
plt.figure()
plt.plot(df["Time(s)"], df["DRd(GB/s)"], color='green')
# plt.xlabel("Time (s)")
# plt.ylabel("Disk Read (GB/s)")
# plt.title("Disk IO Throughput Over Time")
avg_disk = df["DRd(GB/s)"].mean()
plt.axhline(y=avg_disk, color='red', label=f"Avg: {1.4}GB/s", linestyle='--', linewidth=8)
# plt.text(
#     x=df["Time(s)"].iloc[-1] * 0.2,
#     y=1.1 ,
#     fontsize=140,
#     s=f"AVG: {avg_disk:.1f}GB/s",
  
#     color='black'
# )

plt.tick_params(axis='x', pad=15)  # X-axis tick label padding
plt.tick_params(axis='y', pad=15)  # Y-axis tick label padding
plt.tick_params(axis='both',           # 'x', 'y', or 'both'
                which='major',         # 'major', 'minor', or 'both'
                length=10,             # Tick length in points
                width=3,               # Tick line width
                direction='inout')     # Tick direction: 'in', 'out', or 'inout'

plt.yticks([0,1,2])


plt.grid(True, axis='y')
plt.legend(loc = 'center', framealpha=0.5)
plt.tight_layout()
plt.savefig("disk_io_dali_memconst.svg", bbox_inches='tight')