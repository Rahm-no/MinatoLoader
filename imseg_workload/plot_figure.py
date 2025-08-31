import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Replace with your CSV filename
csv_file = "results_allsystems.csv"

# Create results folder if not existing
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Read CSV and strip spaces from column names
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()   # remove hidden spaces in headers

# Try parsing both 12h and 24h formats
def parse_mixed_datetime(x):
    for fmt in ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue
    return pd.NaT  # if parsing fails

df["timestamp"] = df["timestamp"].apply(parse_mixed_datetime)

# Convert seconds column to numeric
df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")

# Keep only the most recent entry per system
latest_df = df.sort_values("timestamp").groupby("system").tail(1)

# Plot histogram
plt.figure(figsize=(8,6))
plt.bar(latest_df["system"], latest_df["seconds"], color=["orange","blue","green"])
plt.ylabel("Seconds")
plt.title("Latest Results per System")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# # Annotate values and timestamps
# for i, (val, ts) in enumerate(zip(latest_df["seconds"], latest_df["timestamp"])):
#     plt.text(i, val+2, f"{val:.2f}\n{ts.strftime('%Y-%m-%d %H:%M:%S')}", 
#              ha="center", fontsize=9)

# Save inside results/ folder
out_file = os.path.join(results_dir, "results_histogram.png")
plt.savefig(out_file, dpi=300)
print(f"Saved histogram → {out_file}")

plt.show()
