import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Current script directory (e.g., .../PyTorch)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Repo root = one level above
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Paths
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# CSV file at root/results
csv_file = os.path.join(RESULTS_DIR, "results_allsystems.csv")

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
plt.ylabel("Training time (seconds)")
plt.title("Comparison of training time between systems")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save inside figures/ folder
out_file = os.path.join(FIGURES_DIR, "results_histogram.png")
plt.savefig(out_file, dpi=300)
print(f"Saved histogram → {out_file}")

plt.show()
