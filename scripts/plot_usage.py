import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Repo root = one level above this script (if it's inside scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Results and figures directories
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define the three CSV files with labels (inside results/{system}/)
files = {
    "PyTorch": os.path.join(RESULTS_DIR, "pytorch", "gpu_cpu_usage.csv"),
    "DALI": os.path.join(RESULTS_DIR, "dali", "gpu_cpu_usage.csv"),
    "Minato": os.path.join(RESULTS_DIR, "minato", "gpu_cpu_usage.csv"),
}

for system, csv_file in files.items():
    if not os.path.exists(csv_file):
        print(f"Skipping {system}, file not found: {csv_file}")
        continue

    # Load CSV
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # fix header spaces

    # Ensure numeric parsing
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df["avg_cpu"] = pd.to_numeric(df["avg_cpu"], errors="coerce")
    df["gpu_avg"] = pd.to_numeric(df["gpu_avg"], errors="coerce")

    # Plot CPU vs GPU usage
    plt.figure(figsize=(10,6))
    plt.plot(df["elapsed_s"], df["avg_cpu"], label="CPU Usage (%)", color="blue")
    plt.plot(df["elapsed_s"], df["gpu_avg"], label="GPU Usage (%)", color="green")

    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Usage (%)")
    plt.title(f"CPU vs GPU Usage Over Time ({system})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, f"cpu_gpu_usage_{system.lower()}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved plot for {system} → {out_file}")
