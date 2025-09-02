import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# Create results folder if not existing
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Define the three CSV files with labels
files = {
    "PyTorch": "PyTorch/cpu_gpu_usage_pytorch.csv",
    "DALI": "DALI/gpu_cpu_usage_dali.csv",
    "Minato": "Minato/cpu_gpu_usage_minato.csv"
}

for system, csv_file in files.items():
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
    out_file = os.path.join(results_dir, f"cpu_gpu_usage_{system.lower()}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved plot for {system} → {out_file}")

