import pandas as pd

# Read the CSV file, skipping the first row which contains repeated column headers
df2 = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech10s_speedy_b8.csv",
                  decimal=',', 
                  thousands='"',
                  header=1,  # Skip the first row (the header that repeats)
                  names=["iteration", "throughput(MBs)", "iteration_time", "time_diff", "iter_persec"])

# Print the first few rows to verify
print("heads", df2.head())

# Filter the DataFrame for 'iteration' <= 900
df2 = df2[df2['iteration'] <= 900]

