import pandas as pd

# Load the CSV
df = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/preprocessing_times.csv")

# Choose the column to clean (replace 'column_name' with the actual name)
df["column_name"] = df["column_name"].str.rstrip(")")

# Save the cleaned CSV
df.to_csv("cleaned_file.csv", index=False)
