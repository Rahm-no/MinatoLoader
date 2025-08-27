import pandas as pd

df = pd.read_csv("/dl-bench/rnouaj/data-preprocessing-loader/accuracy_500epochs.csv")  # Replace with the actual file name

# Compute the average for each iteration (group by 'epoch')
df_avg = df.groupby('epoch', as_index=False).agg({
    'mean dice': 'mean',
    'accuracy': 'mean',
    'l1 dice': 'mean'
})

# Rename the columns to match the desired output
df_avg.rename(columns={
    'mean_dice': 'accuracy',
    'accuracy': 'l1_dice',
    'L2 dice': 'l2_dice',
}, inplace=True)

# Save the result to a new CSV file
df_avg.to_csv("average_accuracy_speedy4GPUs_per_epoch.csv", index=False)

# Optionally, print the average DataFrame to the console to check
print(df_avg)
