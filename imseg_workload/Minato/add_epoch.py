import pandas as pd

# Load your CSV
df = pd.read_csv("/home/2023/rnouaj/Baselines_Speedy/mlcomns_imseg_with_dali/pytorch_v100_b4_4GPUs_dali_50epochs.csv")

# Initialize epoch tracking
epochs = []
current_epoch = 1
prev_iteration = None

for i, iter_val in enumerate(df['iteration']):
    if prev_iteration is not None and iter_val < prev_iteration:
        current_epoch += 1
    epochs.append(current_epoch)
    prev_iteration = iter_val

df['epoch'] = epochs
# Save the modified DataFrame to a new CSV file
df.to_csv("/home/2023/rnouaj/Baselines_Speedy/mlcomns_imseg_with_dali/pytorch_v100_b4_4GPUs_dali_50epochs_withepochs.csv", index=False)
