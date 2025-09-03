import os
import shutil
from glob import glob

src_dir = '/raid/data/imseg/raw-data/kits19/preproc-data'  # your 30GB dataset
import os
import numpy as np
import shutil

# ======== Configuration ========
     # original 30GB dataset
dst_dir = '/raid/data/imseg/raw-data/kits19/200gb_data'         # destination for ~200GB dataset
target_size_gb = 300                               # how big you want it
original_size_gb = 30                              # size of the original dataset
# ===============================

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# List all _x.npy files to use as base references
x_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_x.npy')])
num_original = len(x_files)
replication_factor = int(target_size_gb // original_size_gb)

print(f"Original files: {num_original}")
print(f"Target dataset size: {target_size_gb}GB")
print(f"Replication factor: {replication_factor}x")

new_id = 0

for i in range(replication_factor):
    for x_file in x_files:
        x_path = os.path.join(src_dir, x_file)
        y_file = x_file.replace('_x.npy', '_y.npy')
        y_path = os.path.join(src_dir, y_file)

        # Load arrays to avoid copying the same reference (optional — safer for large arrays)
        x_data = np.load(x_path)
        y_data = np.load(y_path)

        # Save with new unique IDs
        new_x_name = f"case_{new_id:05d}_x.npy"
        new_y_name = f"case_{new_id:05d}_y.npy"
        np.save(os.path.join(dst_dir, new_x_name), x_data)
        np.save(os.path.join(dst_dir, new_y_name), y_data)

        new_id += 1

print(f"\n Dataset expansion complete! Total new files: {new_id*2}")
print(f"Saved to: {dst_dir}")
