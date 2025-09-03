import nibabel as nib
import matplotlib.pyplot as plt

# Load the NIfTI files
image_file = '/raid/data/imseg/raw-data/kits19/data/case_00165/imaging.nii.gz'

# Load the images using nibabel
image_data = nib.load(image_file).get_fdata()

# Create a subplot for all slices
num_slices = image_data.shape[2]  # Get the number of slices along the z-axis
num_rows = int(num_slices ** 0.5)  # Determine the number of rows for subplots
num_cols = (num_slices + num_rows - 1) // num_rows  # Determine the number of columns for subplots

# Create the figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Iterate over slices and plot on subplots
for slice_index, ax in enumerate(axes.flat):
    if slice_index < num_slices:
        ax.imshow(image_data[:, :, slice_index], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')  # Hide empty subplots

# Remove empty subplots if any
if num_slices < num_rows * num_cols:
    for ax in axes.flat[num_slices:]:
        ax.remove()

# Adjust spacing and layout
fig.tight_layout()

# Save the figure
plt.savefig('slices_case00130.png')
plt.close()
