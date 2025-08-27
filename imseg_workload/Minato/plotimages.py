import numpy as np
import matplotlib.pyplot as plt

# Load the image files
image_file = '/raid/data/unet3d/rawdata_npy/case_00159_x.npy'

# Load the images using NumPy
image_data = np.load(image_file)



print(image_data.shape)
# Extract the image array from the loaded data
image = image_data[0]

# Reshape the image array to (150, 512, 512)
image = np.squeeze(image)

# Determine the number of slices in the image
num_slices = image.shape[0]

# Determine the number of rows and columns for the subplots grid
num_rows = int(np.sqrt(num_slices))
num_cols = int(np.ceil(num_slices / num_rows))

# Create a grid of subplots to visualize the image slices
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Flatten the axes array if it's 1-dimensional
if num_slices == 1:
    axes = np.array([axes])

# Iterate over the slices and plot them
for i, ax in enumerate(axes.flat):
    if i < num_slices:
        ax.imshow(image[i], cmap='gray')
        ax.axis('off')

# Hide any unused subplots
for i in range(num_slices, num_rows * num_cols):
    axes.flat[i].axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
plt.savefig('case00159afterofflinelow.png', bbox_inches='tight', dpi=300)

plt.show()
