import numpy as np
import porespy as ps
import imageio.v2 as imageio
from matplotlib.pyplot import subplots
import os

# Generate a 3D blobs image
length = 200
im = ps.generators.blobs(shape=[length, length, length], porosity=0.5, blobiness=2, seed=0)

# Add Gaussian noise to the image
noise_level = 0.2
im_float = im.astype(np.float64)
noise = np.random.default_rng(seed=42).normal(loc=0, scale=noise_level, size=im_float.shape)
im_noisy = im_float + noise

# Clip values to valid range [0, 1]
im_noisy = np.clip(im_noisy, 0, 1)

# Convert to uint8 for saving
im_noisy_uint8 = (im_noisy * 255).astype(np.uint8)

# Print info
print(f"Original binary image: shape={im.shape}, unique values={np.unique(im)}")
print(f"Noisy image: shape={im_noisy_uint8.shape}, range=[{im_noisy_uint8.min()}, {im_noisy_uint8.max()}]")

# Visualize middle slice
fig, axes = subplots(1, 2, figsize=[12, 5])
axes[0].imshow(im[length // 2, :, :], cmap="gray")
axes[0].set_title("Original binary (middle slice)")
axes[0].axis("off")
axes[1].imshow(im_noisy_uint8[length // 2, :, :], cmap="gray")
axes[1].set_title("Noisy image (middle slice)")
axes[1].axis("off")
plt_path = os.path.join(os.path.dirname(__file__), "test_image", "noisy_comparison.png")
os.makedirs(os.path.dirname(plt_path), exist_ok=True)
fig.savefig(plt_path, dpi=150)
print(f"Comparison figure saved to: {plt_path}")

# Save the noisy 3D image as .tif
output_path = os.path.join(os.path.dirname(__file__), "test_image", "test_image.tif")
imageio.volsave(output_path, im_noisy_uint8)
print(f"Noisy 3D image saved to: {output_path}")
