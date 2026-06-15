import imageio.v2 as imageio
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
from skimage.io import imread_collection
import os

# Path to the test image
image_dir = os.path.join(os.path.dirname(__file__), "test_image")
image_file = os.path.join(image_dir, "test_image.tif")
slices_dir = os.path.join(image_dir, "z_slices")  # folder of 2D tif slices

# Output directory
output_dir = os.path.join(image_dir, "3d_result")
os.makedirs(output_dir, exist_ok=True)


# ============ Load the 3D image ============
# If a folder of 2D slices exists, load them using imread_collection (PoreSpy convention)
# Otherwise, load a single 3D tif file using imageio.volread
if os.path.isdir(slices_dir) and len(os.listdir(slices_dir)) > 0:
    seq = imread_collection(os.path.join(slices_dir, "*.tif"))
    im3d = np.stack(seq, axis=-1)
    print(f"Image loaded from 2D slices via imread_collection: shape={im3d.shape}, dtype={im3d.dtype}")
else:
    im3d = imageio.volread(image_file)
    # volread returns (num_slices, H, W) -> transpose to (H, W, num_slices)
    im3d = np.transpose(im3d, (1, 2, 0))
    print(f"Image loaded via imageio.volread: shape={im3d.shape}, dtype={im3d.dtype}")

# ============ Inspect the image ============
print(f"Image dimensions: {im3d.shape}")
print(f"Voxel value range: {im3d.min()} to {im3d.max()}")

# Visualize a slice of the 3D image (z-slices are on the last axis)
fig, ax = plt.subplots(figsize=[6, 6])
mid = im3d.shape[2] // 2
ax.imshow(im3d[:, :, mid], cmap="gray")
ax.set_title("Middle slice (axis 2)")
ax.axis("off")
fig.savefig(os.path.join(output_dir, "slice_preview.png"), dpi=150)
print("Slice preview saved.")

# ============ Convert to binary (if needed) ============
# Check if image is already binary (only contains 2 unique values)
unique_values = np.unique(im3d)
is_binary = len(unique_values) <= 2

if is_binary:
    print(f"Image is already binary (unique values: {unique_values}). Skipping thresholding.")
    im_binary = im3d.astype(bool)
else:
    # ============ Denoise using Non-Local Means (best edge-preserving filter) ============
    print("Image is greyscale. Applying Non-Local Means denoising...")

    # Normalize to [0, 1]
    im_float = im3d.astype(np.float64)
    im_float = (im_float - im_float.min()) / (im_float.max() - im_float.min())

    # Apply PoreSpy's parallelized NL-Means filter (layer-by-layer along z = axis 2)
    def DENOISING(cores=None, patch_size=9, patch_distance=7, h=0.1):
        im_denoised = ps.filters.nl_means_layered(
            im_float,
            axis=2,
            cores=cores,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
        )
        im_denoised = (im_denoised - im_denoised.min()) / (im_denoised.max() - im_denoised.min())
        print("Denoising complete.")
        return im_denoised
    
    # Inspect histogram to choose threshold
    im_denoised = DENOISING()
    fig2, ax2 = plt.subplots(figsize=[6, 4])
    ax2.hist(im_denoised.flatten(), bins=50, edgecolor="k")
    ax2.set_xlabel("Voxel intensity")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram of denoised image")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "histogram.png"), dpi=150)
    print("Histogram saved to histogram.png.")
    print("Close the histogram window to continue.")
    plt.show()

    threshold = None
    while threshold is None:
        user_input = input("Enter threshold value for segmentation [0.0 - 1.0] (default 0.5): ").strip()
        if user_input == "":
            threshold = 0.5
        else:
            try:
                threshold = float(user_input)
                if threshold < 0 or threshold > 1:
                    print("Threshold must be between 0.0 and 1.0. Please try again.")
                    threshold = None
            except ValueError:
                print("Invalid input. Enter a numeric threshold between 0.0 and 1.0.")
                threshold = None

    im_binary = im_denoised < threshold  # True = pore (void), False = solid
    print(f"Threshold selected: {threshold}")

    # Visualize binary result
    fig3, ax3 = plt.subplots(1, 3, figsize=[15, 5])
    mid = im3d.shape[2] // 2
    ax3[0].imshow(im3d[:, :, mid], cmap="gray")
    ax3[0].set_title("Original (noisy)")
    ax3[0].axis("off")
    ax3[1].imshow(im_denoised[:, :, mid], cmap="gray")
    ax3[1].set_title("Denoised (NL-Means)")
    ax3[1].axis("off")
    ax3[2].imshow((~im_binary[:, :, mid]), cmap="gray")
    ax3[2].set_title("Binary (segmented)")
    ax3[2].axis("off")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "binary_preview.png"), dpi=150)
    print("Binary preview saved.")

# ============ Save the 3D binary image ============
# Save as .tif for further processing (as recommended in PoreSpy docs)
# Transpose back to (num_slices, H, W) for volsave (tif convention)
# Using int8 as shown in PoreSpy docs
output_path = os.path.join(output_dir, "test_image_binary_3D.tif")
im_save = np.transpose(np.array(~im_binary, dtype=np.int8), (2, 0, 1))
imageio.volsave(output_path, im_save)
print(f"3D binary image saved to: {output_path}")
