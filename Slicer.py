import os

import imageio.v2 as imageio
import numpy as np
import porespy as ps


def main():
    # Generate a 3D blob image of size 100x100x100
    size = 100
    image = ps.generators.blobs(shape=[size, size, size], porosity=0.5, blobiness=2)

    # Create output folder inside test_image
    output_folder = os.path.join("test_image", "z_slices")
    os.makedirs(output_folder, exist_ok=True)

    # Save each z-axis slice as a single-channel TIFF image
    for i in range(image.shape[2]):
        slice_2d = image[:, :, i]
        slice_2d = np.asarray(slice_2d, dtype=np.uint8) * 255
        file_name = f"slice_z_{i:03d}.tif"
        file_path = os.path.join(output_folder, file_name)

        imageio.imwrite(file_path, slice_2d)

    print(f"Saved {image.shape[2]} z-axis slices to: {output_folder}")


if __name__ == "__main__":
    main()
