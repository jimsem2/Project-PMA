# Notes on `Filters_3D.py`

Companion explanation file for the porous-media filtering helpers in the `logic/` package.

This module collects the filter operators that prepare or transform binary images before
metrics or simulations are run. The functions are split into solid-altering and non-solid-altering
operations, and the GUI exposes them through the Filters step.

## Main themes

- Solid-altering filters change the binary pore/solid structure itself, for example cropping the
  volume, filling closed pores, trimming disconnected voxels, trimming floating solid, or removing
  non-percolating paths.
- Non-solid-altering filters preserve the object topology but annotate or label the image, such as
  applying 2D/3D chords or chord-based overlays.
- Many functions accept a `show_slice` flag so the user can inspect the result interactively in a
  slice view instead of only receiving a returned array.

## Notes on the most important helpers

- `CROP_VOLUME`: crops the image along x/y/z ranges and prints the new shape.
- `FILL_CLOSED_PORES`: fills enclosed pore regions using PoreSpy's filter.
- `TRIM_DISCONNECTED_VOXELS`: removes voxels that are not connected to the chosen inlet faces.
- `TRIM_FLOATING_SOLID`: removes isolated solid blobs that are not connected to the main solid phase.
- `TRIM_NONPERCOLATING_PATHS`: removes path elements that cannot connect inlet and outlet faces.
- `APPLY_CHORDS` / `APPLY_CHORDS_SOLID_PHASE`: draw chords on the pore or solid phase.
- `APPLY_CHORDS_3D`: apply 3D chords for a volumetric representation of the pore network.

## Practical guidance

Use these filters before running memory-intensive metrics or network-based simulations. In particular,
`CROP_VOLUME` is useful for large images, while `TRIM_DISCONNECTED_VOXELS` and
`TRIM_NONPERCOLATING_PATHS` help remove artifacts that can distort later percolation or transport
calculations.
