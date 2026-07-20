# Notes on `Metrics_3D.py`

Companion explanation file for the 3D porous-media metrics implemented in the `logic/` package.

This module gathers the quantitative analysis functions used by the GUI. It includes descriptors for
pore geometry, chord distributions, connectivity, lineal paths, and transport-style metrics.

## Important functions

- `BOND_NUMBER`: computes a bond-number-like size metric from the image.
- `CHORD_COUNTS` and `CHORD_COUNTS_SOLID_PHASE`: count chord lengths for the pore and solid phases.
- `CHORD_LENGTH_DISTRIBUTION` and `CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE`: summarize chord-length
  distributions with plots and optional Excel export.
- `COORDINATION_NUMBER`: builds an OpenPNM network from a PoreSpy SNOW2 extraction and reports the
  per-pore coordination number frequency together with the mean coordination number. The result is
  displayed as a histogram and can be exported to Excel.
- `FIND_H`: estimates the intrusion height or saturation-related characteristic length.
- `TWO_POINT_CORRELATION`: evaluates the two-point correlation of the binary image.

## Notes

- The metric functions are designed to be called from the GUI, but they can also be used in scripts.
- Several functions support `save_excel=True` to write tabular results to `.xlsx` files.
- The coordination-number metric uses OpenPNM and PoreSpy together so it works from the same image
  representation as the rest of the network-based analysis workflow.
