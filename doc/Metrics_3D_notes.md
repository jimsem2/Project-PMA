# Notes on `Metrics_3D.py`

Companion explanation file for the 3D porous-media metrics implemented in the `logic/` package.

This module gathers the quantitative analysis functions used by the GUI. It includes descriptors for
pore geometry, chord distributions, connectivity, lineal paths, and transport-style metrics.

## Important functions

- `BOND_NUMBER`: computes a bond-number-like size metric from the image.
- `CHORD_COUNTS` and `CHORD_COUNTS_SOLID_PHASE`: count chord lengths for the pore and solid phases.
- `CHORD_LENGTH_DISTRIBUTION` and `CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE`: summarize chord-length
  distributions with plots and optional Excel export.
- `POROSITY_PROFILE`: areal porosity of every slice along each of the three axes, plotted both in
  physical units and against fractional distance, with the mean porosity drawn as a dashed
  reference line.
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

## Solid-phase variants: separate outputs

The solid-phase chord metrics write their own files and label their own windows, so that running a
pore-phase and a solid-phase metric in the same session no longer overwrite one another:

| Function | Window title | Excel file |
| --- | --- | --- |
| `CHORD_COUNTS` | CHORD COUNTS | `chord_counts.xlsx` |
| `CHORD_COUNTS_SOLID_PHASE` | CHORD COUNTS - SOLID PHASE | `chord_counts_solid_phase.xlsx` |
| `CHORD_LENGTH_DISTRIBUTION` | CHORD LENGTH DISTRIBUTION | `chord_length_distribution.xlsx` |
| `CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE` | CHORD LENGTH DISTRIBUTION - SOLID PHASE | `chord_length_distribution_solid_phase.xlsx` |

These names are the ones the GUI expects: `PorousMediaAnalysis_UI.py` maps each metric function to
its output filename (in the `ALWAYS_SAVES` / filename table) so the file can be moved to the folder
the user chose. Renaming an output in this module therefore requires updating that table too, or
the save step will not find the file.

## Porosity profile: extent of the mean line

`POROSITY_PROFILE` draws all three axial profiles on one set of axes, so the horizontal dashed line
marking the overall porosity must span the **longest** axis of the image (`max(im.shape)`), not
axis 0. On a non-cubic volume — e.g. 40x90x60 — anchoring it to `im.shape[0]` left the line
covering less than half of the plotted range. The second, normalised plot is unaffected because it
spans `[0, 1]` in fractional distance by construction.
