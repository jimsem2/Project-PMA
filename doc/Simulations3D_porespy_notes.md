# Notes on `Simulations3D_porespy.py`

Companion explanation file — does not modify the source, so all line numbers
below stay valid as long as `Simulations3D_porespy.py` itself is unchanged.

*(Updated July 2026: the module now lives in the `logic/` package. Two
matplotlib `rcParams` lines were added right after `set_mpl_style()` so every
chart the module produces shows minor tick marks — line numbers below shift
by a few lines accordingly.)*

> **Performance trap — `steps=None` in `DRAINAGE` / `IMBIBITION`:** porespy
> 3.0.4 interprets `steps=None` as `Ps = np.unique(pc[im])` — one full
> invasion pass per **unique** capillary-pressure value. With gravity enabled
> (the defaults are `g=9.81`, `rho_nwp=1000`), `capillary_transform` adds a
> height-dependent term that makes almost every pore voxel's pc unique, so a
> run does thousands of passes instead of ~50 and takes hours. Both GUIs now
> pre-fill `steps=50`; leave it blank only deliberately (and prefer `g=0` if
> you truly need every step).

This module wraps PoreSpy's **percolation-based** two-phase-flow simulations
(drainage, imbibition, injection) and the tortuosity calculation. These are
voxel-image algorithms (no pore network is extracted) — contrast with
`Simulations3D_openPNM.py`, which solves equivalent physics on an extracted
pore network instead.

## L1-9 — Setup
- L8 `ps.visualization.set_mpl_style()`: applies PoreSpy's matplotlib theme
  globally so plots from this module look consistent with PoreSpy's own
  documentation examples.
- L9 `np.random.seed(10)`: fixes the RNG so that any randomized step (not
  used directly here, but inherited by called PoreSpy functions) is
  reproducible run-to-run — useful when comparing parameter sweeps.

## L12-34 — `save_results_vtk`
Exports four fields (binary geometry, invasion sequence, non-wetting-phase
saturation, capillary pressure) into a single `.vti` file for ParaView.
- L18-20 (`_vtk`): PoreSpy arrays are `(nx, ny, nz)`, but `pyevtk`'s
  `imageToVTK` expects C-order `(nz, ny, nx)`, so every field is transposed
  `(2, 0, 1)` and made contiguous before export. Forgetting this transpose is
  a common source of "my VTK file looks rotated/sheared in ParaView" bugs.

## L37-79 — `plot_results`
Builds the 3-panel mid-slice figure (sequence map / saturation map /
capillary-pressure map) shown after every simulation.
- L39-40 (`_pore`): masks out solid voxels with `NaN` so matplotlib's
  colour scale isn't skewed by meaningless solid-phase values.
- L42-45 (`_log_pc`): capillary pressure is plotted on a log scale (only
  where `pc > 0` and the voxel is pore) because Pc values typically span
  several orders of magnitude across pore sizes.
- L54-59: picks which axis to slice through for the 2D preview, defaulting
  to the y-mid-slice (`axis is None or axis == 0`) — this mapping (axis 0 →
  slice index 1, axis 1 → index 2, else → index 0) is a quirk of how the
  function indexes `im.shape`; check it against the actual `axis` convention
  used elsewhere if the preview orientation looks wrong.

## L82-133 — `DRAINAGE`
Simulates non-wetting phase invading the pore space (e.g. mercury intrusion,
or oil displacing water).
- L95: `ps.filters.capillary_transform` converts the binary image into a
  capillary-pressure map using the Young-Laplace relation, parameterized by
  surface tension (`sigma`), contact angle (`theta`), gravity (`g`), and the
  density difference between phases (`rho_nwp`, `rho_wp`).
- L96-112: branches on `axis` — `None` invades from **all six faces**
  simultaneously (`ps.generators.borders(mode="faces")`), while `0`/`1`/`2`
  restrict invasion to a single axis with explicit inlet/outlet faces (needed
  for directional/1D-style drainage curves).
- L98: `ps.simulations.drainage(...)` is PoreSpy's percolation-based
  algorithm that determines, for each voxel, the capillary pressure at which
  it would be invaded, subject to connectivity from the inlets.
- L115-116: every run saves a VTK file and pops up the 3-panel result plot
  unconditionally — this is what `_ALWAYS_SAVES` in `app_gui.py` accounts
  for.
- L119: `ps.metrics.pc_map_to_pc_curve` reduces the full Pc map into the
  classic capillary-pressure-vs-saturation curve plotted at L121-127.
- L129-131: the curve is only written to Excel if `save_excel=True` — this is
  the one truly **optional** save in the whole `_ALWAYS_SAVES`/conditional
  split that `app_gui.py`'s `get_save_file()` has to special-case.

## L136-185 — `IMBIBITION`
Structurally identical to `DRAINAGE` (same parameters, same axis branching at
L150-166) but calls `ps.simulations.imbibition`, which models the **wetting**
phase re-invading (displacing the non-wetting phase) — physically the reverse
process, used to generate the imbibition leg of a hysteresis loop when
combined with a prior drainage simulation.

## L188-250 — `INJECTION`
A step-by-step (rather than purely thermodynamic/percolation) invasion
simulation, with extra parameters not present in `DRAINAGE`/`IMBIBITION`:
- `conn` (L199): voxel connectivity used when tracking trapped clusters —
  `'max'` counts corner-adjacency as connected, `'min'` only counts shared
  faces.
- `method` (L201): `'qbip'` (binary-heap-based, faster) vs `'ibip'`
  (image-based) — two different algorithms PoreSpy provides for the same
  invasion-percolation process; `qbip` is the default for performance.
- L226-244: unlike drainage/imbibition, injection can legitimately invade
  **zero** pores (e.g. if `maxiter` is too low or the geometry blocks
  invasion from the chosen inlets), so the saturation curve step is wrapped
  in checks (`np.any(inj.im_snwp > 0)`) and a `try/except` around
  `pc_map_to_pc_curve`, which can otherwise raise on edge-case sequence data.
  This defensive handling is the main structural difference from
  `DRAINAGE`/`IMBIBITION`.

## L253-268 — `TORTUOSITY`
Wrapper around `ps.simulations.tortuosity_fd`, which solves a Fickian
diffusion problem on the binary image directly (finite-difference, no pore
network) along the requested `axis` and returns the tortuosity factor — a
measure of how much longer the effective diffusive path is compared to a
straight line through the material.
- **Memory constraint**: `tortuosity_fd` internally builds a
  `CubicTemplate` OpenPNM network with one node per voxel. A 1000³ image
  requires ~22 GB just for that array, making it impractical on standard
  hardware. The optional `crop_size` parameter (L257-263) crops the centre
  cube of the image to the given side length before solving — values ≤ 300
  keep memory usage manageable while still sampling a representative
  sub-volume of the pore structure.
