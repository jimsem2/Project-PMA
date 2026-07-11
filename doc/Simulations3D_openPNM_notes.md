# Notes on `Simulations3D_openPNM.py`

Companion explanation file — does not modify the source, so all line numbers
below stay valid as long as `Simulations3D_openPNM.py` itself is unchanged.

*(Updated: removed unused imports `from tokenize import Name`, `import imageio`,
`import os`; removed `np.random.seed(10)` and the `__main__` standalone block —
none of these are needed when the module is used only through `app_gui.py`.
Later additions: a `save_vtk` flag on every `simulate_*` function plus a new
`export_vtk()` helper for one-file project export; a scalar
`throat.ad_dif_conductance_scalar` in advection-diffusion so the otherwise
non-exportable directional conductance can colour the throat tubes.)*

*(Updated July 2026: the module now lives in the `logic/` package. Two
matplotlib `rcParams` lines after the imports enable minor tick marks on all
charts, and `ax.grid(True)` was removed from both saturation-curve plots
(drainage and invasion percolation) — the curves are now grid-free.)*

> **Line-number caveat:** the `save_vtk`/`export_vtk` and `ad_dif` scalar
> additions shifted lines below their insertion points; search by name
> (`export_vtk`, `save_vtk`, `ad_dif_conductance_scalar`) rather than trusting
> exact `Lxxx` references.

This module extracts a **pore network** (pores + throats) from a binary image
via PoreSpy's SNOW2 algorithm, then solves transport physics (Stokes flow,
diffusion, advection-diffusion) and two-phase invasion (drainage, invasion
percolation) on that network using OpenPNM. Contrast with
`Simulations3D_porespy.py`, which works directly on the voxel image instead
of an extracted network.

## L1-11 — Imports & module-level state
- L1-5: only the libraries actually used by the module are imported —
  `matplotlib.pyplot` for the saturation-curve plots, `numpy` for array
  arithmetic, `porespy` for the SNOW2 network extraction, `pandas` for the
  optional Excel exports, and `openpnm` for all algorithms and geometry
  models.
- L10-11 (`im`, `pn`): deliberately left `None` at import time rather than
  built eagerly. This is what lets `app_gui.py` import this module cheaply —
  no network extraction happens until `build_network()` is actually called
  (see `app_gui_notes.md`, imports section).

## L12-45 — `build_network`
Converts a binary PoreSpy image into a ready-to-simulate OpenPNM network.
- **Memory constraint**: `ps.networks.snow2` (L21) performs a full
  watershed segmentation of the pore space — for a 1000³ image this requires
  tens of GB of RAM and will crash on typical workstations. All OpenPNM
  simulations pass through this function, so the whole `open_openpnm` window
  is effectively blocked on images larger than ~300–400³. Use the
  `CROP_VOLUME` solid-altering filter (in the Filters step of the GUI) before
  opening the OpenPNM window.
- L21: `ps.networks.snow2` is PoreSpy's SNOW (Sub-Network of an
  Oversegmented Watershed) algorithm — it watershed-segments the pore space
  into discrete pore regions and identifies the throats connecting them,
  given a physical `voxel_size`.
- L22: `op.io.network_from_porespy` converts SNOW2's output into an OpenPNM
  `Network` object.
- L25-28: networks extracted from real (especially imported/segmented) images
  can contain pores with no connection to the rest of the network; these are
  trimmed because the transport algorithms below would otherwise produce
  singular/ill-conditioned systems.
- L30-44: PoreSpy and OpenPNM use different property names for the same
  geometric quantities, so L34-36 alias them (`pore.diameter`,
  `throat.diameter`, `throat.length`). L37-43 then attach the
  `pyramids_and_cuboids` size-factor models (chosen specifically because it
  doesn't require throat diameter to be smaller than pore diameter, which can
  fail for some extracted geometries) and a cylindrical throat-volume model,
  both prerequisites for the conductance models used later in this file.

## `export_vtk` and the `save_vtk` flag — one-file project export
Every simulation in this module runs on the **same** OpenPNM project, and
`op.io.project_to_vtk` exports the *whole* project (network + every phase). So
rather than each `simulate_*` function writing its own near-identical `.vtp`:
- Each function takes a `save_vtk=True` argument. When left `True` (standalone
  use), the function still writes its own file as before. The GUI passes
  `save_vtk=False` so no per-function file is written.
- `export_vtk(network, filename)` does the single `project_to_vtk` call. The
  GUI invokes it **once** after all ticked simulations finish, producing one
  combined file whose name lists the selected simulations (e.g.
  `openpnm_SF_FD_AD_Dr_IP.vtp`). `filename` is a stem — OpenPNM appends `.vtp`.
- Consequence (by design): that one file contains every result field from every
  ticked simulation, so in ParaView you can switch colouring between pressure,
  concentration, invasion pressures, etc. without reloading. See the OpenPNM
  section of `app_gui_notes.md` for the GUI side.

## L47-62 — Axis/face mapping
`AXIS_MAP` and `get_axis_faces()` translate the user-facing axis string
(`'x'`/`'y'`/`'z'`) into the boundary-pore labels SNOW2 assigns
(`xmin`/`xmax`/etc.) plus the corresponding coordinate column index — every
simulation function below calls this once to find its inlet/outlet pores.

## L66-141 — `select_phase`
Builds an OpenPNM `Phase` object (`water`, `air`, or `mercury`) with physical
properties either left at sensible literature defaults or overridden via
function arguments (this is what the GUI's per-phase forms in
`app_gui.py`/`PNM_PHASE_SPECS` feed into).
- L78-89 / L90-101 / L102-110: each phase branch fills in the properties
  OpenPNM's built-in `Water`/`Air`/`Mercury` phases don't define by default
  (e.g. diffusivity, contact angle, surface tension), with literature
  defaults documented inline as comments (e.g. water self-diffusion at 25 °C,
  mercury intrusion contact angle ~140°).
- L114-139: regardless of which phase was selected, pore-scale properties
  are interpolated to throats (`from_neighbor_pores`, using the **minimum**
  of the two adjacent pores — a conservative choice for transport-limiting
  properties), and the four conductance/entry-pressure models used by every
  simulation function in this file are attached: Hagen-Poiseuille
  (hydraulic), ordinary diffusion (diffusive), and Washburn (capillary entry
  pressure for two-phase invasion).

## L144-229 — `simulate_stokes_flow`
Solves single-phase pressure-driven (Darcy-scale) flow.
- L155-162: `StokesFlow` algorithm with a constant pressure boundary
  condition on the inlet and outlet faces.
- L168-174: total flow rate is read at the outlet; OpenPNM's sign convention
  makes outlet rates negative (fluid leaving), so the magnitude `Q` is what's
  reported as the physical flow rate.
- L176-192: per-throat velocity is computed as `Q/A`, then **interpolated
  to pores** by averaging the velocities of each pore's adjacent throats
  (L182-191) purely so ParaView can colour pore-glyph spheres by velocity —
  OpenPNM only natively associates velocity with throats.
- Darcy permeability `K = Q·μ·L / (A·ΔP)` is now computed **three ways** and
  all three are printed. The Stokes problem is solved once — `Q` and `ΔP` are
  identical for all three — so the only thing that changes `K` is how the
  domain cross-section `A` and length `L` are measured on the irregular,
  face-less extracted network:
  1. **image-based (recommended, primary):** the true sample cuboid,
     `A = (N_y·vs)(N_z·vs)`, `L = N_x·vs`, from `image_domain_dimensions()`.
     This is the physical definition a lab permeameter uses.
  2. **coordinate-based:** bounding box of pore *centroids* — robust, but
     systematically off by ~one pore radius because centroids never reach the
     true faces.
  3. **geometric:** OpenPNM `get_domain_area`/`get_domain_length` (convex hull
     of the face pores). Physically motivated but fragile — it under-covers the
     face (→ K too high) and can fail, hence the `try/except`. **Sanity check
     only.**
  The returned/primary `K` is the image-based value (coordinate-based if the
  image size is unavailable). The three can differ by tens of percent, more for
  thin/coarse networks; this is a normalisation ambiguity, not a physics one.
- **Updated July 2026** — previously only the geometric and coordinate methods
  were computed and the *geometric* one was preferred; the image-based value
  (via the new module global `_VOXEL_SIZE`, set in `build_network`) was added
  as the physical primary.
- results are exported to `openpnm_stokes_flow.vtp`.

## L232-307 — `simulate_fickian_diffusion`
Solves steady-state diffusion (no advection) — structurally a mirror of
`simulate_stokes_flow` but with concentration boundary conditions instead of
pressure (L248-249) and `FickianDiffusion` instead of `StokesFlow`.
- effective diffusivity `D_eff = J·L / (A·ΔC)` is computed with the **same
  three-method** (image-based / coordinate / geometric) scheme as the
  permeability above, and the **image-based** value is the primary/returned one.
- **Two distinct tortuosities.** This module and `Simulations3D_porespy.py`
  each compute a tortuosity, and they are *not* the same quantity:
  - **Network diffusive tortuosity factor** (here): `tau = porosity · D_bulk /
    D_eff`, derived from the network `D_eff` above. It measures path
    lengthening/constriction on the *reduced SNOW2 pore network*. Because it is
    inversely proportional to `D_eff`, it inherits the A/L normalisation
    ambiguity, so it is printed for the image-based (physical) normalisation and
    — for transparency — also for the geometric one. The porosity comes from the
    module-level `im` global set by `build_network()`.
  - **Voxel (finite-difference) tortuosity** (`Simulations3D_porespy.TORTUOSITY`
    → `ps.simulations.tortuosity_fd`): solves Laplace/Fickian diffusion on
    *every voxel* of the image directly. It keeps the full geometry (no network
    idealisation) and is usually the more faithful of the two, at much higher
    memory cost.
  The two normally disagree; comparing them is a legitimate cross-check between
  the network and voxel approaches, not a contradiction. Both functions now
  print an explicit note stating which tortuosity they report and how it differs
  from the other.

## L310-383 — `simulate_advection_diffusion`
Combines flow and diffusion (a solute being both carried by and diffusing
through the fluid).
- L325-334: if Stokes flow hasn't already been solved on this phase
  (`"pore.pressure" not in phase.keys()`), it's run first to establish the
  velocity field the advection term needs — this means calling
  `simulate_stokes_flow` beforehand is optional, not required.
- L337-341: `ad_dif_conductance` combines the hydraulic and diffusive
  conductances into one throat-level conductance term used by the
  `AdvectionDiffusion` algorithm.
- **`throat.ad_dif_conductance_scalar`** (added just before the export): the
  raw `throat.ad_dif_conductance` is a *multi-column, directional* array
  (one value per connection direction, upwinded by the local flow), which
  `op.io.project_to_vtk` cannot map to a VTK scalar — so it never appears in
  ParaView. A per-throat mean over those columns is stored under this scalar
  name purely so the throat tubes can be coloured by it. It is a summary
  value; the true conductance remains direction-dependent.
- L366-376: the Péclet number `Pe = v·L/D` is reported as a dimensionless
  indicator of whether advection (`Pe > 1`) or diffusion (`Pe < 1`) dominates
  transport in this geometry/flow regime.

## L389-450 — `simulate_drainage`
Network-based (rather than voxel-based) drainage, using OpenPNM's
`Drainage` algorithm — the network-extraction counterpart to
`Simulations3D_porespy.py`'s `DRAINAGE`.
- L399-400: requires **two** phases — `invading_phase` (e.g. air pushing in)
  and `defending_phase` (e.g. water being displaced) — unlike the
  single-phase transport functions above.
- L404-409: `axis="all"` invades from every boundary face that actually has
  pores (some faces can be empty after the L25-28 disconnected-pore trim),
  using `np.unique(np.concatenate(...))` to combine them into one inlet set.
- L417: deliberately runs `drain.run()` **without** a fixed number of
  pressure steps, so it evaluates at every unique throat entry pressure —
  the comment at L414-416 explains this gives maximum curve resolution
  rather than under-sampling with a fixed step count.
- L440-443: invasion pressure data is copied onto the phase object
  specifically so it gets included in the VTK export at L445-447 (algorithm
  result objects aren't automatically exported, only phase/network data is).

## L453-509 — `simulate_invasion_percolation`
Same two-phase setup and inlet logic as `simulate_drainage` (L468-477 mirrors
L404-413), but uses OpenPNM's `InvasionPercolation` algorithm instead, which
models invasion as a discrete sequence of pore-by-pore displacement events
rather than `Drainage`'s pressure-controlled equilibrium approach — these are
two different physical/numerical models of the same drainage process, useful
to compare against each other.
