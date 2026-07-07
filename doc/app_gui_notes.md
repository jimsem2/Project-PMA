# Notes on `app_gui.py`

Companion explanation file — does not modify the source, so all line numbers
below stay valid as long as `app_gui.py` itself is unchanged.

*(Updated after: DICOM-import changes; denoising-bug workaround; inversion naming fix;
tortuosity crop_size parameter; `CROP_VOLUME` filter (replaces the old `CROP_CENTER`);
`_image_size_label` size warnings; estimated/actual run-time labels; generators excluded
from time estimates; OpenPNM single-combined-VTK export.)*

*(Updated July 2026 — project restructure and UX fixes:*
- *The analysis modules moved into the `logic/` package and this file into `ui/`.
  A small bootstrap right before the `import logic.*` block inserts the project
  root (the parent of `ui/`) into `sys.path`, so the script runs from the IDE or
  the command line without any environment setup.*
- *Filters window: only **solid-altering** filters carry the run-order combobox
  (values 1..5 — one per SA filter). **Non-solid-altering** filters are now plain
  checkboxes (`mode="multi"`); they always run after the SA chain, each receiving
  its result. The duplicate-order resolver watches SA sections only.*
- *`ResultBox._on_scroll` consumes the mouse wheel only while the box still has
  content to scroll in that direction; when empty or at its top/bottom the event
  passes through so the window keeps scrolling.*
- *`steps` in `_DRAIN_IMB_FIELDS` now defaults to `50` instead of blank. Blank →
  `None` makes porespy 3.0.4 run one invasion pass per **unique** capillary
  pressure value, and with gravity enabled (`g=9.81`, `rho_nwp=1000`) the
  height-dependent term makes almost every pore voxel unique — thousands of
  passes instead of ~50, hence runs that blow far past any estimate.*
- *`SATN_PROFILE` gained `s_min` / `s_max` / `s_step` fields (defaults 0.1 / 0.9 /
  0.2 — the original hardcoded series). The function now also skips thresholds
  above the image's maximum reached saturation instead of raising
  "The maximum saturation in the image is less than the given threshold".)*

> **Line-number caveat:** the `CROP_VOLUME` and combined-export changes added a
> handful of lines, so some `Lxxx` references below have drifted by a few lines.
> Search by name (`CROP_VOLUME`, `do_simulate`, `export_vtk`) rather than trusting
> the exact numbers.

## L1-27 — Module docstring
Describes the overall wizard flow (`Main -> Generators -> Filters -> Metrics ->
PoreSpy sims -> OpenPNM sims`, or `Import` instead of `Generators`) and the
four key design decisions: one Tk window per step, functions run in-place on
the main thread, working directory is temporarily switched so each module's
hardcoded output filenames land in the user-chosen folder, and stdout is
captured into the green/red result boxes.

## L29-56 — Imports & setup
- `matplotlib.use("TkAgg")` (L39) must run **before** `pyplot` is imported,
  which is why the imports are deliberately out of PEP-8 order (note the
  `noqa: E402` markers).
- L50-54 import the project's own modules (`Generators_3D`, `Filters_3D`,
  `Metrics_3D`, `Simulations3D_porespy`, `Simulations3D_openPNM`). Because
  `Simulations3D_openPNM` builds its pore network lazily (see its own notes
  file), importing it here is cheap — no network is built until the user
  actually runs an OpenPNM simulation.
- L56: `SCRIPT_DIR` anchors the GUI's working directory in `main()` so saved
  files default to a predictable location regardless of how the script was
  launched.

## L62-132 — Form-field model (`F`, `FormField`, `build_form`, `collect_kwargs`)
This is the small "form engine" the whole GUI is built on:
- `F(...)` (L62) is a plain dict-builder describing one input: its Tk widget
  type (`bool`, `int`, `float`, `choice_str`, `list_int`, ...), default value,
  and an optional tooltip.
- `FormField.get()` (L75-96) converts the raw Tk variable (always a string or
  bool under the hood) into the right Python type — e.g. `"int_none"` lets a
  blank entry mean `None` instead of crashing on `int("")`.
- `build_form()` (L99-121) lays out one `tk.Label` + input widget per field in
  a grid, wiring tooltips and click-to-toggle checkboxes.
- `collect_kwargs()` (L124-132) turns a whole section's fields into a
  `**kwargs` dict ready to pass to the underlying analysis function, raising a
  user-friendly `ValueError` if any field fails to parse.

## L138-235 — Reusable widgets
- `ScrollableFrame` (L138-156): a `Canvas` + inner `Frame` combo that gives
  every step window a scrollable body without each window re-implementing the
  scroll wiring.
- `ResultBox` (L159-187): the green/red bordered text box used to show
  captured stdout or an error message after running a function.
- `Tooltip` (L190-216): hover label bound to a widget's `<Enter>`/`<Leave>`
  events; this is what powers the `tooltip=` text on every `F(...)` field.
- `SingleSelect` (L219-235): a group of checkboxes where ticking one
  automatically unticks the others — used by the Generators window (pick
  exactly one generator) and the phase-selection section of the OpenPNM
  window (pick exactly one phase).

## L241-311 — Section builder + run-order helpers
- `build_section()` (L241-290) builds one bordered `LabelFrame` containing a
  header checkbox/combobox + the field form. The `mode` argument controls the
  header control: `"single"` (radio-like checkbox via `SingleSelect`),
  `"multi"` (independent checkbox), `"order"` (a combobox for picking a
  numeric run-order — used only by the Filters window), or `"none"`.
- `grid_sections()` (L293-299) arranges a list of section frames into an
  N-column grid — this is what gives each step window its 2- or 3-column
  layout.
- `section_is_selected()` / `section_order()` (L302-311) read back whether a
  section was ticked, or what order number was chosen.

## L314-372 — DICOM helpers (new)
A small set of private helpers added to support importing `.dcm` files
alongside the existing `.tif`/`.raw` paths. They live above `_ask_raw_params`
because both feed the same "ask for parameters, then load" pattern in
`_do_import`.
- `_dicom_read_force()` (L314-324): calls `pydicom.dcmread(path, force=True)`
  so files exported without the standard 128-byte preamble + `"DICM"` magic
  bytes (common for some scanner/PACS exports) still load instead of raising.
  If the file is missing a Transfer Syntax UID, it assumes **Implicit VR
  Little Endian** (L322-323) — the most common transfer syntax for that kind
  of raw export; a file using a different, unlabelled syntax would still fail
  or decode incorrectly.
- `_dicom_voxel_size()` (L327-335): reads `PixelSpacing` (in-plane, `dy, dx`
  order per the DICOM standard — note the order!) and `SliceThickness` to
  return physical voxel size `(dx, dy, dz)` in millimetres, defaulting any
  missing field to `1.0` rather than failing. `SliceThickness` is a coarse
  proxy for true slice spacing (it doesn't account for slice gaps/overlap),
  which is acceptable here since it's only used to inform the user.
- `_dicom_pixels()` (L338-343): applies the DICOM rescale equation
  (`stored_value * RescaleSlope + RescaleIntercept`) to convert raw stored
  pixel values into physical units — for CT this turns the raw integers into
  Hounsfield units.
- `_read_dicom_file()` (L346-358): handles a **single** `.dcm` file, which
  may itself contain multiple frames (i.e. a whole 3D volume in one file). A
  2D array is promoted to a 1-slice volume (L352-353) so the return shape is
  always `(H, W, n_slices)`, consistent with every other loader in
  `_do_import`.
- `_read_dicom_series()` (L361-372): handles a **folder** of single-slice
  `.dcm` files. L366-369 try to sort by `InstanceNumber` (the DICOM field
  that records true acquisition/anatomical order) and silently fall back to
  the pre-sorted filename order if that tag is missing or inconsistent —
  filename order is not guaranteed to match physical slice order for all
  scanner exports, so this is a best-effort heuristic, not a guarantee.

## L375-451 — `_ask_raw_params`
Dialog shown when the user imports a headerless `.raw` binary file. Since
`.raw` has no metadata, L379-390 try to **guess** the volume shape and dtype
by checking which `(dtype size) × side³` combination matches the file size
exactly (tries `uint8`, `uint16`, `float32` and integer cube roots). The
dialog (L394-450) pre-fills those guesses but lets the user override them, and
warns (L423-429) if the entered shape/dtype doesn't account for the actual
file size.

## L454-559 — Output-file plumbing
- `ask_output_dir` / `ask_save_file` (L454-458, L533-544): wrap
  `filedialog.askdirectory` / `asksaveasfilename` so the rest of the code
  doesn't repeat boilerplate.
- `_ALWAYS_SAVES` / `_FUNC_SAVE_FILE` (L462-498): because the underlying
  analysis functions in `Generators_3D`, `Filters_3D`, `Metrics_3D`,
  `Simulations3D_porespy` and `Simulations3D_openPNM` save files under
  **fixed, hardcoded names** (e.g. `porespy_drainage.vti`), the GUI needs a
  lookup table mapping each function to the name it will write, so it knows
  what to ask the user about and what to rename afterwards. **Note:** the
  OpenPNM entries in these tables are now effectively dead — the OpenPNM
  window no longer uses `get_save_file`/`_rename_saved` (see the combined-VTK
  export in `open_openpnm` below); they're kept only because the other
  windows still rely on the tables.
- `get_save_file()` / `needs_output_dir()` (L501-517): decide, for a given
  section, whether running it will produce a file at all (either because the
  function always saves, or because its own `save_excel` checkbox is ticked).
- `_rename_saved()` (L547-559): since the module functions can't be told to
  use a custom filename, the GUI lets the function save under its default
  name and then **renames** the file afterwards to whatever the user typed in
  the Save dialog. Still used by the Generators/Filters/Metrics/PoreSpy
  windows; the OpenPNM window instead writes its file with the user-chosen
  name directly via `opnm.export_vtk`.

## L562-578 — `_LARGE_VOXELS`, `_image_size_label`, and run-time helpers
- `_LARGE_VOXELS = 50_000_000` (≈ 370³): the threshold above which an image
  is considered "large" for warning purposes. Displayed in every step window.
- `_image_size_label(parent, image)` packs a single `tk.Label` at the top of
  the scrollable `body` of each step window showing the image's shape and
  total voxel count. When the count exceeds `_LARGE_VOXELS` the text turns red
  and appends "⚠ Large image: use Crop filter before memory-intensive
  operations." — this is the user's cue to apply `CROP_VOLUME` first.
- `_TIME_ESTIMATES`: dict mapping function-title strings to `(base_s, exponent)`
  pairs used by the power-law formula `t(N) = base_s × (N / 1_000_000) ^ exp`.
  **Generators are intentionally excluded**: their output size is set by the
  user via the `shape` parameter at run time, so no meaningful estimate can be
  shown when the window opens (unlike filters/metrics/sims where `self.image`
  already has a known size). All other categories (solid/NSA filters, metrics,
  PoreSpy sims, OpenPNM sims) are included.
- `_format_time(s)` converts a float number of seconds into a compact human
  string (`"<1 s"`, `"~12 s"`, `"~3 min"`, `"~1.2 h"`).
- `_est_time(title, n_voxels)` looks up the title in `_TIME_ESTIMATES` and
  returns the formatted estimate, or `"unknown"` if the title is absent.

## L580-634 — `run_operation` (the core "run a function safely" helper)
This is the most important plumbing function in the file. For any analysis
call it:
1. Pops up a non-blocking "Processing..." overlay (L588-612) — deliberately
   **without** `grab_set()` (see the comment at L608), because the underlying
   functions open native matplotlib windows that the user must be able to
   click and close.
2. Temporarily `os.chdir()`s into the chosen output directory (L618-619) so
   the module's hardcoded relative filenames land where the user picked, then
   always restores the original cwd in `finally` (L627) — even on error.
3. Captures everything printed to stdout via `contextlib.redirect_stdout`
   (L620) and strips ANSI colour codes from it (L36, L634) so the result box
   shows clean text.
4. Catches any exception, formats it for the user, and also prints the full
   traceback to the real console for debugging (L622-625).
5. **Returns a 5-tuple** `(ok, result, stdout_text, error_text, elapsed_seconds)`.
   `elapsed_seconds` is measured with `time.perf_counter()` around the actual
   computation (inside the try block), so it excludes UI overhead. Every
   `do_*` closure in the step windows accumulates `_total_elapsed` across
   multiple `run_operation` calls (e.g. solid-filter chain + NSA filters) and
   updates a `run_time_var` `StringVar` at the end, which drives the "Run time:"
   label in the button bar.

## L637-649 — `make_window`
Builds the title + scrollable body + bottom button bar skeleton shared by
every step window (Generators, Filters, Metrics, PoreSpy, OpenPNM).

## L655-969 — Parameter specifications (`GEN_SPECS`, `FILTER_GROUPS`, `METRIC_SPECS`, `PSIM_SPECS`, `PNM_PHASE_SPECS`, `PNM_SIM_SPECS`)
This whole block is **declarative data**, not logic: each entry pairs a
display title, the actual function from the project module it calls, and the
list of `F(...)` fields describing its parameters. Adding a brand-new
generator/filter/metric to the GUI is mostly a matter of adding one more
tuple here — the generic `build_section`/`run_operation` machinery handles
the rest. Note `FILTER_GROUPS` (L719-794) is split into "Solid Altering" vs
"Non-Solid Altering" filters (see `open_filters` below for why that split
matters). The "Solid Altering" group now opens with `CROP_VOLUME` (~L721)
— it crops the image to a user-specified voxel box (`x_min/x_max`, `y_min/y_max`,
`z_min/z_max`, each `int_none` so a blank means 0 / the image extent, and the
`max` bounds are **exclusive**), permanently replacing `self.image` for all
downstream steps. (It replaces the earlier `CROP_CENTER`, which could only cut a
centred cube of a fixed side length.) `_DRAIN_IMB_FIELDS` (~L895) is reused
by both `DRAINAGE` and `IMBIBITION` since they share the same parameter set.

## L975-1711 — The `App` class and `main()`
- `__init__` (~L976): holds the GUI's entire mutable state: the current
  working image, the unfiltered image (kept so "Filters" can always restart
  from the same source on repeated runs), and a cached OpenPNM network
  (`self.pn`) plus the image id / voxel size it was built from, so the
  network is only rebuilt when actually necessary (see `open_openpnm`).
- `build_main` (~L995): the landing window — choose Generate or Import.
- `run_import` / `_do_import` (~L1008-1204): the Import flow.
  - `run_import` offers **three** file types (`.tif`/`.tiff`, `.raw`, `.dcm`)
    and folders of `.tif` **or** `.dcm` slices — the chooser window is
    `380x220` with `width=34` buttons to fit the longer DICOM-aware labels.
  - `_do_import` supports four sources: a folder of 2D `.tif` slices, a
    folder of 2D `.dcm` slices, a single 3D `.tif`/volume file, a single
    (possibly multi-frame) `.dcm` file, or a headerless `.raw` file (with
    `_ask_raw_params` collecting shape/dtype first).
  - `is_dcm_file` is determined up front from the extension; `dcm_voxel_size`
    starts as `None` and is only filled in by `_load()` if a DICOM source was
    actually used — this is what later decides whether to show the
    voxel-spacing info box at all.
  - The `_load` inner function: folder loading checks for `.dcm` files first
    and only falls back to the original `imread_collection` `.tif` path if
    none are found; a single `.dcm` file is handled by the `elif is_dcm_file`
    branch ahead of the pre-existing `.raw` and generic `imageio.volread`
    branches.
  - **Axis convention (fixed July 2026):** every load path returns the working
    image as `(x, y, z)` with **app-x = ParaView X**. `imageio.volread`, the
    single-DICOM and `.raw` paths all `transpose(2, 1, 0)`; the folder-stack
    and DICOM-series `np.stack(..., axis=-1)` loaders add a `(1, 0, 2)`; and the
    binary `.tif` save (`_ALWAYS_SAVES` path) writes with `(2, 1, 0)`. These all
    used to be `(2, 0, 1)` / `(1, 2, 0)`, which silently swapped X↔Y versus
    ParaView (obvious on `pseudo_gravity_packing`). This mirrors the identical
    fix in `Fable_UI.py` — see `doc/Fable_UI_notes.md` "Volume axis convention".
  - When a DICOM source was loaded, a `messagebox.showinfo` shows the voxel
    spacing from the file header (in mm) and reminds the user to convert it
    to **metres** when entering `voxel_size` elsewhere in the GUI — this
    value is informational only and isn't auto-filled into those fields.
  - If the loaded image is **not already binary**, it runs PoreSpy's NL-means
    denoiser, shows a histogram to help pick a threshold, asks the user for a
    threshold via `simpledialog.askfloat`, and binarizes with
    `denoised < threshold` (note: `True` = pore/void by this convention).
    The NL-means call works around a bug in `ps.filters.nl_means_layered`
    (which internally loops `range(im.shape[0])` instead of the axis-swapped
    shape) by pre-moving the slice axis to position 0 with
    `np.moveaxis(im_float, 2, 0)`, calling with `axis=0`, then moving it back.
  - Near the end of `_do_import`: if the source was already a binary `.tif`
    and the user did **not** invert it, the original file already serves as
    the binary result and saving is skipped. The `if already_binary and
    user_inverted:` guard (not just `if already_binary:`) fixes an earlier
    bug where the file was always named `_inverted.tif` even when the user
    declined the inversion.
- `open_generators` (~L1207-1263): builds one `SingleSelect` section per
  entry in `GEN_SPECS`; `do_generate` enforces "exactly one ticked" and, on
  success, stores the result as both `self.image` and `self.image_unfiltered`
  and invalidates the cached OpenPNM network (`self.pn = None`). Generator
  sections show **no "Estimated time:" label** because the output image size
  is controlled by the user's `shape` field — any pre-computed estimate would
  only be valid at the exact default shape. A "Run time:" label in the button
  bar still shows the actual elapsed time after the generator finishes.
- `open_filters` (~L1266-1433): the most involved window.
  - An `_image_size_label` bar is added at the top of the scrollable body
    showing the current image dimensions. When the image exceeds 50 M voxels
    the bar turns red and prompts the user to crop first — this is what
    directs them to the `CROP_VOLUME` solid-altering filter.
  - Filtering **always** restarts from `self.image_unfiltered`, not from a
    previous filtered result — this is what makes pressing "← Previous" then
    "Filter" again behave correctly instead of compounding.
  - `_make_conflict_resolver`: since solid-altering filters run as an ordered
    **chain**, two filters can't share the same order number; this auto-clears
    the order on whichever filter previously held that number.
  - `do_filter`: solid-altering filters run sequentially, each feeding its
    output into the next, producing one chained `base` image. Non-solid-
    altering filters then each run **independently** against that same `base`
    (they don't chain with each other). Because `CROP_VOLUME` is a solid-
    altering filter, giving it order=1 crops the image first and all
    subsequent solid-altering filters in the chain receive the cropped result,
    as does `self.image` after the run — ensuring all downstream steps
    (Metrics, PoreSpy sims, OpenPNM sims) automatically use the smaller volume.
- `open_metrics` (~L1447): also shows an `_image_size_label` bar at the top,
  then multi-select sections each run independently against `self.image`.
- `open_porespy` (~L1513): same pattern plus the `_image_size_label` bar.
  Memory-intensive operations to be aware of here:
  - `TORTUOSITY`: 22 GB for 1000³ via internal `CubicTemplate` — mitigated by
    its own `crop_size` field (use ≤ 300 for large volumes).
  - `DRAINAGE` / `IMBIBITION` / `INJECTION`: `capillary_transform` alone
    allocates ~8 GB float64 for a 1000³ image; drainage arrays add more.
  The correct mitigation for all of these is `CROP_VOLUME` in the Filters step.
- `open_openpnm` (~L1579): also shows the `_image_size_label` bar.
  - A single voxel-size field shared by the whole window, since snow2 (the
    network extraction step) needs it once for the network's geometry.
  - The pore network is rebuilt via `opnm.build_network()` only if it doesn't
    exist yet, or the source image / voxel size changed since the last build —
    this is the caching behaviour `self.pn_image_id` and `self.pn_voxel_size`
    exist to support. **`build_network` runs SNOW2 on the entire image — for a
    1000³ volume this alone exhausts RAM on most workstations; crop first.**
  - Two-phase functions (`simulate_drainage`, `simulate_invasion_percolation`)
    need an *invading* and a *defending* phase, so the chosen phase becomes
    `inv_phase` and the opposite of `water`/`air` becomes `def_phase`;
    single-phase transport functions just get `chosen_phase`.
  - **Single combined VTK export.** All ticked simulations run on the *same*
    OpenPNM network/project, so `do_simulate` passes `save_vtk=False` to each
    one (suppressing their individual `.vtp` writes) and then calls
    `opnm.export_vtk(self.pn, stem)` **once** at the end. The file is named
    from the selected simulations via `_PNM_SIM_ABBR` (SF/FD/AD/Dr/IP joined by
    `_`), e.g. ticking all five yields `openpnm_SF_FD_AD_Dr_IP.vtp`. The user is
    asked for the name/location a single time (Save-As pre-filled with that
    default), and the saved path is echoed in the "Run time:" label. This is
    why every field/coloring array from every ticked simulation lives together
    in one ParaView load (a deliberate convenience — one file, switch coloring
    freely — not a bug). The per-simulation `save_excel` outputs for
    drainage/IP are unaffected and still land in the chosen folder.
- `main()` (~L1718): `chdir`s into the script's own directory first so
  relative output paths are predictable, then starts the Tk event loop.
