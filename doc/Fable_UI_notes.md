# Notes on `ui/Fable_UI.py` — "Porous Media Analysis"

Companion explanation file — does not modify the source. The file is large
(~2 800 lines), so sections are referenced **by name** rather than by line
number; search for the quoted identifiers.

## What it is

A modern single-window Tkinter app that drives the same thesis pipeline as
`app_gui.py` (Generate/Import → Filters → Metrics → PoreSpy sims → OpenPNM
sims) but with a sidebar instead of a chain of wizard windows, a background
worker thread, an in-app plot gallery, and a built-in slice viewer. All
computation calls go to the modules in the `logic/` package — nothing
scientific lives in this file.

## Start-up & module loading

- `UI_DIR` / `PROJECT_ROOT`: the file lives in `<project>/ui/`; the analysis
  modules live in the `<project>/logic/` package, so `load_project_modules()`
  puts the **project root** on `sys.path` before running `import logic.*`.
  The heavy imports (porespy/openpnm) are deferred until after the splash
  screen is painted so start-up doesn't look frozen.
- `APP_TITLE = "Porous Media Analysis"` — used in the window title, sidebar
  and splash.
- `OUTPUT_DIR_DEFAULT`: `UI_output` **next to this .py file**, or next to the
  `.exe` when frozen (`sys.frozen` → `dirname(sys.executable)`). Created in
  `main()`; the process `chdir`s there so the modules' hardcoded relative
  output filenames land somewhere predictable.
- `asset_path(name)` resolves files in `assets/` for three situations:
  PyInstaller onefile (`sys._MEIPASS`), next-to-exe, and the dev tree.
- `load_app_icon(size)`: loads `assets/Icon.png`, applies a **circular mask**
  (the artwork's baked-in background is opaque on its left edge — a straight
  edge would show otherwise) and caches per-size PhotoImages so Tk doesn't
  garbage-collect them.

## Image → screen helpers

- `_rgb_from_plane(plane, vmin, vmax)`: bool planes use the PoreSpy colours
  (yellow = pore, violet = solid); numeric planes go through a 256-entry
  viridis LUT scaled to `[vmin, vmax]` (inf/NaN-safe).
- `slice_to_photo` / `png_fit_photo`: Pillow-based scaling (NEAREST for
  voxel slices, LANCZOS for plots) with a Tk zoom/subsample fallback if
  Pillow is missing. `slice_to_photo` flips the plane vertically (`plane[::-1]`)
  so it draws **bottom-origin** — the first plane axis increases upward, like
  ParaView / matplotlib `origin="lower"`. Without it, gravity packings looked
  upside-down (spheres appeared to pile at the top of the thumbnail/slice
  viewer). Both the sidebar thumbnail and the `SliceViewer` go through here, so
  they stay consistent.
- `read_vti(path)`: minimal reader for the `.vti` files written by
  `save_results_vtk` in `logic/Simulations3D_porespy.py` (pyevtk raw-appended
  format, UInt64 block headers, Fortran-ravelled arrays). The writer now stores
  arrays in the app's `(x, y, z)` order directly (pyevtk maps array axis 0/1/2 →
  VTK X/Y/Z), so the reader just Fortran-reshapes to `(x, y, z)` with **no**
  transpose, and casts 0/1 float fields to bool. (Pre-July-2026 `.vti` files
  were written transposed `(2, 0, 1)` and will open swapped — regenerate them.)

## Volume axis convention (files ↔ memory)

Working images live in memory as `(x, y, z)` with **app-x = ParaView X**. All
external volume I/O maps to that so a generated/loaded image keeps the same
axes when opened in ParaView:

- **TIFF** (`imageio.volread`/`volsave`) — files are `(slices, rows, cols) =
  (z, y, x)`, so read and write both `transpose(2, 1, 0)`. Folder-of-slices and
  DICOM-series loaders (`np.stack(..., axis=-1)`) add a `(1, 0, 2)` to reach the
  same `(x, y, z)`.
- **VTI** (pyevtk) — passed through unchanged; see `read_vti` above.

This was corrected in **July 2026**: everything previously used
`(2, 0, 1)` / `(1, 2, 0)`, which is internally self-consistent but silently
swaps X↔Y versus ParaView (invisible on symmetric shapes, obvious on
`pseudo_gravity_packing`). Because the swap was uniform, the fix had to change
every read *and* write together — files written by the old code open swapped.

## The worker thread & Cancel

- `submit(title, func, kwargs, out_dir, on_done, no_figs)` queues one job;
  a single daemon worker executes jobs, capturing stdout/stderr and
  snapshotting matplotlib figures. Results come back to the UI thread via
  `res_q` + a 100 ms `_poll_results` loop.
- **Cancel** (`_cancel_ops`, the red ✕ button in the status bar): bumps a
  generation counter (`_job_gen`) so queued jobs and late results are
  discarded, injects `KeyboardInterrupt` into the worker via
  `PyThreadState_SetAsyncExc`, and starts a fresh worker. Python cannot kill
  a thread stuck inside C code (numpy/porespy), so a cancelled computation
  may keep burning CPU for a while — but its result is discarded and the UI
  is released immediately. `on_done` chains of cancelled jobs never fire.
- `_snapshot_figures()` saves each open figure as a PNG **and keeps the live
  `Figure` object** — see "Plot gallery" below.

## Pages

Built once in `_build_pages()`, stacked with `place`, raised by
`show_page()`. The card system (`build_card` / `grid_cards`) is declarative:
`SPEC` (filled by `build_specs()` after the modules import) pairs every
operation with its `F(...)` field list.

- `grid_cards` uses a **masonry** layout — one packed sub-frame per column —
  so expanding a card only pushes down its own column.
- Filters page: solid-altering filters have run-order comboboxes (1..5) and
  always run first as a chain; non-solid-altering filters are checkboxes and
  each independently receives the SA result.
- `ScrollableFrame` ignores the mouse wheel when the content fits the
  viewport and snaps back to the top when cards collapse.
- `ResultBox._on_scroll` consumes the wheel only while the box itself can
  still scroll in that direction; otherwise the page scrolls.

### Metric Excel export (`save_excel`) and `SPEC["SAVE_FILE"]`

- The coordination-number metric now opens a histogram plot instead of printing
  the full frequency table into the result box, while still exporting the same
  Excel sheet when `save_excel` is checked.
- `TWO_POINT_CORRELATION` is intentionally listed at the end of the Metrics
  catalog so the UI order follows the ordering in `logic/Metrics_3D.py`.

A metric writes an Excel file only if its card exposes a ticked `save_excel`
field. `get_save_file(sec)` returns a filename either when the function is in
`SPEC["ALWAYS_SAVES"]` (generators, PoreSpy/OpenPNM sims, `REGIONPROPS_3D`) or
when its `save_excel` box is checked; the filename itself comes from
`SPEC["SAVE_FILE"]`. To make a new metric exportable you therefore add both a
`F("save_excel", "bool", False, tooltip=_TT_XL)` field **and** a
`func: "name.xlsx"` entry in `SPEC["SAVE_FILE"]`. As of July 2026 the
exportable metrics include `CHORD_COUNTS` (`chord_counts.xlsx`),
`POROSITY_PROFILE` (`porosity_profile.xlsx`) and `SATN_PROFILE`
(`saturation_profile.xlsx`) alongside the earlier distribution metrics.
`REGIONPROPS_3D` also gained a `min_size` field (default 4) that drops tiny,
convex-hull-degenerate regions before the property table is built.

## Slice viewer (Home)

`SliceViewer` shows orthogonal slices of the **working image** by default,
but its "view" dropdown can also show:
- snapshots of every volume that became the working image (capped at 6),
- the filtered result after a Filters run,
- `.tif` / `.vti` files opened via "Open .tif / .vti…" (a `.vti` entry is
  parsed lazily in the background and expands into one entry per field).

The view selection never changes the working image — a ⚠ notice appears when
previewing anything else, and the sidebar always shows a mid-slice thumbnail
plus stats of the actual working image. Numeric fields render in viridis
with a colorbar (field-name title + numbered major ticks via `MaxNLocator`);
chord/label fields get colours but no bar. The slice rescales to fill the
display area on window resize (debounced `<Configure>`).

## Plot gallery & viewers

Every figure a run produces is captured into the Plots page. Clicking a
plot opens, in order of preference:
1. **Interactive matplotlib viewer** (`_open_interactive_plot`): the live
   `Figure` embedded in `FigureCanvasTkAgg` with `NavigationToolbar2Tk` —
   rectangle-zoom / pan / home / save, re-rendered as vectors so curves stay
   sharp at any zoom (equivalent to the old app_gui native windows). The
   first draw is deferred until after Tk layout (`_sync_size`), otherwise the
   figure appears off-centre until a manual resize.
2. PNG fallback viewer with drag-a-rectangle zoom, for items without a live
   figure (e.g. the import histogram).

## Packaging as an .exe (PyInstaller)

- Include the assets: `--add-data "assets;assets" --icon assets/icon.ico`.
- `OUTPUT_DIR_DEFAULT` automatically becomes `UI_output` next to the exe.
- The `logic/` package must be importable — run PyInstaller from the project
  root (e.g. `pyinstaller ui/Fable_UI.py --paths .`) so `logic` is bundled.

## Renaming the file

Renaming `Fable_UI.py` is safe: nothing imports it (it is a standalone entry
point) and `OUTPUT_DIR_DEFAULT`/asset lookup are relative to the file's own
directory, so they move with it. Update only: (1) whatever
shortcut/command/launch config starts it, (2) the PyInstaller build command
and the exe name it produces, and (3) this notes file's title. Internal
branding is already decoupled — the visible name comes from `APP_TITLE`, not
the filename.
