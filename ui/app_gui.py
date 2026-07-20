"""
Porous-media analysis GUI
=========================

A Tkinter wizard that drives the project's analysis pipeline:

    Main  ->  Generators  ->  Filters  ->  Metrics  ->  PoreSpy sims  ->  OpenPNM sims
              (or Import an image, then continue from Filters)

Design notes
------------
* Each step is its own window. The "Next" button closes the current window and
  opens the following one (this does not apply to the first/Main window).
* The action button on each window (Generate / Filter / Calculate / Simulate)
  runs the underlying functions from the project modules *in place* so the user
  can inspect the native matplotlib pop-up windows and the on-screen result
  boxes before advancing with "Next".
* Plots use the native matplotlib windows (TkAgg). Computation runs on the main
  thread; while it runs a modal "Processing..." dialog is shown and the rest of
  the UI is blocked. (Close any plot pop-ups to let the function return.)
* Functions in the project modules save files with fixed names into the current
  working directory. Before each run the user is asked for an output folder and
  the working directory is temporarily switched there, so every file that the
  run produces lands in the chosen folder without changing the module code.
* Printed output (the messages the functions send to the terminal) is captured
  and shown in green (success) / red (error) result boxes where relevant.
"""

import os
import io
import re
import sys
import traceback
import contextlib
import time

_ANSI_RE = re.compile(r'\x1b\[[\d;]*[A-Za-z]')

import matplotlib
matplotlib.use("TkAgg")          # native pop-up plot windows
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

import numpy as np               # noqa: E402

import tkinter as tk             # noqa: E402
from tkinter import ttk, filedialog, messagebox, simpledialog  # noqa: E402

# --- Project modules -------------------------------------------------------
# This file lives in <project>/ui/ ; the analysis modules live in the
# <project>/logic/ package. Put the project root (the parent of ui/) on
# sys.path so `import logic.*` resolves when the script is run directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Importing these is cheap (Simulations3D_openPNM was refactored so its network
# is built lazily via build_network()).
import logic.Generators_3D as gen          # noqa: E402
import logic.Filters_3D as filt            # noqa: E402
import logic.Metrics_3D as met             # noqa: E402
import logic.Simulations3D_porespy as psim  # noqa: E402
import logic.Simulations3D_openPNM as opnm  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
#  Form-field model + value parsing
# ===========================================================================
def F(name, ftype, default, choices=None, label=None, tooltip=None):
    """Create a field description used to build a form widget."""
    return {"name": name, "type": ftype, "default": default,
            "choices": choices, "label": label or name, "tooltip": tooltip}


class FormField:
    """Wraps a Tk variable and knows how to convert its content to a Python value."""

    def __init__(self, field, var):
        self.field = field
        self.var = var

    def get(self):
        f = self.field
        t = f["type"]
        if t == "bool":
            return bool(self.var.get())
        s = str(self.var.get()).strip()
        if t in ("int_none",) and s == "":
            return None
        if t == "float_none" and s == "":
            return None
        if t == "choice_int_none" and s in ("", "None"):
            return None
        if t in ("int", "choice_int", "int_none", "choice_int_none"):
            return int(s)
        if t in ("float", "float_none"):
            return float(s)
        if t == "list_int":
            return [int(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
        if t == "list_float":
            return [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
        # str, choice_str, choice_conn, ...
        return s


def build_form(parent, fields):
    """Build a grid of labelled widgets; return {name: FormField}."""
    ff = {}
    for i, f in enumerate(fields):
        lbl = tk.Label(parent, text=f["label"] + ":")
        lbl.grid(row=i, column=0, sticky="w", padx=4, pady=2)
        if f.get("tooltip"):
            Tooltip(lbl, f["tooltip"])
        t = f["type"]
        if t == "bool":
            var = tk.BooleanVar(value=bool(f["default"]))
            tk.Checkbutton(parent, variable=var).grid(row=i, column=1, sticky="w")
            lbl.configure(cursor="hand2")
            lbl.bind("<Button-1>", lambda e, v=var: v.set(not v.get()))
        elif t.startswith("choice"):
            var = tk.StringVar(value=str(f["default"]))
            ttk.Combobox(parent, textvariable=var, values=f["choices"],
                         state="readonly", width=18).grid(row=i, column=1, sticky="w", padx=4)
        else:
            var = tk.StringVar(value=str(f["default"]))
            tk.Entry(parent, textvariable=var, width=20).grid(row=i, column=1, sticky="w", padx=4)
        ff[f["name"]] = FormField(f, var)
    return ff


def collect_kwargs(ff):
    """Read all fields; raises ValueError with a helpful message on bad input."""
    out = {}
    for name, field in ff.items():
        try:
            out[name] = field.get()
        except ValueError:
            raise ValueError(f"Invalid value for '{field.field['label']}'.")
    return out


# ===========================================================================
#  Reusable widgets
# ===========================================================================
class ScrollableFrame(tk.Frame):
    """A vertically scrollable container; add children to `self.inner`."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(canvas)
        self._win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(self._win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.canvas = canvas


class ResultBox(tk.Frame):
    """A selectable text area with a coloured border for captured output / errors."""

    def __init__(self, master):
        super().__init__(master, bg="#bbbbbb", padx=2, pady=2)
        self._text = tk.Text(self, height=4, bg="white", fg="#000000",
                             relief="flat", wrap="word", state="disabled",
                             font=("Segoe UI", 10), cursor="xterm")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)
        # Scroll inside the box without moving the outer window.
        self._text.bind("<MouseWheel>", self._on_scroll)
        self._set("(no result yet)")

    def _on_scroll(self, event):
        # Scroll the box only while it still has content in that direction;
        # at its top/bottom (or when empty) let the event through so the
        # outer window keeps scrolling instead.
        first, last = self._text.yview()
        down = event.delta < 0
        if (down and last >= 1.0) or (not down and first <= 0.0):
            return None
        self._text.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _set(self, content):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", content)
        self._text.configure(state="disabled")

    def show(self, text, ok=True):
        self.configure(bg="#2e7d32" if ok else "#c62828")
        self._set(text if text else ("Done." if ok else "Error"))


class Tooltip:
    """Lemonchiffon hover label shown when the cursor rests on a widget."""

    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._win = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self._win or not self._text:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 2
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text, background="lemonchiffon",
                 relief="solid", borderwidth=1, wraplength=300,
                 font=("Segoe UI", 9), justify="left",
                 padx=5, pady=3).pack()

    def _hide(self, _event=None):
        if self._win:
            self._win.destroy()
            self._win = None


class SingleSelect:
    """Group of checkboxes where ticking one unticks the others."""

    def __init__(self):
        self.vars = []

    def add(self, parent):
        v = tk.BooleanVar(value=False)

        def _enforce(*_):
            if v.get():
                for other in self.vars:
                    if other is not v:
                        other.set(False)
        v.trace_add("write", _enforce)
        self.vars.append(v)
        return tk.Checkbutton(parent, variable=v), v


# ===========================================================================
#  Section builder + run helpers
# ===========================================================================
def build_section(parent, title, fields, mode="multi", single=None,
                  order_values=None, with_result=False, subtitle=None, est_time=None):
    """Create one bordered section with a header control and a form.

    mode: "single" (single-select checkbox), "multi" (independent checkbox),
          "order" (ordering combobox), or "none".
    Returns a dict describing the section.
    """
    lf = tk.LabelFrame(parent, padx=8, pady=6)
    header = tk.Frame(lf)
    header.pack(fill="x")

    sel = {"mode": mode, "title": title, "var": None}
    if mode == "single":
        cb, var = single.add(header)
        cb.pack(side="left")
        sel["var"] = var
    elif mode == "multi":
        var = tk.BooleanVar(value=False)
        tk.Checkbutton(header, variable=var).pack(side="left")
        sel["var"] = var
    elif mode == "order":
        var = tk.StringVar(value="")
        tk.Label(header, text="run order:").pack(side="left")
        ttk.Combobox(header, textvariable=var, values=order_values,
                     state="readonly", width=4).pack(side="left", padx=(2, 0))
        sel["var"] = var

    title_lbl = tk.Label(header, text=title, font=("Segoe UI", 10, "bold"))
    title_lbl.pack(side="left", padx=8)
    if mode in ("single", "multi"):
        title_lbl.configure(cursor="hand2")
        title_lbl.bind("<Button-1>", lambda e, v=var: v.set(not v.get()))
    if subtitle:
        tk.Label(header, text=subtitle, fg="#777").pack(side="left")

    form = tk.Frame(lf)
    form.pack(fill="x", anchor="w", pady=4)
    sel["ff"] = build_form(form, fields)

    sel["result"] = None
    if with_result:
        rb = ResultBox(lf)
        rb.pack(fill="x", pady=(4, 2))
        sel["result"] = rb

    if est_time is not None:
        tk.Label(lf, text=f"Est: {est_time}", fg="#888",
                 font=("Segoe UI", 8, "italic"), anchor="e").pack(
            fill="x", padx=4, pady=(0, 2))

    # The frame is placed by grid_sections() so windows can lay sections out in
    # multiple columns.
    sel["frame"] = lf
    return sel


def grid_sections(container, sels, columns):
    """Arrange the section frames in `sels` into a grid with `columns` columns."""
    for c in range(columns):
        container.columnconfigure(c, weight=1, uniform="sec")
    for idx, sel in enumerate(sels):
        r, c = divmod(idx, columns)
        sel["frame"].grid(row=r, column=c, sticky="nsew", padx=6, pady=6)


def section_is_selected(sel):
    if sel["mode"] in ("single", "multi"):
        return bool(sel["var"].get())
    return False


def section_order(sel):
    """For 'order' sections: return the chosen integer or None."""
    s = str(sel["var"].get()).strip()
    return int(s) if s else None


def _dicom_read_force(path):
    """Read a DICOM file, tolerating a missing file-meta header (no 'DICM'
    preamble) which some scanner exports omit. Assumes Implicit VR Little
    Endian — the most common transfer syntax for such raw exports — if the
    Transfer Syntax UID is absent."""
    import pydicom
    from pydicom.uid import ImplicitVRLittleEndian
    ds = pydicom.dcmread(path, force=True)
    if "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    return ds


def _dicom_voxel_size(ds):
    """Return (dx, dy, dz) pixel/slice spacing in mm, falling back to 1.0 for
    any missing field."""
    try:
        dy, dx = (float(v) for v in ds.PixelSpacing)
    except Exception:
        dx = dy = 1.0
    dz = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
    return dx, dy, dz


def _dicom_pixels(ds):
    """Return the rescaled pixel array (Hounsfield units if CT) as float64."""
    arr = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr * slope + intercept


def _read_dicom_file(path):
    """Load a single .dcm file. Supports multi-frame (3D-in-one-file) DICOM;
    a plain single-frame file is returned as a 1-slice volume.
    Returns (array shaped (H, W, n_slices), (dx, dy, dz) voxel size in mm)."""
    ds = _dicom_read_force(path)
    arr = _dicom_pixels(ds)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))   # (frames,H,W)=(z,y,x) → (x,y,z)
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape}")
    return arr, _dicom_voxel_size(ds)


def _read_dicom_series(paths):
    """Load a folder of single-slice .dcm files into one 3D volume, ordered
    by InstanceNumber when available (falls back to filename order).
    Returns (array shaped (H, W, n_slices), (dx, dy, dz) voxel size in mm)."""
    datasets = [_dicom_read_force(p) for p in paths]
    try:
        datasets.sort(key=lambda d: int(d.InstanceNumber))
    except Exception:
        pass  # keep filename-sorted order (paths were pre-sorted)
    slices = [_dicom_pixels(ds) for ds in datasets]
    # slices are (rows, cols) = (y, x); stack to z then swap to (x, y, z)
    arr = np.transpose(np.stack(slices, axis=-1), (1, 0, 2))
    return arr, _dicom_voxel_size(datasets[0])


def _ask_raw_params(parent, file_size):
    """Dialog to collect shape and dtype for a headerless .raw file.
    Auto-suggests values from the file size. Returns (shape_tuple, dtype_str) or (None, None)."""
    suggested_shape, suggested_dtype = "", "uint8"
    for dtype_bytes, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
        n = file_size // dtype_bytes
        if n * dtype_bytes != file_size:
            continue
        side = round(n ** (1 / 3))
        for s in range(max(1, side - 2), side + 3):
            if s ** 3 == n:
                suggested_shape = f"{s},{s},{s}"
                suggested_dtype = dtype_name
                break
        if suggested_shape:
            break

    result = {"shape": None, "dtype": None}

    dlg = tk.Toplevel(parent)
    dlg.title("Raw image parameters")
    dlg.resizable(False, False)
    dlg.transient(parent)
    dlg.grab_set()

    tk.Label(dlg, text="Raw image import",
             font=("Segoe UI", 11, "bold")).pack(padx=16, pady=(14, 4))
    tk.Label(dlg, text=f"File size: {file_size:,} bytes\n"
                       "Enter the volume dimensions and data type.",
             fg="#555", justify="center").pack(padx=16, pady=(0, 10))

    frm = tk.Frame(dlg)
    frm.pack(fill="x", padx=16, pady=4)
    tk.Label(frm, text="Shape (slices, H, W):", anchor="w").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 8))
    shape_var = tk.StringVar(value=suggested_shape)
    tk.Entry(frm, textvariable=shape_var, width=22).grid(row=0, column=1, sticky="w", pady=3)
    tk.Label(frm, text="Data type:", anchor="w").grid(row=1, column=0, sticky="w", pady=3, padx=(0, 8))
    dtype_var = tk.StringVar(value=suggested_dtype)
    ttk.Combobox(frm, textvariable=dtype_var, state="readonly", width=20,
                 values=["uint8", "uint16", "uint32", "int16", "float32", "float64"]
                 ).grid(row=1, column=1, sticky="w", pady=3)

    def _ok():
        try:
            parts = [int(x.strip()) for x in shape_var.get().replace(";", ",").split(",") if x.strip()]
            if len(parts) != 3:
                raise ValueError("Need exactly 3 values.")
            expected = parts[0] * parts[1] * parts[2] * np.dtype(dtype_var.get()).itemsize
            if expected != file_size:
                if not messagebox.askyesno(
                        "Size mismatch",
                        f"Shape {parts} × {dtype_var.get()} needs {expected:,} bytes "
                        f"but file has {file_size:,} bytes.\nContinue anyway?",
                        parent=dlg):
                    return
            result["shape"] = tuple(parts)
            result["dtype"] = dtype_var.get()
            dlg.destroy()
        except ValueError as e:
            messagebox.showwarning("Invalid input",
                                   f"Enter three comma-separated integers.\n{e}", parent=dlg)

    bf = tk.Frame(dlg)
    bf.pack(pady=(8, 14))
    tk.Button(bf, text="OK", width=10, command=_ok).pack(side="left", padx=6)
    tk.Button(bf, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=6)

    dlg.update_idletasks()
    try:
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - dlg.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - dlg.winfo_height() // 2
        dlg.geometry(f"+{max(px, 0)}+{max(py, 0)}")
    except tk.TclError:
        pass
    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
    dlg.wait_window()
    return result["shape"], result["dtype"]


def ask_output_dir(parent):
    """Ask for an output folder; None -> keep current working directory."""
    d = filedialog.askdirectory(parent=parent,
                                title="Choose a folder for the files this step will save")
    return d or None


# Functions that always write files regardless of their parameters.
_ALWAYS_SAVES = frozenset([
    gen.BLOBS, gen.CYLINDERS, gen.OVERLAPPING_SPHERES,
    gen.POLYDISPERSE_SPHERES, gen.PSEUDO_ELECTROSTATIC_PACKING,
    gen.PSEUDO_GRAVITY_PACKING, gen.VORONOI_EDGES,
    psim.DRAINAGE, psim.IMBIBITION, psim.INJECTION,
    met.REGIONPROPS_3D,
    opnm.simulate_stokes_flow, opnm.simulate_fickian_diffusion,
    opnm.simulate_advection_diffusion, opnm.simulate_drainage,
    opnm.simulate_invasion_percolation,
])

# Default filename saved by each function (used to pre-fill the dialog and to rename).
_FUNC_SAVE_FILE = {
    gen.BLOBS: "Blobs.tif",
    gen.CYLINDERS: "Cylinders.tif",
    gen.OVERLAPPING_SPHERES: "Overlapping_Spheres.tif",
    gen.POLYDISPERSE_SPHERES: "Polydisperse_Spheres.tif",
    gen.PSEUDO_ELECTROSTATIC_PACKING: "Pseudo_Electrostatic_Packing.tif",
    gen.PSEUDO_GRAVITY_PACKING: "Pseudo_Gravity_Packing.tif",
    gen.VORONOI_EDGES: "Voronoi_Edges.tif",
    met.CHORD_COUNTS: "chord_counts.xlsx",
    met.CHORD_COUNTS_SOLID_PHASE: "chord_counts_solid_phase.xlsx",
    met.CHORD_LENGTH_DISTRIBUTION: "chord_length_distribution.xlsx",
    met.CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE: "chord_length_distribution_solid_phase.xlsx",
    met.COORDINATION_NUMBER: "coordination_number.xlsx",
    met.TWO_POINT_CORRELATION: "two_point_correlation.xlsx",
    met.LINEAL_PATH_DISTRIBUTION: "lineal_path_distribution.xlsx",
    met.POROSITY_PROFILE: "porosity_profile.xlsx",
    met.SATN_PROFILE: "saturation_profile.xlsx",
    met.PC_MAP_TO_PC_CURVE: "pc_map_to_pc_curve.xlsx",
    met.PORE_SIZE_DISTRIBUTION: "pore_size_distribution.xlsx",
    met.RADIAL_DENSITY_DISTRIBUTION: "radial_density_distribution.xlsx",
    met.REGION_SURFACE_AREAS: "region_surface_areas.xlsx",
    met.REGION_VOLUMES: "region_volumes.xlsx",
    met.REGIONPROPS_3D: "regionprops_3D.xlsx",
    psim.DRAINAGE: "porespy_drainage.vti",
    psim.IMBIBITION: "porespy_imbibition.vti",
    psim.INJECTION: "porespy_injection.vti",
    opnm.simulate_stokes_flow: "openpnm_stokes_flow.vtp",
    opnm.simulate_fickian_diffusion: "openpnm_fickian_diffusion.vtp",
    opnm.simulate_advection_diffusion: "openpnm_advection_diffusion.vtp",
    opnm.simulate_drainage: "openpnm_drainage.vtp",
    opnm.simulate_invasion_percolation: "openpnm_invasion_percolation.vtp",
}


def get_save_file(sec):
    """Return the filename this section will write, or None if it won't save."""
    func = sec.get("func")
    if func in _ALWAYS_SAVES:
        return _FUNC_SAVE_FILE.get(func)
    if "save_excel" in sec.get("ff", {}):
        try:
            if sec["ff"]["save_excel"].get():
                return _FUNC_SAVE_FILE.get(func)
        except Exception:
            pass
    return None


def needs_output_dir(sections):
    """Return True if any section will write files to disk."""
    return any(get_save_file(s) is not None for s in sections)


def _filetypes_for(filename):
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    if ext in (".tif", ".tiff"):
        return [("TIFF image", "*.tif *.tiff"), ("All files", "*.*")]
    if ext == ".xlsx":
        return [("Excel spreadsheet", "*.xlsx"), ("All files", "*.*")]
    if ext == ".vti":
        return [("VTK ImageData", "*.vti"), ("All files", "*.*")]
    if ext == ".vtp":
        return [("VTK PolyData", "*.vtp"), ("All files", "*.*")]
    return [("All files", "*.*")]


def ask_save_file(parent, default_name):
    """Native Save As dialog; returns (directory, filename) or (None, None)."""
    path = filedialog.asksaveasfilename(
        parent=parent,
        initialfile=default_name,
        defaultextension=os.path.splitext(default_name)[1] if default_name else "",
        filetypes=_filetypes_for(default_name),
        title="Save output file",
    )
    if not path:
        return None, None
    return os.path.dirname(path), os.path.basename(path)


def _rename_saved(out_dir, default_name, chosen_name, subdir=None):
    """Rename a file from its module-hardcoded default_name to chosen_name."""
    target = (chosen_name or "").strip() or default_name
    if not out_dir or not default_name or target == default_name:
        return
    base = os.path.join(out_dir, subdir) if subdir else out_dir
    old = os.path.join(base, default_name)
    new = os.path.join(base, target)
    if os.path.exists(old):
        try:
            os.rename(old, new)
        except OSError as e:
            print(f"Warning: could not rename {default_name!r} → {target!r}: {e}")


_LARGE_VOXELS = 50_000_000  # ≈ 370³ — threshold above which memory-intensive ops risk OOM

# Estimated run times: (base_seconds at 1 M voxels, scaling exponent α)
# t(N) = base_s × (N / 1_000_000) ^ α  — rough guide, varies ±3× with hardware.
_TIME_ESTIMATES = {
    # Generators are omitted: their output size is set by the user via the
    # "shape" parameter, so no meaningful estimate can be shown at window-open.
    # Solid altering filters
    "CROP_VOLUME":                  (0.05, 1.0),
    "FILL_CLOSED_PORES":            (0.5,  1.0),
    "TRIM_DISCONNECTED_VOXELS":     (0.8,  1.0),
    "TRIM_FLOATING_SOLID":          (0.5,  1.0),
    "TRIM_NONPERCOLATING_PATHS":    (1.5,  1.0),
    # Non-solid altering filters
    "APPLY_CHORDS":                 (0.3,  1.0),
    "APPLY_CHORDS_3D":              (1.0,  1.0),
    "APPLY_CHORDS_SOLID_PHASE":     (0.3,  1.0),
    "APPLY_CHORDS_3D_SOLID_PHASE":  (1.0,  1.0),
    "FIND_DISCONNECTED_VOXELS":     (0.5,  1.0),
    "LOCAL_THICKNESS":              (5.0,  1.1),
    "POROSIMETRY":                  (10.0, 1.1),
    "REGION_SIZE":                  (1.0,  1.0),
    "SNOW2_REGIONS":                (15.0, 1.2),
    # Metrics
    "BOND_NUMBER":                  (5.0,  1.1),
    "CHORD_COUNTS":                 (1.0,  1.0),
    "CHORD_COUNTS_SOLID_PHASE":     (1.0,  1.0),
    "CHORD_LENGTH_DISTRIBUTION":    (1.5,  1.0),
    "CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE": (1.5, 1.0),
    "COORDINATION_NUMBER":          (1.0,  1.0),
    "TWO_POINT_CORRELATION":        (1.0,  1.0),
    "FIND_H":                       (30.0, 1.2),
    "FIND_POROSITY_THRESHOLD":      (0.5,  1.0),
    "IS_PERCOLATING":               (0.5,  1.0),
    "LINEAL_PATH_DISTRIBUTION":     (3.0,  1.0),
    "MESH_SURFACE_AREA":            (2.0,  1.0),
    "MESH_VOLUME":                  (2.0,  1.0),
    "PC_MAP_TO_PC_CURVE":           (30.0, 1.2),
    "PERCOLATING_POROSITY":         (0.5,  1.0),
    "PORE_SIZE_DISTRIBUTION":       (10.0, 1.1),
    "POROSITY":                     (0.05, 1.0),
    "POROSITY_PROFILE":             (0.2,  1.0),
    "RADIAL_DENSITY_DISTRIBUTION":  (1.0,  1.0),
    "REGION_SURFACE_AREAS":         (15.0, 1.2),
    "REGION_VOLUMES":               (15.0, 1.2),
    "REGIONPROPS_3D":               (5.0,  1.1),
    "REV_POROSITY":                 (5.0,  1.1),
    "SATN_PROFILE":                 (30.0, 1.2),
    # PoreSpy simulations
    "DRAINAGE":                     (30.0, 1.2),
    "IMBIBITION":                   (30.0, 1.2),
    "INJECTION":                    (45.0, 1.2),
    "TORTUOSITY":                   (20.0, 1.5),
    # OpenPNM simulations (includes ~15 s build_network at 1 M vox)
    "Stokes flow":                  (18.0, 1.2),
    "Fickian diffusion":            (18.0, 1.2),
    "Advection-diffusion":          (22.0, 1.2),
    "Drainage":                     (17.0, 1.2),
    "Invasion percolation":         (17.0, 1.2),
}


def _format_time(s):
    """Format a duration in seconds as a compact human-readable string."""
    if s < 1:
        return "<1 s"
    if s < 60:
        return f"~{int(s)} s"
    if s < 3600:
        return f"~{s / 60:.0f} min"
    return f"~{s / 3600:.1f} h"


def _est_time(title, n_voxels):
    """Return a rough estimated run time string for section *title*."""
    entry = _TIME_ESTIMATES.get(title)
    if entry is None:
        return "unknown"
    base_s, exp = entry
    return _format_time(base_s * (n_voxels / 1_000_000) ** exp)


def _image_size_label(parent, image):
    """Pack an image-size info bar into parent; turns red when the image is large."""
    if image is None:
        return
    total = int(np.prod(image.shape))
    txt = (f"Current image: {image.shape[0]} × {image.shape[1]} × {image.shape[2]}"
           f"  ({total / 1e6:.1f} M voxels)")
    color = "#555555"
    if total > _LARGE_VOXELS:
        txt += "  —  ⚠ Large image: use Crop filter before memory-intensive operations."
        color = "#b71c1c"
    tk.Label(parent, text=txt, fg=color, font=("Segoe UI", 9, "italic"),
             anchor="w").pack(fill="x", padx=10, pady=(4, 0))


def run_operation(parent, title, func, kwargs, output_dir=None):
    """Run `func(**kwargs)` on the main thread behind a modal 'Processing' dialog.

    Captures stdout and temporarily switches the working directory to
    `output_dir` (so the module's hard-coded relative filenames are saved there).

    Returns (ok, result, stdout_text, error_text, elapsed_seconds).
    """
    overlay = tk.Toplevel(parent)
    overlay.title("Please wait")
    overlay.transient(parent)
    overlay.resizable(False, False)
    tk.Label(overlay, text=f"⏳  {title}",
             font=("Segoe UI", 12, "bold"), padx=30).pack(pady=(18, 4))
    tk.Label(overlay, text="Processing... please wait.",
             padx=30, justify="center").pack(pady=(0, 6))
    pb = ttk.Progressbar(overlay, mode="indeterminate", length=260)
    pb.pack(pady=(0, 16))
    pb.start(12)

    # Centre the overlay over the parent window.
    overlay.update_idletasks()
    try:
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - overlay.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - overlay.winfo_height() // 2
        overlay.geometry(f"+{max(px, 0)}+{max(py, 0)}")
    except tk.TclError:
        pass
    # NOTE: deliberately no grab_set() here. The functions open native matplotlib
    # windows that the user must be able to interact with and close (closing them
    # lets the blocking call return); a global grab would prevent that.
    overlay.lift()
    parent.update()  # force the dialog to paint before the (blocking) computation

    buf = io.StringIO()
    ok, result, err = True, None, None
    elapsed = 0.0
    old_cwd = os.getcwd()
    _t_start = time.perf_counter()
    try:
        if output_dir:
            os.chdir(output_dir)
        with contextlib.redirect_stdout(buf):
            result = func(**kwargs)
    except Exception as e:  # noqa: BLE001  (surface any failure to the user)
        ok = False
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        traceback.print_exc()  # full traceback to the real console for debugging
    finally:
        elapsed = time.perf_counter() - _t_start
        os.chdir(old_cwd)
        try:
            pb.stop()
            overlay.grab_release()
            overlay.destroy()
        except tk.TclError:
            pass
    return ok, result, _ANSI_RE.sub('', buf.getvalue()).strip(), err, elapsed


def make_window(app, title, intro):
    """Create a step window with a title, scrollable body and a bottom button bar."""
    win = tk.Toplevel(app.root)
    win.title(title)
    win.geometry("960x684")
    tk.Label(win, text=title, font=("Segoe UI", 14, "bold")).pack(pady=(10, 2))
    if intro:
        tk.Label(win, text=intro, fg="#555", wraplength=640, justify="left").pack(padx=12, pady=(0, 6))
    bar = tk.Frame(win)
    bar.pack(side="bottom", fill="x", pady=8, padx=10)
    sf = ScrollableFrame(win)
    sf.pack(side="top", fill="both", expand=True)
    return win, sf.inner, bar


# ===========================================================================
#  Parameter specifications  (label, function, fields)
# ===========================================================================
AXIS3 = ["0", "1", "2"]
AXIS_NONE = ["None", "0", "1", "2"]
XYZ = ["x", "y", "z"]
XYZ_ALL = ["x", "y", "z", "all"]

GEN_SPECS = [
    ("BLOBS", gen.BLOBS, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("porosity", "float", "0.5", tooltip="Target fraction of void space"),
        F("blobiness", "int", "1", tooltip="Higher value → more, smaller blobs"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
    ("CYLINDERS", gen.CYLINDERS, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("radius", "int", "4", tooltip="Cylinder radius (voxels)"),
        F("ncylinders", "int", "25", tooltip="Number of cylinders to insert"),
        F("porosity", "float", "0.5", tooltip="Target fraction of void space"),
        F("cyl_length", "int_none", "", tooltip="Cylinder length (voxels); empty → spans full domain. When set, porosity controls insertion count"),
        F("phi_max", "float", "90", tooltip="Max rotation around z-axis (degrees)"),
        F("theta_max", "float", "90", tooltip="Max rotation around y-axis (degrees)"),
        F("maxiter", "int", "10", tooltip="Max iterations to match target porosity"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
    ("OVERLAPPING_SPHERES", gen.OVERLAPPING_SPHERES, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("radius", "int", "3", tooltip="Sphere radius (voxels)"),
        F("porosity", "float", "0.5", tooltip="Target fraction of void space"),
        F("maxiter", "int", "10", tooltip="Max iterations to match target porosity"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
    ("POLYDISPERSE_SPHERES", gen.POLYDISPERSE_SPHERES, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("porosity", "float", "0.5", tooltip="Target fraction of void space"),
        F("r_min", "int", "3", tooltip="Minimum sphere radius (voxels)"),
        F("nbins", "int", "5", tooltip="Number of discrete sphere size classes"),
        F("loc", "float", "10", tooltip="Mean of sphere size distribution; larger → bigger spheres"),
        F("scale", "float", "5", tooltip="Std dev of size distribution; larger → more size variation"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
    ("PSEUDO_ELECTROSTATIC_PACKING", gen.PSEUDO_ELECTROSTATIC_PACKING, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("r", "int", "3", tooltip="Sphere radius (voxels)"),
        F("clearance", "int", "0", tooltip="Gap between spheres (voxels); negative → allow overlap"),
        F("protrusion", "int", "0", tooltip="Max voxels spheres can extend beyond domain boundary"),
        F("maxiter", "int", "500", tooltip="Max number of spheres to insert"),
        F("phi", "float_none", "0.3", label="phi (opt.)", tooltip="Target solid volume fraction; overrides maxiter when set"),
        F("compactness", "float", "0.5", tooltip="Packing tightness; <1 → looser packing"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
    ("PSEUDO_GRAVITY_PACKING", gen.PSEUDO_GRAVITY_PACKING, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("r", "int", "5", tooltip="Sphere radius (voxels)"),
        F("maxiter", "int", "200", tooltip="Max number of spheres to insert"),
        F("clearance", "int", "0", tooltip="Gap between spheres (voxels); negative → allow overlap"),
        F("pk_axis", "choice_int", "0", AXIS3, tooltip="Axis along which gravity acts"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
    ("VORONOI_EDGES", gen.VORONOI_EDGES, [
        F("shape", "list_int", "100,100,100", tooltip="3D volume dimensions as 'x,y,z'"),
        F("thickness", "int", "3", tooltip="Edge thickness (voxels)"),
        F("ncells", "int", "50", tooltip="Number of Voronoi cells"),
        F("flat_faces", "bool", True, tooltip="Make domain-face edges flat"),
        F("invert", "bool", False, tooltip="Swap pore and solid phases")]),
]

_TT_CONN_26 = "Voxel connectivity: 6 (face-adjacent only) or 26 (includes edges and corners)"
_TT_CONN = "max = corners count as connections; min = faces only"
_TT_SLICE = "Axis for the 2D visualization slice (0=x, 1=y, 2=z)"
_TT_SHOW_SLICE = "Show the 2D slice visualization"

FILTER_GROUPS = [
    ("Solid Altering Filters", [
        ("CROP_VOLUME", filt.CROP_VOLUME, [
            F("x_min", "int_none", "0", tooltip="Start voxel along axis 0 (x), inclusive; blank = 0. "
                                                "The max is exclusive, so x 50→150 keeps 100 voxels."),
            F("x_max", "int_none", "", tooltip="End voxel along axis 0 (x), EXCLUSIVE; blank = image extent. "
                                               "e.g. min 50, max 150 keeps 100 voxels (indices 50…149)."),
            F("y_min", "int_none", "0", tooltip="Start voxel along axis 1 (y), inclusive; blank = 0. "
                                                "The max is exclusive, so y 100→200 keeps 100 voxels."),
            F("y_max", "int_none", "", tooltip="End voxel along axis 1 (y), EXCLUSIVE; blank = image extent. "
                                               "e.g. min 100, max 200 keeps 100 voxels (indices 100…199)."),
            F("z_min", "int_none", "0", tooltip="Start voxel along axis 2 (z), inclusive; blank = 0. "
                                                "The max is exclusive, so z 0→100 keeps 100 voxels."),
            F("z_max", "int_none", "", tooltip="End voxel along axis 2 (z), EXCLUSIVE; blank = image extent. "
                                               "e.g. min 0, max 100 keeps 100 voxels (indices 0…99). "
                                               "Permanently replaces the working image — all subsequent steps use "
                                               "the cropped volume. Crop 1000³+ images to ≤ ~300³ to avoid OOM errors."),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("FILL_CLOSED_PORES", filt.FILL_CLOSED_PORES, [
            F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("TRIM_DISCONNECTED_VOXELS", filt.TRIM_DISCONNECTED_VOXELS, [
            F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
            F("inlet_faces", "choice_str", "all", ["all", "x0", "x1", "y0", "y1", "z0", "z1"],
              tooltip="Faces used as inlets: all, x0/x1, y0/y1, z0/z1"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("TRIM_FLOATING_SOLID", filt.TRIM_FLOATING_SOLID, [
            F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("TRIM_NONPERCOLATING_PATHS", filt.TRIM_NONPERCOLATING_PATHS, [
            F("perc_axis", "choice_int", "0", AXIS3, tooltip="Axis along which percolation is tested"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
    ]),
    ("Non-Solid Altering Filters", [
        ("APPLY_CHORDS", filt.APPLY_CHORDS, [
            F("spacing", "int", "1", tooltip="Voxels between parallel chord lines"),
            F("axis", "choice_int", "1", AXIS3, tooltip="Axis along which chords are drawn"),
            F("trim_edges", "bool", False, tooltip="Remove chords that touch the domain boundary"),
            F("label", "bool", True, tooltip="Assign a unique label to each chord"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("APPLY_CHORDS_3D", filt.APPLY_CHORDS_3D, [
            F("spacing", "int", "2", tooltip="Voxels between parallel chord lines"),
            F("trim_edges", "bool", True, tooltip="Remove chords that touch the domain boundary"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("APPLY_CHORDS_SOLID_PHASE", filt.APPLY_CHORDS_SOLID_PHASE, [
            F("spacing", "int", "1", tooltip="Voxels between parallel chord lines"),
            F("axis", "choice_int", "1", AXIS3, tooltip="Axis along which chords are drawn"),
            F("trim_edges", "bool", False, tooltip="Remove chords that touch the domain boundary"),
            F("label", "bool", True, tooltip="Assign a unique label to each chord"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("APPLY_CHORDS_3D_SOLID_PHASE", filt.APPLY_CHORDS_3D_SOLID_PHASE, [
            F("spacing", "int", "2", tooltip="Voxels between parallel chord lines"),
            F("trim_edges", "bool", True, tooltip="Remove chords that touch the domain boundary"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("FIND_DISCONNECTED_VOXELS", filt.FIND_DISCONNECTED_VOXELS, [
            F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
            F("show_3d", "bool", True, tooltip="Show 3D visualization of disconnected voxels")]),
        ("LOCAL_THICKNESS", filt.LOCAL_THICKNESS, [
            F("sizes", "int", "10", tooltip="Number of size classes for the thickness map"),
            F("method", "choice_str", "dt", ["dt", "conv", "bf", "imj"],
              tooltip="Algorithm: dt (fastest), conv, bf, imj"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("POROSIMETRY", filt.POROSIMETRY, [
            F("sizes", "int", "15", tooltip="Number of sphere diameters to test"),
            F("method", "choice_str", "dt", ["dt", "conv", "dsi"],
              tooltip="Algorithm: dt, conv, dsi"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE),
            F("inv_axis", "choice_int_none", "None", AXIS_NONE,
              tooltip="Invasion axis; None → invasion from all faces")]),
        ("REGION_SIZE", filt.REGION_SIZE, [
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
        ("SNOW2_REGIONS", filt.SNOW2_REGIONS, [
            F("voxel_size", "float", "1", tooltip="Physical length of each voxel (m) — used by snow2 for pore geometry"),
            F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
            F("show_slice", "bool", True, tooltip=_TT_SHOW_SLICE)]),
    ]),
]

_TT_VS = "Physical length of each voxel (m)"
_TT_XL = "Export results to Excel"
_TT_BINS = "Number of histogram bins"
_TT_LOG = "Use logarithmic x-axis"
_TT_SIGMA = "Fluid-fluid surface tension (N/m)"
_TT_THETA = "Contact angle through invading phase (degrees)"
_TT_G = "Gravitational acceleration (m/s²)"

METRIC_SPECS = [
    ("BOND_NUMBER", met.BOND_NUMBER, [
        F("g", "float", "9.81", tooltip=_TT_G),
        F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
        F("delta_rho", "float", "1000", tooltip="Density difference between phases (kg/m³)"),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("source", "choice_str", "lt", ["lt", "dt"],
          tooltip="Characteristic size source: lt = local thickness, dt = distance transform"),
        F("method", "choice_str", "median", ["mean", "median", "min", "max"],
          tooltip="Aggregation method applied to the characteristic size values")]),
    ("CHORD_COUNTS", met.CHORD_COUNTS, [
        F("spacing", "int", "1", tooltip="Voxels between parallel chord lines"),
        F("trim_edges", "bool", True, tooltip="Remove chords touching domain boundary"),
        F("apply_chords3D", "bool", True, tooltip="Use 3D chords; if False, uses 1D chords along axis"),
        F("axis", "choice_int", "0", AXIS3, tooltip="Chord axis when apply_chords3D is False"),
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("CHORD_COUNTS_SOLID_PHASE", met.CHORD_COUNTS_SOLID_PHASE, [
        F("spacing", "int", "1", tooltip="Voxels between parallel chord lines"),
        F("trim_edges", "bool", True, tooltip="Remove chords touching domain boundary"),
        F("apply_chords3D", "bool", True, tooltip="Use 3D chords; if False, uses 1D chords along axis"),
        F("axis", "choice_int", "0", AXIS3, tooltip="Chord axis when apply_chords3D is False"),
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("CHORD_LENGTH_DISTRIBUTION", met.CHORD_LENGTH_DISTRIBUTION, [
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("log", "bool", False, tooltip=_TT_LOG),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("normalization", "choice_str", "number", ["count", "number", "length"],
          tooltip="Bin height normalization: count, number, or length"),
        F("apply_chords3D", "bool", True, tooltip="Use 3D chords; if False, uses 1D chords along axis"),
        F("axis", "choice_int", "0", AXIS3, tooltip="Chord axis when apply_chords3D is False"),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE", met.CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE, [
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("log", "bool", False, tooltip=_TT_LOG),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("normalization", "choice_str", "number", ["count", "number", "length"],
          tooltip="Bin height normalization: count, number, or length"),
        F("apply_chords3D", "bool", True, tooltip="Use 3D chords; if False, uses 1D chords along axis"),
        F("axis", "choice_int", "0", AXIS3, tooltip="Chord axis when apply_chords3D is False"),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("COORDINATION_NUMBER", met.COORDINATION_NUMBER, [
        F("voxel_size", "float", "1", tooltip="Physical length of each voxel (m) passed to snow2"),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("FIND_H", met.FIND_H, [
        F("srange", "list_float", "0.01,0.95",
          tooltip="Saturation range [min,max] to search over"),
        F("axis_of_intrusion", "choice_int", "0", AXIS3, tooltip="Axis along which fluid intrudes"),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("g", "float", "0", tooltip=_TT_G),
        F("s", "float", "0.5", tooltip="Target global saturation at which to find h")]),
    ("FIND_POROSITY_THRESHOLD", met.FIND_POROSITY_THRESHOLD, [
        F("axis", "choice_int", "0", AXIS3, tooltip="Percolation test axis"),
        F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN)]),
    ("IS_PERCOLATING", met.IS_PERCOLATING, [
        F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN)]),
    ("LINEAL_PATH_DISTRIBUTION", met.LINEAL_PATH_DISTRIBUTION, [
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("log", "bool", False, tooltip=_TT_LOG),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("MESH_SURFACE_AREA", met.MESH_SURFACE_AREA, [
        F("voxel_size", "float", "1", tooltip=_TT_VS)]),
    ("MESH_VOLUME", met.MESH_VOLUME, [
        F("voxel_size", "float", "1", tooltip=_TT_VS)]),
    ("PC_MAP_TO_PC_CURVE", met.PC_MAP_TO_PC_CURVE, [
        F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
        F("theta", "float", "180", tooltip=_TT_THETA),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("steps", "int", "25", tooltip="Number of capillary pressure levels to evaluate"),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("PERCOLATING_POROSITY", met.PERCOLATING_POROSITY, [
        F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN)]),
    ("PORE_SIZE_DISTRIBUTION", met.PORE_SIZE_DISTRIBUTION, [
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("log", "bool", False, tooltip=_TT_LOG),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("POROSITY", met.POROSITY, []),
    ("POROSITY_PROFILE", met.POROSITY_PROFILE, [
        F("voxel_size", "float", "1", tooltip="Physical length of each voxel (µm)"),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("RADIAL_DENSITY_DISTRIBUTION", met.RADIAL_DENSITY_DISTRIBUTION, [
        F("bins", "int", "25", tooltip=_TT_BINS),
        F("log", "bool", False, tooltip=_TT_LOG),
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("REGION_SURFACE_AREAS", met.REGION_SURFACE_AREAS, [
        F("voxel_size", "float", "1", tooltip=_TT_VS),
        F("strel_size", "int", "2",
          tooltip="Blurring element size; larger → smoother but less accurate surface"),
        F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("REGION_VOLUMES", met.REGION_VOLUMES, [
        F("method", "choice_str", "voxel", ["voxel", "marching_cubes"],
          tooltip="Volume method: voxel (fast) or marching_cubes (more accurate)"),
        F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("REGIONPROPS_3D", met.REGIONPROPS_3D, [
        F("sphericity", "bool", False, tooltip="Compute and output per-region sphericity"),
        F("solidity", "bool", False, tooltip="Compute and output per-region solidity"),
        F("num_pixels", "bool", False, tooltip="Include voxel count per region"),
        F("min_size", "int", "4",
          tooltip="Drop regions smaller than this many voxels before building "
                  "the table. Below 4 voxels a region has no valid 3D convex "
                  "hull, which crashes feret_diameter_max. Such specks are "
                  "usually segmentation noise")]),
    ("REV_POROSITY", met.REV_POROSITY, [
        F("n", "int", "1000", tooltip="Number of random sampling boxes")]),
    ("SATN_PROFILE", met.SATN_PROFILE, [
        F("span", "int", "2", tooltip="Width of the averaging window (voxels)"),
        F("mode", "choice_str", "tile", ["tile", "slide"],
          tooltip="tile = non-overlapping windows; slide = overlapping windows"),
        F("intr_axis", "choice_int", "0", AXIS3, tooltip="Axis along which fluid intrudes"),
        F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
        F("theta", "float", "180", tooltip=_TT_THETA),
        F("g", "float", "0", tooltip=_TT_G),
        F("s_min", "float", "0.1",
          tooltip="Lowest saturation threshold to profile. Must not exceed "
                  "the maximum saturation reached in the image"),
        F("s_max", "float", "0.9",
          tooltip="Highest saturation threshold to profile; values above "
                  "the image's maximum saturation are skipped"),
        F("s_step", "float", "0.2",
          tooltip="Spacing between saturation thresholds (one curve each)"),
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
    ("TWO_POINT_CORRELATION", met.TWO_POINT_CORRELATION, [
        F("save_excel", "bool", False, tooltip=_TT_XL)]),
]

_TT_RHO_NWP = "Non-wetting phase density (kg/m³)"
_TT_RHO_WP = "Wetting phase density (kg/m³)"
_TT_VS_SIM = "Physical length of each voxel (m)"
_TT_AXIS_SIM = "Inlet/outlet axis; None → all faces"
_TT_MIN_SIZE = "Minimum trapped cluster size to retain (voxels)"

_DRAIN_IMB_FIELDS = [
    F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
    F("theta", "float", "180", tooltip=_TT_THETA),
    F("g", "float", "9.81", tooltip=_TT_G),
    F("rho_nwp", "float", "1000", tooltip=_TT_RHO_NWP),
    F("rho_wp", "float", "0", tooltip=_TT_RHO_WP),
    F("voxel_size", "float", "1e-4", tooltip=_TT_VS_SIM),
    F("axis", "choice_int_none", "None", AXIS_NONE, tooltip=_TT_AXIS_SIM),
    F("min_size", "int", "5", tooltip=_TT_MIN_SIZE),
    F("steps", "int_none", "50",
      tooltip="Number of pressure steps. WARNING: empty → one pass per "
              "unique capillary pressure value; with gravity (g>0) almost "
              "every voxel is unique, so the run takes hours instead of "
              "minutes. Keep ~25–100 unless you really need every step"),
    F("save_excel", "bool", False, tooltip="Export capillary pressure–saturation curve to Excel"),
]
PSIM_SPECS = [
    ("DRAINAGE", psim.DRAINAGE, list(_DRAIN_IMB_FIELDS)),
    ("IMBIBITION", psim.IMBIBITION, list(_DRAIN_IMB_FIELDS)),
    ("INJECTION", psim.INJECTION, [
        F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
        F("theta", "float", "180", tooltip=_TT_THETA),
        F("g", "float", "9.81", tooltip=_TT_G),
        F("rho_nwp", "float", "1000", tooltip=_TT_RHO_NWP),
        F("rho_wp", "float", "0", tooltip=_TT_RHO_WP),
        F("voxel_size", "float", "1e-4", tooltip=_TT_VS_SIM),
        F("axis", "choice_int_none", "None", AXIS_NONE, tooltip=_TT_AXIS_SIM),
        F("min_size", "int", "5", tooltip=_TT_MIN_SIZE),
        F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
        F("maxiter", "int", "10000", tooltip="Maximum invasion iterations"),
        F("method", "choice_str", "qbip", ["qbip", "ibip"],
          tooltip="qbip = faster (binary heap); ibip = image-based"),
        F("save_excel", "bool", False, tooltip="Export capillary pressure–saturation curve to Excel")]),
    ("TORTUOSITY", psim.TORTUOSITY, [
        F("axis", "choice_int", "0", AXIS3, tooltip="Axis along which tortuosity is calculated"),
        F("crop_size", "int_none", "", label="Crop size (opt.)",
          tooltip="Crop the centre cube to this side length before solving. "
                  "tortuosity_fd needs ~22 GB for a 1000³ image; use ≤300 for large volumes.")]),
]

# OpenPNM phase property forms (only the properties relevant to each phase).
PNM_PHASE_SPECS = [
    ("water", [
        F("diffusivity", "float", "2.3e-9", tooltip="Molecular diffusivity (m²/s)"),
        F("viscosity", "float", "1.0e-3", tooltip="Dynamic viscosity (Pa·s)"),
        F("contact_angle", "float", "110.0", tooltip="Contact angle (degrees)")]),
    ("air", [
        F("diffusivity", "float", "2.5e-5", tooltip="Molecular diffusivity (m²/s)"),
        F("viscosity", "float", "1.85e-5", tooltip="Dynamic viscosity (Pa·s)"),
        F("contact_angle", "float", "120.0", tooltip="Contact angle (degrees)"),
        F("surface_tension", "float", "0.072", tooltip="Surface tension (N/m)")]),
    ("mercury", [
        F("diffusivity", "float", "1.7e-9", tooltip="Molecular diffusivity (m²/s)"),
        F("viscosity", "float", "1.53e-3", tooltip="Dynamic viscosity (Pa·s)"),
        F("contact_angle", "float", "140.0", tooltip="Contact angle (degrees)"),
        F("surface_tension", "float", "0.485", tooltip="Surface tension (N/m)")]),
]

# OpenPNM simulations: only the requested subset of parameters per function.
PNM_SIM_SPECS = [
    ("Stokes flow", opnm.simulate_stokes_flow, [
        F("axis", "choice_str", "x", XYZ, tooltip="Axis along which pressure drop is applied"),
        F("inlet_pressure", "float", "1.0", tooltip="Inlet pressure boundary condition"),
        F("outlet_pressure", "float", "0.0", tooltip="Outlet pressure boundary condition")]),
    ("Fickian diffusion", opnm.simulate_fickian_diffusion, [
        F("axis", "choice_str", "x", XYZ, tooltip="Axis along which concentration gradient is applied"),
        F("inlet_concentration", "float", "1.0", tooltip="Inlet concentration (mol/m³ for mol/s rate; dimensionless 0–1 gives m³/s rate)"),
        F("outlet_concentration", "float", "0.0", tooltip="Outlet concentration (same units as inlet)")]),
    ("Advection-diffusion", opnm.simulate_advection_diffusion, [
        F("axis", "choice_str", "x", XYZ, tooltip="Axis along which flow and diffusion occur"),
        F("inlet_pressure", "float", "1.0", tooltip="Inlet pressure boundary condition"),
        F("outlet_pressure", "float", "0.0", tooltip="Outlet pressure boundary condition"),
        F("inlet_concentration", "float", "1.0", tooltip="Inlet concentration (mol/m³ for mol/s rate; dimensionless 0–1 gives m³/s rate)"),
        F("outlet_concentration", "float", "0.0", tooltip="Outlet concentration (same units as inlet)")]),
    ("Drainage", opnm.simulate_drainage, [
        F("axis", "choice_str", "x", XYZ_ALL, tooltip="Invasion axis; 'all' = all boundary faces simultaneously"),
        F("save_excel", "bool", False, tooltip="Export capillary pressure–saturation curve to Excel")]),
    ("Invasion percolation", opnm.simulate_invasion_percolation, [
        F("axis", "choice_str", "x", XYZ_ALL, tooltip="Invasion axis; 'all' = all boundary faces simultaneously"),
        F("save_excel", "bool", False, tooltip="Export capillary pressure–saturation curve to Excel")]),
]

# Short tags used to build the single combined VTK filename, e.g. openpnm_SF_FD_AD_Dr_IP.vtp
_PNM_SIM_ABBR = {
    "Stokes flow": "SF",
    "Fickian diffusion": "FD",
    "Advection-diffusion": "AD",
    "Drainage": "Dr",
    "Invasion percolation": "IP",
}


# ===========================================================================
#  Application
# ===========================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.image = None             # current working image (may be SA-filtered)
        self.image_unfiltered = None  # image as it left the generator/import step
        self.image_name = None        # base name used for filter output filenames
        self.pn = None                # cached OpenPNM network (built lazily)
        self.pn_image_id = None       # id() of the image the network was built from
        self.pn_voxel_size = None     # voxel_size used to build the cached network

    # ---- helpers ---------------------------------------------------------
    def require_image(self, parent):
        if self.image is None:
            messagebox.showwarning("No image",
                                   "No image is available yet. Generate or import one first.",
                                   parent=parent)
            return False
        return True

    # ---- Main window -----------------------------------------------------
    def build_main(self):
        self.root.title("Porous Media Analysis")
        self.root.geometry("420x260")
        tk.Label(self.root, text="Porous Media Analysis",
                 font=("Segoe UI", 16, "bold")).pack(pady=(28, 6))
        tk.Label(self.root, text="Choose how to obtain the 3D image to analyse.",
                 fg="#555").pack(pady=(0, 20))
        tk.Button(self.root, text="Generate", width=20, height=2,
                  command=self.open_generators).pack(pady=6)
        tk.Button(self.root, text="Import", width=20, height=2,
                  command=self.run_import).pack(pady=6)

    # ---- Import flow -----------------------------------------------------
    def run_import(self):
        chooser = tk.Toplevel(self.root)
        chooser.title("Import image")
        chooser.geometry("380x220")
        chooser.transient(self.root)
        tk.Label(chooser, text="Select the image source",
                 font=("Segoe UI", 12, "bold")).pack(pady=(18, 4))
        tk.Label(chooser, text="A single 3D .tif, .raw, or .dcm file, or a folder of "
                               "2D .tif or .dcm slices.",
                 fg="#555", wraplength=340).pack(pady=(0, 12))

        def pick_file():
            path = filedialog.askopenfilename(
                parent=chooser, title="Select a 3D image file",
                filetypes=[("3D image", "*.tif *.tiff *.raw *.dcm"),
                            ("TIFF image", "*.tif *.tiff"),
                            ("Raw binary", "*.raw"),
                            ("DICOM", "*.dcm"),
                            ("All files", "*.*")])
            if path:
                chooser.destroy()
                self._do_import(path, is_folder=False)

        def pick_folder():
            path = filedialog.askdirectory(parent=chooser,
                                           title="Select a folder of 2D slices (.tif or .dcm)")
            if path:
                chooser.destroy()
                self._do_import(path, is_folder=True)

        tk.Button(chooser, text="Select 3D image file (.tif / .raw / .dcm)", width=34,
                  command=pick_file).pack(pady=4)
        tk.Button(chooser, text="Select folder of 2D slices (.tif / .dcm)", width=34,
                  command=pick_folder).pack(pady=4)

    def _do_import(self, path, is_folder):
        import imageio.v2 as imageio
        from skimage.io import imread_collection

        # --- .raw: ask for shape/dtype before loading -------------------
        raw_shape, raw_dtype = None, None
        if not is_folder and os.path.splitext(path)[1].lower() == ".raw":
            raw_shape, raw_dtype = _ask_raw_params(self.root, os.path.getsize(path))
            if raw_shape is None:
                return  # user cancelled

        is_dcm_file = not is_folder and os.path.splitext(path)[1].lower() == ".dcm"
        dcm_voxel_size = None  # (dx, dy, dz) in mm, filled in by _load() for DICOM sources

        # --- load -------------------------------------------------------
        def _load():
            nonlocal dcm_voxel_size
            if is_folder:
                dcm_files = sorted(
                    f for f in os.listdir(path) if f.lower().endswith(".dcm"))
                if dcm_files:
                    arr, dcm_voxel_size = _read_dicom_series(
                        [os.path.join(path, f) for f in dcm_files])
                    print(f"Loaded {len(dcm_files)} DICOM slices: shape={arr.shape}")
                else:
                    seq = imread_collection(os.path.join(path, "*.tif"))
                    if len(seq) == 0:
                        raise ValueError(
                            "No .tif or .dcm slices found in the selected folder.")
                    # slices are (rows, cols) = (y, x); stack to z then swap to (x, y, z)
                    arr = np.transpose(np.stack(seq, axis=-1), (1, 0, 2))
                    print(f"Loaded {len(seq)} slices via imread_collection: shape={arr.shape}")
            elif is_dcm_file:
                arr, dcm_voxel_size = _read_dicom_file(path)
                print(f"Loaded DICOM file: shape={arr.shape}")
            elif raw_shape is not None:
                arr = np.fromfile(path, dtype=np.dtype(raw_dtype)).reshape(raw_shape)
                arr = np.transpose(arr, (2, 1, 0))   # (slices,H,W)=(z,y,x) → (x,y,z)
                print(f"Loaded .raw image: shape={arr.shape}, dtype={raw_dtype}")
            else:
                arr = imageio.volread(path)
                arr = np.transpose(arr, (2, 1, 0))
                print(f"Loaded 3D image via volread: shape={arr.shape}")
            return arr

        ok, im3d, out, err, _ = run_operation(self.root, "Loading image", _load, {})
        if not ok:
            messagebox.showerror("Import failed", err, parent=self.root)
            return

        if dcm_voxel_size is not None:
            dx, dy, dz = dcm_voxel_size
            messagebox.showinfo(
                "DICOM voxel spacing",
                f"Voxel size from DICOM header: {dx:.5g} x {dy:.5g} x {dz:.5g} mm.\n"
                "Use this value (in metres) for 'voxel_size' fields in Metrics / "
                "Simulations for physically correct results.",
                parent=self.root)

        base_name = (os.path.basename(os.path.normpath(path))
                     if is_folder
                     else os.path.splitext(os.path.basename(path))[0])
        is_tif_source = (not is_folder and
                         os.path.splitext(path)[1].lower() in ('.tif', '.tiff'))

        unique = np.unique(im3d)
        already_binary = len(unique) <= 2
        user_inverted = False

        if already_binary:
            im_binary = im3d.astype(bool)
            poro_as_is  = float(im_binary.sum())  / im_binary.size
            poro_invert = float((~im_binary).sum()) / im_binary.size
            messagebox.showinfo("Import",
                                f"Image is already binary (values {unique}).\n"
                                "No thresholding needed.\n\n"
                                f"As-is  → porosity = {poro_as_is:.3f}  (True = pore)\n"
                                f"Inverted → porosity = {poro_invert:.3f}  (True = pore)\n\n"
                                "PoreSpy treats True voxels as pore/void.\n"
                                "In most CT scans bright = solid, dark = pore,\n"
                                "so inversion is usually needed.",
                                parent=self.root)
            if messagebox.askyesno("Invert image",
                                   "Do you want to invert the image?\n"
                                   "(swaps pore and solid phases)\n\n"
                                   "Say Yes if pores are dark (low values) in your scan.",
                                   parent=self.root):
                im_binary = ~im_binary
                user_inverted = True
        else:
            # --- denoise (slow) -----------------------------------------
            im_float = im3d.astype(np.float64)
            im_float = (im_float - im_float.min()) / (im_float.max() - im_float.min())

            def _denoise():
                import porespy as ps
                # porespy's nl_means_layered has a bug: it loops range(im.shape[0])
                # instead of range(swapped_shape[0]), which only happens to work for
                # cubic volumes. Pre-move the slice axis to position 0 and call with
                # the default axis=0 (swapaxes(im,0,0) is a no-op there) to sidestep it.
                im_axis0 = np.moveaxis(im_float, 2, 0)
                d = ps.filters.nl_means_layered(im_axis0, axis=0)
                d = np.moveaxis(d, 0, 2)
                return (d - d.min()) / (d.max() - d.min())

            ok, denoised, out, err, _ = run_operation(self.root, "Denoising (NL-means)", _denoise, {})
            if not ok:
                messagebox.showerror("Import failed", err, parent=self.root)
                return

            # histogram to help choose the threshold
            fig, ax = plt.subplots(figsize=[6, 4])
            ax.hist(denoised.flatten(), bins=50, edgecolor="k")
            ax.set_xlabel("Voxel intensity")
            ax.set_ylabel("Count")
            ax.set_title("Histogram of denoised image (close to continue)")
            fig.tight_layout()
            plt.show()

            threshold = simpledialog.askfloat(
                "Threshold", "Enter threshold value for segmentation [0.0 - 1.0]:",
                parent=self.root, minvalue=0.0, maxvalue=1.0, initialvalue=0.5)
            if threshold is None:
                threshold = 0.5
            im_binary = denoised < threshold  # True = pore (void)

            if messagebox.askyesno("Invert image",
                                   "Do you want to invert the image?\n"
                                   "(swaps pore and solid phases)",
                                   parent=self.root):
                im_binary = ~im_binary
                user_inverted = True

            mid = im3d.shape[2] // 2
            fig, ax = plt.subplots(1, 3, figsize=[15, 5])
            ax[0].imshow(im3d[:, :, mid], cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
            ax[1].imshow(denoised[:, :, mid], cmap="gray"); ax[1].set_title("Denoised"); ax[1].axis("off")
            ax[2].imshow(~im_binary[:, :, mid], cmap="gray"); ax[2].set_title("Binary"); ax[2].axis("off")
            fig.tight_layout()
            plt.show(block=False)

        # Save the result unless the source was already a binary .tif and
        # the user did not invert (the original file already serves as the .tif).
        skip_save = is_tif_source and already_binary and not user_inverted
        if not skip_save:
            if already_binary and user_inverted:
                default_name = f"{base_name}_inverted.tif"
            elif user_inverted:
                default_name = f"{base_name}_binary_inverted.tif"
            else:
                default_name = f"{base_name}_binary.tif"
            out_dir, chosen_name = ask_save_file(self.root, default_name)
            if out_dir:
                save_name = chosen_name or default_name
                try:
                    arr = np.transpose(im_binary.astype("uint8") * 255, (2, 1, 0))
                    imageio.volsave(os.path.join(out_dir, save_name), arr)
                    messagebox.showinfo("Saved",
                                        f"Binary image saved as:\n"
                                        f"{os.path.join(out_dir, save_name)}",
                                        parent=self.root)
                except Exception as e:  # noqa: BLE001
                    messagebox.showwarning("Save warning",
                                           f"Could not save image:\n{e}", parent=self.root)

        self.image_name = base_name
        self.image = im_binary
        self.image_unfiltered = im_binary
        messagebox.showinfo("Import complete",
                            "Image imported. Continuing to the Filters step.",
                            parent=self.root)
        self.open_filters()

    # ---- Generators window ----------------------------------------------
    def open_generators(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "Generators",
            "Tick exactly one generator, set its parameters, then click Generate. "
            "The generated image becomes the working image for the next steps.")
        single = SingleSelect()
        sections = []
        for title, func, fields in GEN_SPECS:
            sec = build_section(body, title, fields, mode="single", single=single)
            sec["func"] = func
            sections.append(sec)
        grid_sections(body, sections, 2)

        run_time_var = tk.StringVar(value="Run time: —")

        def do_generate():
            chosen = [s for s in sections if section_is_selected(s)]
            if len(chosen) != 1:
                messagebox.showwarning("Selection", "Tick exactly one generator.", parent=win)
                return
            sec = chosen[0]
            try:
                kwargs = collect_kwargs(sec["ff"])
            except ValueError as e:
                messagebox.showerror("Invalid input", str(e), parent=win)
                return
            save_file = get_save_file(sec)
            if save_file:
                out_dir, chosen_name = ask_save_file(win, save_file)
            else:
                out_dir, chosen_name = None, None
            # out_dir is None when the user cancels — generation still proceeds
            ok, result, out, err, elapsed = run_operation(win, f"Generating {sec['title']}",
                                                           sec["func"], kwargs, output_dir=out_dir)
            run_time_var.set(f"Run time: {elapsed:.1f} s")
            if ok:
                _rename_saved(out_dir, save_file, chosen_name)
                self.image = result
                self.image_unfiltered = result
                self.image_name = sec['title'].capitalize()
                self.pn = None  # invalidate any cached network
                messagebox.showinfo("Done",
                                    f"{sec['title']} generated. Click Next to filter the image.",
                                    parent=win)
                for s in sections:
                    s["var"].set(False)
            else:
                messagebox.showerror("Generation failed", err, parent=win)

        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=win.destroy).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Generate", width=14, bg="#90EE90",
                  command=do_generate).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_filters(win)).grid(row=0, column=2, sticky="e", padx=5)
        tk.Label(bar, textvariable=run_time_var, fg="#555",
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=0, columnspan=3, pady=(4, 0))

    # ---- Filters window --------------------------------------------------
    def open_filters(self, prev=None):
        if prev:
            prev.destroy()
        # Always start from the generated/imported image, never from a
        # previously SA-filtered result (handles Back navigation correctly).
        _original_image = self.image_unfiltered if self.image_unfiltered is not None else self.image

        n_solid_filters = len(FILTER_GROUPS[0][1])
        win, body, bar = make_window(
            self, "Filters",
            "Solid altering filters chain together in the order you set (1, 2, 3, ...) "
            "and always run first. "
            "Non-solid altering filters are simply ticked; each independently receives "
            "the solid-altering result (or the original image if none were selected). "
            "Tick 'Save file' on any non-solid filter to save its output as a separate .tif.")
        _image_size_label(body, _original_image)
        _n_voxels = int(np.prod(_original_image.shape)) if _original_image is not None else 1_000_000
        order_values = [""] + [str(i) for i in range(1, n_solid_filters + 1)]
        sections = []
        solid_sections = []
        nonsolid_sections = []
        for group_idx, (group_label, specs) in enumerate(FILTER_GROUPS):
            is_nonsolid = (group_idx == 1)
            tk.Label(body, text=group_label,
                     font=("Segoe UI", 11, "bold"), fg="#1a1a1a",
                     anchor="w").pack(fill="x", padx=10, pady=(10, 2))
            ttk.Separator(body, orient="horizontal").pack(fill="x", padx=10, pady=(0, 4))
            group_container = tk.Frame(body)
            group_container.pack(fill="x")
            group_secs = []
            for title, func, fields in specs:
                # SA filters keep the run-order combobox; NSA filters are
                # plain checkboxes (they run after the SA chain, unordered).
                sec = build_section(group_container, title, fields,
                                    mode="multi" if is_nonsolid else "order",
                                    order_values=order_values,
                                    est_time=_est_time(title, _n_voxels))
                sec["func"] = func
                if is_nonsolid:
                    save_var = tk.BooleanVar(value=False)
                    save_row = tk.Frame(sec["frame"])
                    save_row.pack(fill="x", padx=4, pady=(0, 2))
                    tk.Checkbutton(save_row, text="Save file", variable=save_var).pack(side="left")
                    sec["save_var"] = save_var
                    nonsolid_sections.append(sec)
                else:
                    sec["save_var"] = None
                    solid_sections.append(sec)
                group_secs.append(sec)
                sections.append(sec)
            grid_sections(group_container, group_secs, 2)

        # When an order number is set on a filter, auto-clear any other filter
        # that already holds the same number (last-set wins).
        def _make_conflict_resolver(this_sec, all_secs):
            def _resolve(*_):
                val = this_sec["var"].get()
                if not val:
                    return
                for other in all_secs:
                    if other is not this_sec and other["var"].get() == val:
                        other["var"].set("")
            return _resolve

        for _sec in solid_sections:
            _sec["var"].trace_add("write",
                                  _make_conflict_resolver(_sec, solid_sections))

        def do_filter():
            if not self.require_image(win):
                return
            solid_chosen = sorted(
                [(section_order(s), s) for s in solid_sections if section_order(s) is not None],
                key=lambda t: t[0])
            nonsolid_chosen = [s for s in nonsolid_sections
                               if section_is_selected(s)]

            if not solid_chosen and not nonsolid_chosen:
                messagebox.showinfo("No filters",
                                    "Nothing selected. The current image will be passed on unchanged.",
                                    parent=win)
                return

            import imageio.v2 as _iio

            # Always start from the image captured when this window opened,
            # so repeated Filter presses are independent of each other.
            base = _original_image
            _total_elapsed = 0.0
            for order, sec in solid_chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    messagebox.showerror("Invalid input", f"{sec['title']}: {e}", parent=win)
                    return
                kwargs["im"] = base
                ok, result, out, err, _t = run_operation(
                    win, f"Filtering: {sec['title']} (step {order})", sec["func"], kwargs)
                _total_elapsed += _t
                if not ok:
                    messagebox.showerror("Filter failed", f"{sec['title']}:\n{err}", parent=win)
                    return
                base = result
            # base is now the solid-altering result (or unchanged original if no SA filters)

            # Determine the output directory
            nsa_to_save = [s for s in nonsolid_chosen
                           if s["save_var"] and s["save_var"].get()]
            out_dir = None

            if solid_chosen:
                # Save As dialog for the solid-altering result
                img_base = self.image_name or "image"
                sa_suffix = "_".join(s["title"].lower() for _, s in solid_chosen)
                default_name = f"{img_base}_{sa_suffix}.tif"
                out_dir, chosen_name = ask_save_file(win, default_name)
                if out_dir is None:
                    return
                save_name = chosen_name or default_name
                try:
                    arr = base.astype(np.uint8) * 255 if base.dtype == bool else base.astype(np.float32)
                    _iio.volsave(os.path.join(out_dir, save_name),
                                 np.transpose(arr, (2, 1, 0)))
                except Exception as _e:
                    messagebox.showwarning("Save warning",
                                           f"Could not save {save_name}:\n{_e}", parent=win)
            elif nsa_to_save:
                # No SA filters but some NSA files need saving — ask for directory
                out_dir = filedialog.askdirectory(
                    parent=win, title="Choose a folder for the filter output files")
                if not out_dir:
                    return

            # Run non-solid-altering filters independently; each receives `base` as input
            for sec in nonsolid_chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    messagebox.showerror("Invalid input", f"{sec['title']}: {e}", parent=win)
                    continue
                kwargs["im"] = base
                ok, result, out, err, _t = run_operation(
                    win, f"Filtering: {sec['title']}", sec["func"], kwargs)
                _total_elapsed += _t
                if not ok:
                    messagebox.showerror("Filter failed", f"{sec['title']}:\n{err}", parent=win)
                    continue
                if sec["save_var"] and sec["save_var"].get() and out_dir is not None:
                    nsa_name = f"{self.image_name or 'image'}_{sec['title'].lower()}.tif"
                    try:
                        arr = (result.astype(np.uint8) * 255 if result.dtype == bool
                               else result.astype(np.float32))
                        _iio.volsave(os.path.join(out_dir, nsa_name),
                                     np.transpose(arr, (2, 1, 0)))
                    except Exception as _e:
                        messagebox.showwarning("Save warning",
                                               f"Could not save {nsa_name}:\n{_e}", parent=win)

            # self.image is the solid-altering result only (or original if no SA filters)
            self.image = base
            self.pn = None
            run_time_var.set(f"Run time: {_total_elapsed:.1f} s")
            messagebox.showinfo("Done", "Filtering complete. Click Next for metrics.", parent=win)
            for s in solid_sections:
                s["var"].set("")
            for s in nonsolid_sections:
                s["var"].set(False)

        run_time_var = tk.StringVar(value="Run time: —")
        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_generators(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Filter", width=14, bg="#90EE90",
                  command=do_filter).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_metrics(win)).grid(row=0, column=2, sticky="e", padx=5)
        tk.Label(bar, textvariable=run_time_var, fg="#555",
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=0, columnspan=3, pady=(4, 0))

    # ---- Metrics window --------------------------------------------------
    def open_metrics(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "Metrics",
            "Tick any number of metrics, set their parameters, then click Calculate. "
            "Each section shows its printed result (green) or error (red); some also "
            "produce plots or Excel files.")
        _image_size_label(body, self.image)
        _n_voxels = int(np.prod(self.image.shape)) if self.image is not None else 1_000_000
        section_frame = tk.Frame(body)
        section_frame.pack(fill="both", expand=True)
        sections = []
        for title, func, fields in METRIC_SPECS:
            sec = build_section(section_frame, title, fields, mode="multi", with_result=True,
                                est_time=_est_time(title, _n_voxels))
            sec["func"] = func
            sections.append(sec)
        grid_sections(section_frame, sections, 2)

        def do_calculate():
            if not self.require_image(win):
                return
            chosen = [s for s in sections if section_is_selected(s)]
            if not chosen:
                messagebox.showwarning("Selection", "Tick at least one metric.", parent=win)
                return
            saving = [(s, get_save_file(s)) for s in chosen if get_save_file(s) is not None]
            if not saving:
                out_dir, name_map = None, {}
            elif len(saving) == 1:
                out_dir, chosen_name = ask_save_file(win, saving[0][1])
                if out_dir is None:
                    return
                name_map = {saving[0][1]: chosen_name}
            else:
                out_dir = filedialog.askdirectory(parent=win,
                                                  title="Choose a folder for the output files")
                if not out_dir:
                    return
                name_map = {f: f for _, f in saving}
            _total_elapsed = 0.0
            for sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    sec["result"].show(str(e), ok=False)
                    continue
                kwargs["im"] = self.image
                ok, result, out, err, _t = run_operation(
                    win, f"Calculating {sec['title']}", sec["func"], kwargs, output_dir=out_dir)
                _total_elapsed += _t
                if ok:
                    sec["result"].show(out or "Done (see plots / saved files).", ok=True)
                    sf = get_save_file(sec)
                    if sf and out_dir and sf in name_map:
                        _rename_saved(out_dir, sf, name_map[sf])
                else:
                    sec["result"].show(err, ok=False)
            run_time_var.set(f"Run time: {_total_elapsed:.1f} s")
            for s in sections:
                s["var"].set(False)

        run_time_var = tk.StringVar(value="Run time: —")
        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_filters(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Calculate", width=14, bg="#90EE90",
                  command=do_calculate).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_porespy(win)).grid(row=0, column=2, sticky="e", padx=5)
        tk.Label(bar, textvariable=run_time_var, fg="#555",
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=0, columnspan=3, pady=(4, 0))

    # ---- PoreSpy simulations window -------------------------------------
    def open_porespy(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "PoreSpy simulations",
            "Tick any number of simulations, set their parameters, then click Simulate. "
            "Printed output (e.g. the tortuosity value) is shown in each section's box.")
        _image_size_label(body, self.image)
        _n_voxels = int(np.prod(self.image.shape)) if self.image is not None else 1_000_000
        section_frame = tk.Frame(body)
        section_frame.pack(fill="both", expand=True)
        sections = []
        for title, func, fields in PSIM_SPECS:
            sec = build_section(section_frame, title, fields, mode="multi", with_result=True,
                                est_time=_est_time(title, _n_voxels))
            sec["func"] = func
            sections.append(sec)
        grid_sections(section_frame, sections, 2)

        def do_simulate():
            if not self.require_image(win):
                return
            chosen = [s for s in sections if section_is_selected(s)]
            if not chosen:
                messagebox.showwarning("Selection", "Tick at least one simulation.", parent=win)
                return
            saving = [(s, get_save_file(s)) for s in chosen if get_save_file(s) is not None]
            if not saving:
                out_dir, name_map = None, {}
            elif len(saving) == 1:
                out_dir, chosen_name = ask_save_file(win, saving[0][1])
                if out_dir is None:
                    return
                name_map = {saving[0][1]: chosen_name}
            else:
                out_dir = filedialog.askdirectory(parent=win,
                                                  title="Choose a folder for the output files")
                if not out_dir:
                    return
                name_map = {f: f for _, f in saving}
            _total_elapsed = 0.0
            for sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    sec["result"].show(str(e), ok=False)
                    continue
                kwargs["im"] = self.image
                ok, result, out, err, _t = run_operation(
                    win, f"Simulating {sec['title']}", sec["func"], kwargs, output_dir=out_dir)
                _total_elapsed += _t
                if ok:
                    sec["result"].show(out or "Done (see plots / saved files).", ok=True)
                    sf = get_save_file(sec)
                    if sf and out_dir and sf in name_map:
                        _rename_saved(out_dir, sf, name_map[sf])
                else:
                    sec["result"].show(err, ok=False)
            run_time_var.set(f"Run time: {_total_elapsed:.1f} s")
            for s in sections:
                s["var"].set(False)

        run_time_var = tk.StringVar(value="Run time: —")
        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_metrics(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Simulate", width=14, bg="#90EE90",
                  command=do_simulate).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_openpnm(win)).grid(row=0, column=2, sticky="e", padx=5)
        tk.Label(bar, textvariable=run_time_var, fg="#555",
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=0, columnspan=3, pady=(4, 0))

    # ---- OpenPNM simulations window -------------------------------------
    def open_openpnm(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "OpenPNM simulations",
            "Top: choose exactly one phase and edit its properties. "
            "Bottom: tick any number of simulations and set their parameters. "
            "Click Simulate to build the pore network from the current image and run "
            "the ticked simulations with the chosen phase.")
        _image_size_label(body, self.image)
        _n_voxels = int(np.prod(self.image.shape)) if self.image is not None else 1_000_000

        # Voxel size — used by snow2 to set network geometry in physical units
        vs_frame = tk.Frame(body)
        vs_frame.pack(fill="x", padx=12, pady=(4, 0))
        vs_lbl = tk.Label(vs_frame, text="Voxel size (m):")
        vs_lbl.pack(side="left")
        Tooltip(vs_lbl, "Physical length of each voxel (m) — passed to snow2 for network geometry")
        vs_var = tk.StringVar(value="1")
        tk.Entry(vs_frame, textvariable=vs_var, width=14).pack(side="left", padx=6)

        tk.Label(body, text="Phase selection", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10, pady=(6, 0))
        phase_container = tk.Frame(body)
        phase_container.pack(fill="x")
        phase_single = SingleSelect()
        phase_secs = {}
        for name, fields in PNM_PHASE_SPECS:
            sec = build_section(phase_container, name.capitalize(), fields,
                                mode="single", single=phase_single, est_time=None)
            phase_secs[name] = sec
        grid_sections(phase_container, list(phase_secs.values()), 3)

        tk.Label(body, text="Simulations", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10, pady=(12, 0))
        sim_container = tk.Frame(body)
        sim_container.pack(fill="x")
        sim_secs = []
        for title, func, fields in PNM_SIM_SPECS:
            sec = build_section(sim_container, title, fields, mode="multi", with_result=True,
                                est_time=_est_time(title, _n_voxels))
            sec["func"] = func
            sim_secs.append(sec)
        grid_sections(sim_container, sim_secs, 2)

        def do_simulate():
            if not self.require_image(win):
                return
            chosen_phase = [n for n, s in phase_secs.items() if section_is_selected(s)]
            if len(chosen_phase) != 1:
                messagebox.showwarning("Phase", "Tick exactly one phase.", parent=win)
                return
            phase_name = chosen_phase[0]
            try:
                overrides = collect_kwargs(phase_secs[phase_name]["ff"])
            except ValueError as e:
                messagebox.showerror("Invalid phase input", str(e), parent=win)
                return

            chosen = [s for s in sim_secs if section_is_selected(s)]
            if not chosen:
                messagebox.showwarning("Selection", "Tick at least one simulation.", parent=win)
                return

            # All selected simulations share one OpenPNM project, so they're exported
            # together into a SINGLE VTK file whose name lists the chosen simulations
            # (e.g. openpnm_SF_FD_AD_Dr_IP.vtp). Ask for the name/location once.
            combined_default = ("openpnm_"
                                + "_".join(_PNM_SIM_ABBR[s["title"]] for s in chosen)
                                + ".vtp")
            out_dir, vtk_name = ask_save_file(win, combined_default)
            if out_dir is None:
                return

            # Parse voxel size
            try:
                voxel_size = float(vs_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid input",
                                     "Voxel size must be a number.", parent=win)
                return

            # Build (or reuse) the pore network; rebuild if image or voxel_size changed.
            _total_elapsed = 0.0
            if (self.pn is None
                    or self.pn_image_id != id(self.image)
                    or self.pn_voxel_size != voxel_size):
                ok, pn, out, err, _t = run_operation(
                    win, "Building pore network (snow2)", opnm.build_network,
                    {"image": self.image, "voxel_size": voxel_size},
                    output_dir=out_dir)
                _total_elapsed += _t
                if not ok:
                    messagebox.showerror("Network build failed", err, parent=win)
                    return
                self.pn = pn
                self.pn_image_id = id(self.image)
                self.pn_voxel_size = voxel_size

            n_ok = 0
            for sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    sec["result"].show(str(e), ok=False)
                    continue
                kwargs["network"] = self.pn
                kwargs["phase_overrides"] = overrides
                # Each simulation must NOT write its own VTK — the whole project is
                # exported once, below, into the single combined file.
                kwargs["save_vtk"] = False
                func = sec["func"]
                if func in (opnm.simulate_drainage, opnm.simulate_invasion_percolation):
                    kwargs["inv_phase"] = phase_name
                    kwargs["def_phase"] = "air" if phase_name == "water" else "water"
                else:
                    kwargs["chosen_phase"] = phase_name
                ok, result, out, err, _t = run_operation(
                    win, f"OpenPNM: {sec['title']}", func, kwargs, output_dir=out_dir)
                _total_elapsed += _t
                if ok:
                    sec["result"].show(out or "Done.", ok=True)
                    n_ok += 1
                else:
                    sec["result"].show(err, ok=False)

            # Export the whole project (all successful simulations) to one VTK file.
            if n_ok:
                stem = os.path.splitext(vtk_name)[0]
                ok, _res, out, err, _t = run_operation(
                    win, "Exporting combined VTK", opnm.export_vtk,
                    {"network": self.pn, "filename": stem}, output_dir=out_dir)
                _total_elapsed += _t
                if ok:
                    run_time_var.set(f"Run time: {_total_elapsed:.1f} s   |   Saved: {stem}.vtp")
                else:
                    messagebox.showerror("VTK export failed", err, parent=win)
                    run_time_var.set(f"Run time: {_total_elapsed:.1f} s")
            else:
                run_time_var.set(f"Run time: {_total_elapsed:.1f} s")
            for s in sim_secs:
                s["var"].set(False)
            for s in phase_secs.values():
                s["var"].set(False)

        run_time_var = tk.StringVar(value="Run time: —")
        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_porespy(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Simulate", width=14, bg="#90EE90",
                  command=do_simulate).grid(row=0, column=1)
        tk.Button(bar, text="Finish", width=12,
                  command=win.destroy).grid(row=0, column=2, sticky="e", padx=5)
        tk.Label(bar, textvariable=run_time_var, fg="#555",
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=0, columnspan=3, pady=(4, 0))


def main():
    os.chdir(SCRIPT_DIR)
    root = tk.Tk()
    app = App(root)
    app.build_main()
    root.mainloop()


if __name__ == "__main__":
    main()
