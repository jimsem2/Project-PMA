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

_ANSI_RE = re.compile(r'\x1b\[[\d;]*[A-Za-z]')

import matplotlib
matplotlib.use("TkAgg")          # native pop-up plot windows
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

import numpy as np               # noqa: E402

import tkinter as tk             # noqa: E402
from tkinter import ttk, filedialog, messagebox, simpledialog  # noqa: E402

# --- Project modules -------------------------------------------------------
# Importing these is cheap (Simulations3D_openPNM was refactored so its network
# is built lazily via build_network()).
import Generators_3D as gen          # noqa: E402
import Filters_3D as filt            # noqa: E402
import Metrics_3D as met             # noqa: E402
import Simulations3D_porespy as psim  # noqa: E402
import Simulations3D_openPNM as opnm  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
#  Form-field model + value parsing
# ===========================================================================
def F(name, ftype, default, choices=None, label=None):
    """Create a field description used to build a form widget."""
    return {"name": name, "type": ftype, "default": default,
            "choices": choices, "label": label or name}


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
        if t in ("int", "choice_int"):
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
        tk.Label(parent, text=f["label"] + ":").grid(row=i, column=0, sticky="w", padx=4, pady=2)
        t = f["type"]
        if t == "bool":
            var = tk.BooleanVar(value=bool(f["default"]))
            tk.Checkbutton(parent, variable=var).grid(row=i, column=1, sticky="w")
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
                  order_values=None, with_result=False, subtitle=None):
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

    tk.Label(header, text=title, font=("Segoe UI", 10, "bold")).pack(side="left", padx=8)
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
    met.CHORD_LENGTH_DISTRIBUTION: "chord_length_distribution.xlsx",
    met.LINEAL_PATH_DISTRIBUTION: "lineal_path_distribution.xlsx",
    met.PC_MAP_TO_PC_CURVE: "pc_map_to_pc_curve.xlsx",
    met.PORE_SIZE_DISTRIBUTION: "pore_size_distribution.xlsx",
    met.RADIAL_DENSITY_DISTRIBUTION: "radial_density_distribution.xlsx",
    met.REGION_SURFACE_AREAS: "region_surface_areas.xlsx",
    met.REGION_VOLUMES: "region_volumes.xlsx",
    met.REGIONPROPS_3D: "regionprops_3D.xlsx",
    psim.DRAINAGE: "drainage.vti",
    psim.IMBIBITION: "imbibition.vti",
    psim.INJECTION: "injection.vti",
    opnm.simulate_stokes_flow: "stokes_flow.vtp",
    opnm.simulate_fickian_diffusion: "fickian_diffusion.vtp",
    opnm.simulate_advection_diffusion: "advection_diffusion.vtp",
    opnm.simulate_drainage: "drainage.vtp",
    opnm.simulate_invasion_percolation: "invasion_percolation.vtp",
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


def run_operation(parent, title, func, kwargs, output_dir=None):
    """Run `func(**kwargs)` on the main thread behind a modal 'Processing' dialog.

    Captures stdout and temporarily switches the working directory to
    `output_dir` (so the module's hard-coded relative filenames are saved there).

    Returns (ok, result, stdout_text, error_text).
    """
    overlay = tk.Toplevel(parent)
    overlay.title("Please wait")
    overlay.transient(parent)
    overlay.resizable(False, False)
    tk.Label(overlay, text=f"⏳  {title}",
             font=("Segoe UI", 12, "bold"), padx=30).pack(pady=(18, 4))
    tk.Label(overlay, text="Processing... please wait.\n"
                           "Close any pop-up plot windows to continue.",
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
    old_cwd = os.getcwd()
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
        os.chdir(old_cwd)
        try:
            pb.stop()
            overlay.grab_release()
            overlay.destroy()
        except tk.TclError:
            pass
    return ok, result, _ANSI_RE.sub('', buf.getvalue()).strip(), err


def make_window(app, title, intro):
    """Create a step window with a title, scrollable body and a bottom button bar."""
    win = tk.Toplevel(app.root)
    win.title(title)
    win.geometry("960x780")
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

GEN_SPECS = [
    ("BLOBS", gen.BLOBS, [
        F("shape", "list_int", "100,100,100"),
        F("porosity", "float", "0.5"),
        F("blobiness", "int", "1"),
        F("invert", "bool", False)]),
    ("CYLINDERS", gen.CYLINDERS, [
        F("shape", "list_int", "100,100,100"),
        F("radius", "int", "2"),
        F("ncylinders", "int", "10"),
        F("porosity", "float", "0.5"),
        F("cyl_length", "int_none", ""),
        F("phi_max", "float", "90"),
        F("theta_max", "float", "90"),
        F("maxiter", "int", "10"),
        F("invert", "bool", False)]),
    ("OVERLAPPING_SPHERES", gen.OVERLAPPING_SPHERES, [
        F("shape", "list_int", "100,100,100"),
        F("radius", "int", "5"),
        F("porosity", "float", "0.5"),
        F("maxiter", "int", "10"),
        F("invert", "bool", False)]),
    ("POLYDISPERSE_SPHERES", gen.POLYDISPERSE_SPHERES, [
        F("shape", "list_int", "100,100,100"),
        F("porosity", "float", "0.5"),
        F("r_min", "int", "4"),
        F("nbins", "int", "3"),
        F("invert", "bool", False)]),
    ("PSEUDO_ELECTROSTATIC_PACKING", gen.PSEUDO_ELECTROSTATIC_PACKING, [
        F("shape", "list_int", "100,100,100"),
        F("r", "int", "3"),
        F("clearance", "int", "0"),
        F("protrusion", "int", "0"),
        F("maxiter", "int", "50"),
        F("phi", "float_none", "", label="phi (opt.)"),
        F("compactness", "float", "1"),
        F("invert", "bool", False)]),
    ("PSEUDO_GRAVITY_PACKING", gen.PSEUDO_GRAVITY_PACKING, [
        F("shape", "list_int", "100,100,100"),
        F("radius", "int", "3"),
        F("maxiter", "int", "50"),
        F("clearance", "int", "0"),
        F("pk_axis", "choice_int", "0", AXIS3),
        F("invert", "bool", False)]),
    ("VORONOI_EDGES", gen.VORONOI_EDGES, [
        F("shape", "list_int", "100,100,100"),
        F("thickness", "int", "2"),
        F("ncells", "int", "10"),
        F("flat_faces", "bool", True),
        F("invert", "bool", False)]),
]

FILTER_GROUPS = [
    ("Solid Altering Filters", [
        ("FILL_CLOSED_PORES", filt.FILL_CLOSED_PORES, [
            F("conn", "choice_int", "26", ["6", "26"]),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
        ("TRIM_DISCONNECTED_VOXELS", filt.TRIM_DISCONNECTED_VOXELS, [
            F("conn", "choice_str", "max", ["max", "min"]),
            F("inlet_faces", "choice_str", "all", ["all", "x0", "x1", "y0", "y1", "z0", "z1"]),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
        ("TRIM_FLOATING_SOLID", filt.TRIM_FLOATING_SOLID, [
            F("conn", "choice_str", "max", ["max", "min"]),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
        ("TRIM_NONPERCOLATING_PATHS", filt.TRIM_NONPERCOLATING_PATHS, [
            F("perc_axis", "choice_int", "0", AXIS3),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ]),
    ("Non-Solid Altering Filters", [
        ("APPLY_CHORDS", filt.APPLY_CHORDS, [
            F("spacing", "int", "1"), F("axis", "choice_int", "1", AXIS3),
            F("trim_edges", "bool", False), F("label", "bool", True),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
        ("APPLY_CHORDS_3D", filt.APPLY_CHORDS_3D, [
            F("spacing", "int", "2"), F("trim_edges", "bool", True),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
        ("FIND_DISCONNECTED_VOXELS", filt.FIND_DISCONNECTED_VOXELS, [
            F("conn", "choice_str", "max", ["max", "min"]),
            F("show_3d", "bool", True)]),
        ("LOCAL_THICKNESS", filt.LOCAL_THICKNESS, [
            F("sizes", "int", "10"),
            F("method", "choice_str", "dt", ["dt", "conv", "bf", "imj"]),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
        ("POROSIMETRY", filt.POROSIMETRY, [
            F("sizes", "int", "15"),
            F("method", "choice_str", "dt", ["dt", "conv", "dsi"]),
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True),
            F("inv_axis", "choice_int_none", "None", AXIS_NONE)]),
        ("REGION_SIZE", filt.REGION_SIZE, [
            F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ]),
]

METRIC_SPECS = [
    ("BOND_NUMBER", met.BOND_NUMBER, [
        F("g", "float", "9.81"), F("sigma", "float", "0.01"),
        F("delta_rho", "float", "1000"), F("voxel_size", "float", "0.001"),
        F("source", "choice_str", "lt", ["lt", "dt"]),
        F("method", "choice_str", "median", ["mean", "median", "min", "max"])]),
    ("CHORD_COUNTS", met.CHORD_COUNTS, [
        F("spacing", "int", "1"), F("trim_edges", "bool", True),
        F("apply_chords3D", "bool", True), F("axis", "choice_int", "0", AXIS3),
        F("bins", "int", "25")]),
    ("CHORD_LENGTH_DISTRIBUTION", met.CHORD_LENGTH_DISTRIBUTION, [
        F("bins", "int", "25"), F("log", "bool", False), F("voxel_size", "float", "1"),
        F("normalization", "choice_str", "number", ["count", "number", "length"]),
        F("apply_chords3D", "bool", True), F("axis", "choice_int", "0", AXIS3),
        F("save_excel", "bool", False)]),
    ("FIND_H", met.FIND_H, [
        F("srange", "list_float", "0.01,0.93"),
        F("axis_of_intrusion", "choice_int", "0", AXIS3),
        F("voxel_size", "float", "1"), F("g", "float", "0"), F("s", "float", "0.5")]),
    ("FIND_POROSITY_THRESHOLD", met.FIND_POROSITY_THRESHOLD, [
        F("axis", "choice_int", "0", AXIS3),
        F("conn", "choice_str", "max", ["max", "min"])]),
    ("IS_PERCOLATING", met.IS_PERCOLATING, [
        F("conn", "choice_str", "max", ["max", "min"])]),
    ("LINEAL_PATH_DISTRIBUTION", met.LINEAL_PATH_DISTRIBUTION, [
        F("bins", "int", "25"), F("log", "bool", False),
        F("voxel_size", "float", "1"), F("save_excel", "bool", False)]),
    ("MESH_SURFACE_AREA", met.MESH_SURFACE_AREA, []),
    ("MESH_VOLUME", met.MESH_VOLUME, []),
    ("PC_MAP_TO_PC_CURVE", met.PC_MAP_TO_PC_CURVE, [
        F("sigma", "float", "0.01"), F("theta", "float", "180"),
        F("voxel_size", "float", "1"), F("steps", "int", "25"),
        F("save_excel", "bool", False)]),
    ("PERCOLATING_POROSITY", met.PERCOLATING_POROSITY, [
        F("conn", "choice_str", "max", ["max", "min"])]),
    ("PORE_SIZE_DISTRIBUTION", met.PORE_SIZE_DISTRIBUTION, [
        F("bins", "int", "25"), F("voxel_size", "float", "1"),
        F("log", "bool", False), F("save_excel", "bool", False)]),
    ("POROSITY", met.POROSITY, []),
    ("POROSITY_PROFILE", met.POROSITY_PROFILE, [
        F("voxel_size", "float", "1")]),
    ("RADIAL_DENSITY_DISTRIBUTION", met.RADIAL_DENSITY_DISTRIBUTION, [
        F("bins", "int", "25"), F("log", "bool", False),
        F("voxel_size", "float", "1"), F("save_excel", "bool", False)]),
    ("REGION_SURFACE_AREAS", met.REGION_SURFACE_AREAS, [
        F("voxel_size", "float", "1"), F("strel_size", "int", "2"),
        F("slice", "choice_int", "0", AXIS3), F("save_excel", "bool", False)]),
    ("REGION_VOLUMES", met.REGION_VOLUMES, [
        F("method", "choice_str", "voxel", ["voxel", "marching_cubes"]),
        F("slice", "choice_int", "0", AXIS3), F("save_excel", "bool", False)]),
    ("REGIONPROPS_3D", met.REGIONPROPS_3D, [
        F("sphericity", "bool", False), F("solidity", "bool", False),
        F("num_pixels", "bool", False)]),
    ("REV_POROSITY", met.REV_POROSITY, [
        F("n", "int", "1000")]),
    ("SATN_PROFILE", met.SATN_PROFILE, [
        F("span", "int", "2"), F("mode", "choice_str", "tile", ["tile", "slide"]),
        F("intr_axis", "choice_int", "0", AXIS3), F("sigma", "float", "0.01"),
        F("theta", "float", "180"), F("g", "float", "0")]),
]

_DRAIN_IMB_FIELDS = [
    F("sigma", "float", "0.01"), F("theta", "float", "180"), F("g", "float", "9.81"),
    F("rho_nwp", "float", "1000"), F("rho_wp", "float", "0"),
    F("voxel_size", "float", "1e-4"), F("axis", "choice_int_none", "None", AXIS_NONE),
    F("min_size", "int", "5"), F("steps", "int_none", ""), F("save_excel", "bool", False),
]
PSIM_SPECS = [
    ("DRAINAGE", psim.DRAINAGE, list(_DRAIN_IMB_FIELDS)),
    ("IMBIBITION", psim.IMBIBITION, list(_DRAIN_IMB_FIELDS)),
    ("INJECTION", psim.INJECTION, [
        F("sigma", "float", "0.01"), F("theta", "float", "180"), F("g", "float", "9.81"),
        F("rho_nwp", "float", "1000"), F("rho_wp", "float", "0"),
        F("voxel_size", "float", "1e-4"), F("axis", "choice_int_none", "None", AXIS_NONE),
        F("min_size", "int", "5"), F("conn", "choice_str", "max", ["max", "min"]),
        F("maxiter", "int", "500"), F("method", "choice_str", "qbip", ["qbip", "ibip"]),
        F("save_excel", "bool", False)]),
    ("TORTUOSITY", psim.TORTUOSITY, [
        F("axis", "choice_int", "0", AXIS3)]),
]

# OpenPNM phase property forms (only the properties relevant to each phase).
PNM_PHASE_SPECS = [
    ("water", [
        F("diffusivity", "float", "2.3e-9"),
        F("viscosity", "float", "1.0e-3"),
        F("contact_angle", "float", "110.0")]),
    ("air", [
        F("diffusivity", "float", "2.5e-5"),
        F("viscosity", "float", "1.85e-5"),
        F("contact_angle", "float", "120.0"),
        F("surface_tension", "float", "0.072")]),
    ("mercury", [
        F("viscosity", "float", "1.53e-3"),
        F("diffusivity", "float", "1.7e-9"),
        F("surface_tension", "float", "0.485"),
        F("contact_angle", "float", "140.0")]),
]

# OpenPNM simulations: only the requested subset of parameters per function.
PNM_SIM_SPECS = [
    ("Stokes flow", opnm.simulate_stokes_flow, [
        F("axis", "choice_str", "x", XYZ),
        F("inlet_pressure", "float", "1.0"), F("outlet_pressure", "float", "0.0")]),
    ("Fickian diffusion", opnm.simulate_fickian_diffusion, [
        F("axis", "choice_str", "x", XYZ),
        F("inlet_concentration", "float", "1.0"), F("outlet_concentration", "float", "0.0")]),
    ("Advection-diffusion", opnm.simulate_advection_diffusion, [
        F("axis", "choice_str", "x", XYZ),
        F("inlet_pressure", "float", "1.0"), F("outlet_pressure", "float", "0.0"),
        F("inlet_concentration", "float", "1.0"), F("outlet_concentration", "float", "0.0")]),
    ("Drainage", opnm.simulate_drainage, [
        F("axis", "choice_str", "x", XYZ), F("save_excel", "bool", False)]),
    ("Invasion percolation", opnm.simulate_invasion_percolation, [
        F("axis", "choice_str", "x", XYZ), F("save_excel", "bool", False)]),
]


# ===========================================================================
#  Application
# ===========================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.image = None          # current working binary image (numpy bool array)
        self.image_name = None     # base name used for filter output filenames
        self.pn = None             # cached OpenPNM network (built lazily)
        self.pn_image_id = None    # id() of the image the network was built from

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
        chooser.geometry("380x200")
        chooser.transient(self.root)
        tk.Label(chooser, text="Select the image source",
                 font=("Segoe UI", 12, "bold")).pack(pady=(18, 4))
        tk.Label(chooser, text="A single 3D .tif or .raw file, or a folder of 2D .tif slices.",
                 fg="#555", wraplength=340).pack(pady=(0, 12))

        def pick_file():
            path = filedialog.askopenfilename(
                parent=chooser, title="Select a 3D image file",
                filetypes=[("3D image", "*.tif *.tiff *.raw"),
                            ("TIFF image", "*.tif *.tiff"),
                            ("Raw binary", "*.raw"),
                            ("All files", "*.*")])
            if path:
                chooser.destroy()
                self._do_import(path, is_folder=False)

        def pick_folder():
            path = filedialog.askdirectory(parent=chooser,
                                           title="Select a folder of 2D .tif slices")
            if path:
                chooser.destroy()
                self._do_import(path, is_folder=True)

        tk.Button(chooser, text="Select 3D image file (.tif / .raw)", width=30,
                  command=pick_file).pack(pady=4)
        tk.Button(chooser, text="Select folder of 2D slices", width=30,
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

        # --- load -------------------------------------------------------
        def _load():
            if is_folder:
                seq = imread_collection(os.path.join(path, "*.tif"))
                if len(seq) == 0:
                    raise ValueError("No .tif slices found in the selected folder.")
                arr = np.stack(seq, axis=-1)
                print(f"Loaded {len(seq)} slices via imread_collection: shape={arr.shape}")
            elif raw_shape is not None:
                arr = np.fromfile(path, dtype=np.dtype(raw_dtype)).reshape(raw_shape)
                arr = np.transpose(arr, (1, 2, 0))   # (slices,H,W) → (H,W,slices)
                print(f"Loaded .raw image: shape={arr.shape}, dtype={raw_dtype}")
            else:
                arr = imageio.volread(path)
                arr = np.transpose(arr, (1, 2, 0))
                print(f"Loaded 3D image via volread: shape={arr.shape}")
            return arr

        ok, im3d, out, err = run_operation(self.root, "Loading image", _load, {})
        if not ok:
            messagebox.showerror("Import failed", err, parent=self.root)
            return

        unique = np.unique(im3d)
        if len(unique) <= 2:
            messagebox.showinfo("Import",
                                f"Image is already binary (values {unique}).\n"
                                "No thresholding needed.", parent=self.root)
            im_binary = im3d.astype(bool)
        else:
            # --- denoise (slow) -----------------------------------------
            im_float = im3d.astype(np.float64)
            im_float = (im_float - im_float.min()) / (im_float.max() - im_float.min())

            def _denoise():
                import porespy as ps
                d = ps.filters.nl_means_layered(im_float, axis=2)
                return (d - d.min()) / (d.max() - d.min())

            ok, denoised, out, err = run_operation(self.root, "Denoising (NL-means)", _denoise, {})
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

            mid = im3d.shape[2] // 2
            fig, ax = plt.subplots(1, 3, figsize=[15, 5])
            ax[0].imshow(im3d[:, :, mid], cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
            ax[1].imshow(denoised[:, :, mid], cmap="gray"); ax[1].set_title("Denoised"); ax[1].axis("off")
            ax[2].imshow(~im_binary[:, :, mid], cmap="gray"); ax[2].set_title("Binary"); ax[2].axis("off")
            fig.tight_layout()
            plt.show()

            # --- save the binary image (only when thresholding was needed) ----
            out_dir = ask_output_dir(self.root)
            if out_dir:
                try:
                    save_arr = np.transpose(np.array(~im_binary, dtype=np.int8), (2, 0, 1))
                    imageio.volsave(os.path.join(out_dir, "imported_binary_3D.tif"), save_arr)
                    messagebox.showinfo("Saved",
                                        f"Binary image saved to:\n{out_dir}", parent=self.root)
                except Exception as e:  # noqa: BLE001
                    messagebox.showwarning("Save warning", f"Could not save image:\n{e}",
                                           parent=self.root)

        self.image_name = (os.path.basename(os.path.normpath(path))
                           if is_folder
                           else os.path.splitext(os.path.basename(path))[0])
        self.image = im_binary
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
                if out_dir is None:
                    return
            else:
                out_dir, chosen_name = None, None
            ok, result, out, err = run_operation(win, f"Generating {sec['title']}",
                                                  sec["func"], kwargs, output_dir=out_dir)
            if ok:
                _rename_saved(out_dir, save_file, chosen_name)
                self.image = result
                self.image_name = sec['title'].capitalize()
                self.pn = None  # invalidate any cached network
                messagebox.showinfo("Done",
                                    f"{sec['title']} generated. Click Next to filter the image.",
                                    parent=win)
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

    # ---- Filters window --------------------------------------------------
    def open_filters(self, prev=None):
        if prev:
            prev.destroy()
        total_filters = sum(len(specs) for _, specs in FILTER_GROUPS)
        win, body, bar = make_window(
            self, "Filters",
            "Filters are applied to the current image. To apply one or more filters, "
            "set a run order (1, 2, 3, ...) next to each filter you want; the image is "
            "passed through them in that order. Leave the order blank to skip a filter.")
        order_values = [""] + [str(i) for i in range(1, total_filters + 1)]
        sections = []
        for group_label, specs in FILTER_GROUPS:
            tk.Label(body, text=group_label,
                     font=("Segoe UI", 11, "bold"), fg="#1a1a1a",
                     anchor="w").pack(fill="x", padx=10, pady=(10, 2))
            ttk.Separator(body, orient="horizontal").pack(fill="x", padx=10, pady=(0, 4))
            group_container = tk.Frame(body)
            group_container.pack(fill="x")
            group_secs = []
            for title, func, fields in specs:
                sec = build_section(group_container, title, fields,
                                    mode="order", order_values=order_values)
                sec["func"] = func
                group_secs.append(sec)
                sections.append(sec)
            grid_sections(group_container, group_secs, 2)

        def do_filter():
            if not self.require_image(win):
                return
            chosen = [(section_order(s), s) for s in sections if section_order(s) is not None]
            if not chosen:
                messagebox.showinfo("No filters",
                                    "No run order set. The current image will be passed on unchanged.",
                                    parent=win)
                return
            orders = [o for o, _ in chosen]
            if len(set(orders)) != len(orders):
                messagebox.showerror("Order clash",
                                     "Each filter must have a unique run order.", parent=win)
                return
            chosen.sort(key=lambda t: t[0])

            base = self.image_name or "image"
            suffix = "_".join(s["title"].lower() for _, s in chosen)
            default_name = f"{base}_{suffix}.tif"
            out_dir, chosen_name = ask_save_file(win, default_name)
            if out_dir is None:
                return

            current = self.image
            for order, sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    messagebox.showerror("Invalid input", f"{sec['title']}: {e}", parent=win)
                    return
                kwargs["im"] = current
                ok, result, out, err = run_operation(
                    win, f"Filtering: {sec['title']} (step {order})",
                    sec["func"], kwargs)
                if not ok:
                    messagebox.showerror("Filter failed", f"{sec['title']}:\n{err}", parent=win)
                    return
                current = result

            import imageio.v2 as _iio
            save_name = chosen_name or default_name
            try:
                arr = current.astype(np.uint8) if current.dtype == bool else current.astype(np.float32)
                save_arr = np.transpose(arr, (2, 0, 1))
                _iio.volsave(os.path.join(out_dir, save_name), save_arr)
            except Exception as _e:
                messagebox.showwarning("Save warning",
                                       f"Could not save {save_name}:\n{_e}", parent=win)

            self.image = current
            self.pn = None
            messagebox.showinfo("Done", "Filtering complete. Click Next for metrics.", parent=win)

        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_generators(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Filter", width=14, bg="#90EE90",
                  command=do_filter).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_metrics(win)).grid(row=0, column=2, sticky="e", padx=5)

    # ---- Metrics window --------------------------------------------------
    def open_metrics(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "Metrics",
            "Tick any number of metrics, set their parameters, then click Calculate. "
            "Each section shows its printed result (green) or error (red); some also "
            "produce plots or Excel files.")
        sections = []
        for title, func, fields in METRIC_SPECS:
            sec = build_section(body, title, fields, mode="multi", with_result=True)
            sec["func"] = func
            sections.append(sec)
        grid_sections(body, sections, 2)

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
            for sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    sec["result"].show(str(e), ok=False)
                    continue
                kwargs["im"] = self.image
                ok, result, out, err = run_operation(
                    win, f"Calculating {sec['title']}", sec["func"], kwargs, output_dir=out_dir)
                if ok:
                    sec["result"].show(out or "Done (see plots / saved files).", ok=True)
                    sf = get_save_file(sec)
                    if sf and out_dir and sf in name_map:
                        _rename_saved(out_dir, sf, name_map[sf])
                else:
                    sec["result"].show(err, ok=False)

        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_filters(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Calculate", width=14, bg="#90EE90",
                  command=do_calculate).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_porespy(win)).grid(row=0, column=2, sticky="e", padx=5)

    # ---- PoreSpy simulations window -------------------------------------
    def open_porespy(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "PoreSpy simulations",
            "Tick any number of simulations, set their parameters, then click Simulate. "
            "Printed output (e.g. the tortuosity value) is shown in each section's box.")
        sections = []
        for title, func, fields in PSIM_SPECS:
            sec = build_section(body, title, fields, mode="multi", with_result=True)
            sec["func"] = func
            sections.append(sec)
        grid_sections(body, sections, 2)

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
            for sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    sec["result"].show(str(e), ok=False)
                    continue
                kwargs["im"] = self.image
                ok, result, out, err = run_operation(
                    win, f"Simulating {sec['title']}", sec["func"], kwargs, output_dir=out_dir)
                if ok:
                    sec["result"].show(out or "Done (see plots / saved files).", ok=True)
                    sf = get_save_file(sec)
                    if sf and out_dir and sf in name_map:
                        _rename_saved(out_dir, sf, name_map[sf])
                else:
                    sec["result"].show(err, ok=False)

        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_metrics(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Simulate", width=14, bg="#90EE90",
                  command=do_simulate).grid(row=0, column=1)
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_openpnm(win)).grid(row=0, column=2, sticky="e", padx=5)

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

        tk.Label(body, text="Phase selection", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10, pady=(6, 0))
        phase_container = tk.Frame(body)
        phase_container.pack(fill="x")
        phase_single = SingleSelect()
        phase_secs = {}
        for name, fields in PNM_PHASE_SPECS:
            sec = build_section(phase_container, name.capitalize(), fields,
                                mode="single", single=phase_single)
            phase_secs[name] = sec
        grid_sections(phase_container, list(phase_secs.values()), 3)

        tk.Label(body, text="Simulations", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10, pady=(12, 0))
        sim_container = tk.Frame(body)
        sim_container.pack(fill="x")
        sim_secs = []
        for title, func, fields in PNM_SIM_SPECS:
            sec = build_section(sim_container, title, fields, mode="multi", with_result=True)
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

            # Build (or reuse) the pore network from the current image.
            if self.pn is None or self.pn_image_id != id(self.image):
                ok, pn, out, err = run_operation(
                    win, "Building pore network (snow2)", opnm.build_network,
                    {"image": self.image}, output_dir=out_dir)
                if not ok:
                    messagebox.showerror("Network build failed", err, parent=win)
                    return
                self.pn = pn
                self.pn_image_id = id(self.image)

            for sec in chosen:
                try:
                    kwargs = collect_kwargs(sec["ff"])
                except ValueError as e:
                    sec["result"].show(str(e), ok=False)
                    continue
                kwargs["network"] = self.pn
                kwargs["phase_overrides"] = overrides
                func = sec["func"]
                if func in (opnm.simulate_drainage, opnm.simulate_invasion_percolation):
                    kwargs["inv_phase"] = phase_name
                    kwargs["def_phase"] = "air" if phase_name == "water" else "water"
                else:
                    kwargs["chosen_phase"] = phase_name
                ok, result, out, err = run_operation(
                    win, f"OpenPNM: {sec['title']}", func, kwargs, output_dir=out_dir)
                if ok:
                    sec["result"].show(out or "Done (see saved VTK files).", ok=True)
                    sf = get_save_file(sec)
                    if sf and out_dir and sf in name_map:
                        _rename_saved(out_dir, sf, name_map[sf], subdir="results_paraview")
                else:
                    sec["result"].show(err, ok=False)

        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(2, weight=1)
        tk.Button(bar, text="← Previous", width=12,
                  command=lambda: self.open_porespy(win)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Button(bar, text="Simulate", width=14, bg="#90EE90",
                  command=do_simulate).grid(row=0, column=1)
        tk.Button(bar, text="Finish", width=12,
                  command=win.destroy).grid(row=0, column=2, sticky="e", padx=5)


def main():
    os.chdir(SCRIPT_DIR)
    root = tk.Tk()
    app = App(root)
    app.build_main()
    root.mainloop()


if __name__ == "__main__":
    main()
