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
import sys
import traceback
import contextlib

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
        if t == "choice_int_none" and s in ("", "None"):
            return None
        if t in ("int", "choice_int"):
            return int(s)
        if t == "float":
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
    """A label with a coloured border used to show captured output / errors."""

    def __init__(self, master):
        super().__init__(master, bg="#bbbbbb", padx=2, pady=2)
        self.label = tk.Label(self, text="(no result yet)", bg="white", anchor="w",
                              justify="left", wraplength=380)
        self.label.pack(fill="both", expand=True)

    def show(self, text, ok=True):
        self.configure(bg="#2e7d32" if ok else "#c62828")
        self.label.configure(text=text if text else ("Done." if ok else "Error"),
                             fg="#000000")


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


def ask_output_dir(parent):
    """Ask for an output folder; None -> keep current working directory."""
    d = filedialog.askdirectory(parent=parent,
                                title="Choose a folder for the files this step will save")
    return d or None


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
    return ok, result, buf.getvalue().strip(), err


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
        F("blobiness", "int", "1")]),
    ("CYLINDERS", gen.CYLINDERS, [
        F("shape", "list_int", "100,100,100"),
        F("radius", "int", "2"),
        F("ncylinders", "int", "10"),
        F("porosity", "float", "0.5"),
        F("cyl_length", "int_none", "")]),
    ("OVERLAPPING_SPHERES", gen.OVERLAPPING_SPHERES, [
        F("shape", "list_int", "100,100,100"),
        F("radius", "int", "5"),
        F("porosity", "float", "0.5"),
        F("maxiter", "int", "10")]),
    ("POLYDISPERSE_SPHERES", gen.POLYDISPERSE_SPHERES, [
        F("shape", "list_int", "100,100,100"),
        F("porosity", "float", "0.5"),
        F("r_min", "int", "4"),
        F("nbins", "int", "3")]),
    ("PSEUDO_GRAVITY_PACKING", gen.PSEUDO_GRAVITY_PACKING, [
        F("im", "list_int", "50,50,50", label="shape"),
        F("radius", "int", "5"),
        F("maxiter", "int", "500"),
        F("clearance", "int", "0"),
        F("pk_axis", "choice_int", "0", AXIS3)]),
    ("VORONOI_EDGES", gen.VORONOI_EDGES, [
        F("shape", "list_int", "100,100,100"),
        F("thickness", "int", "2"),
        F("ncells", "int", "10"),
        F("flat_faces", "bool", True)]),
]

FILTER_SPECS = [
    ("APPLY_CHORDS", filt.APPLY_CHORDS, [
        F("spacing", "int", "1"), F("axis", "choice_int", "1", AXIS3),
        F("trim_edges", "bool", False), F("label", "bool", True),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("APPLY_CHORDS_3D", filt.APPLY_CHORDS_3D, [
        F("spacing", "int", "2"), F("trim_edges", "bool", True),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("FILL_CLOSED_PORES", filt.FILL_CLOSED_PORES, [
        F("conn", "choice_int", "26", ["6", "26"]),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("FIND_DISCONNECTED_VOXELS", filt.FIND_DISCONNECTED_VOXELS, [
        F("conn", "choice_str", "max", ["max", "min"]),
        F("show_3d", "bool", True)]),
    ("LOCAL_THICKNESS", filt.LOCAL_THICKNESS, [
        F("sizes", "int", "10"),
        F("method", "choice_str", "dt", ["dt", "conv", "bf", "imj"]),
        F("divs", "int", "2"),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("POROSIMETRY", filt.POROSIMETRY, [
        F("sizes", "int", "15"),
        F("method", "choice_str", "dt", ["dt", "conv", "dsi"]),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True),
        F("inv_axis", "choice_int_none", "None", AXIS_NONE)]),
    ("REGION_SIZE", filt.REGION_SIZE, [
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("TRIM_DISCONNECTED_VOXELS", filt.TRIM_DISCONNECTED_VOXELS, [
        F("conn", "choice_str", "max", ["max", "min"]),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("TRIM_FLOATING_SOLID", filt.TRIM_FLOATING_SOLID, [
        F("conn", "choice_str", "max", ["max", "min"]),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
    ("TRIM_NONPERCOLATING_PATHS", filt.TRIM_NONPERCOLATING_PATHS, [
        F("perc_axis", "choice_int", "0", AXIS3),
        F("slice", "choice_int", "0", AXIS3), F("show_slice", "bool", True)]),
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
        tk.Label(chooser, text="A single 3D .tif file, or a folder of 2D .tif slices.",
                 fg="#555", wraplength=340).pack(pady=(0, 12))

        def pick_file():
            path = filedialog.askopenfilename(
                parent=chooser, title="Select a 3D .tif image",
                filetypes=[("TIFF image", "*.tif *.tiff"), ("All files", "*.*")])
            if path:
                chooser.destroy()
                self._do_import(path, is_folder=False)

        def pick_folder():
            path = filedialog.askdirectory(parent=chooser,
                                           title="Select a folder of 2D .tif slices")
            if path:
                chooser.destroy()
                self._do_import(path, is_folder=True)

        tk.Button(chooser, text="Select 3D image file (.tif)", width=30,
                  command=pick_file).pack(pady=4)
        tk.Button(chooser, text="Select folder of 2D slices", width=30,
                  command=pick_folder).pack(pady=4)

    def _do_import(self, path, is_folder):
        import imageio.v2 as imageio
        from skimage.io import imread_collection

        # --- load -------------------------------------------------------
        def _load():
            if is_folder:
                seq = imread_collection(os.path.join(path, "*.tif"))
                if len(seq) == 0:
                    raise ValueError("No .tif slices found in the selected folder.")
                arr = np.stack(seq, axis=-1)
                print(f"Loaded {len(seq)} slices via imread_collection: shape={arr.shape}")
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

        # --- save the binary image -------------------------------------
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
            out_dir = ask_output_dir(win)
            ok, result, out, err = run_operation(win, f"Generating {sec['title']}",
                                                  sec["func"], kwargs, output_dir=out_dir)
            if ok:
                self.image = result
                self.pn = None  # invalidate any cached network
                messagebox.showinfo("Done",
                                    f"{sec['title']} generated. Click Next to filter the image.",
                                    parent=win)
            else:
                messagebox.showerror("Generation failed", err, parent=win)

        tk.Button(bar, text="Generate", width=14, command=do_generate).pack(side="left")
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_filters(win)).pack(side="right")

    # ---- Filters window --------------------------------------------------
    def open_filters(self, prev=None):
        if prev:
            prev.destroy()
        win, body, bar = make_window(
            self, "Filters",
            "Filters are applied to the current image. To apply one or more filters, "
            "set a run order (1, 2, 3, ...) next to each filter you want; the image is "
            "passed through them in that order. Leave the order blank to skip a filter.")
        order_values = [""] + [str(i) for i in range(1, len(FILTER_SPECS) + 1)]
        sections = []
        for title, func, fields in FILTER_SPECS:
            sec = build_section(body, title, fields, mode="order", order_values=order_values)
            sec["func"] = func
            sections.append(sec)
        grid_sections(body, sections, 2)

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
            # Filters don't write any files to disk, so no output folder is needed.
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
            self.image = current
            self.pn = None
            messagebox.showinfo("Done", "Filtering complete. Click Next for metrics.", parent=win)

        tk.Button(bar, text="Filter", width=14, command=do_filter).pack(side="left")
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_metrics(win)).pack(side="right")

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
            out_dir = ask_output_dir(win)
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
                else:
                    sec["result"].show(err, ok=False)

        tk.Button(bar, text="Calculate", width=14, command=do_calculate).pack(side="left")
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_porespy(win)).pack(side="right")

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
            out_dir = ask_output_dir(win)
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
                else:
                    sec["result"].show(err, ok=False)

        tk.Button(bar, text="Simulate", width=14, command=do_simulate).pack(side="left")
        tk.Button(bar, text="Next →", width=12,
                  command=lambda: self.open_openpnm(win)).pack(side="right")

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

            out_dir = ask_output_dir(win)

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
                else:
                    sec["result"].show(err, ok=False)

        tk.Button(bar, text="Simulate", width=14, command=do_simulate).pack(side="left")
        tk.Button(bar, text="Finish", width=12, command=win.destroy).pack(side="right")


def main():
    os.chdir(SCRIPT_DIR)
    root = tk.Tk()
    app = App(root)
    app.build_main()
    root.mainloop()


if __name__ == "__main__":
    main()
