"""
Fable UI — Porous Media Studio
==============================

A modern single-window app that drives the thesis analysis pipeline built on
the project modules (Generators_3D, Filters_3D, Metrics_3D,
Simulations3D_porespy, Simulations3D_openPNM):

    Generate / Import  ->  Filters  ->  Metrics  ->  PoreSpy sims  ->  OpenPNM sims

What is different from the original app_gui.py
----------------------------------------------
* One window with a sidebar — no chain of pop-up wizard windows. You can jump
  between steps at any time; the working image is shared by every page.
* All computations run on a background worker thread, so the UI stays
  responsive; a status bar shows what is running and for how long.
* Matplotlib figures produced by the module functions are captured into an
  in-app, zoomable Plot Gallery (no scattered native windows to close).
* A built-in slice viewer shows the current working volume (axis + slider),
  using the PoreSpy colour convention (yellow = pore/void, blue = solid).
* Collapsible parameter cards with tooltips and per-operation time estimates.
* A session Log page records every run, its output and its duration.

The philosophy of the original GUI is preserved:
* Filters: solid-altering filters chain in a user-chosen order and replace the
  working image; non-solid-altering filters run independently on that result.
* Metrics / simulations: tick any number, each shows captured stdout in a
  green (success) / red (error) result box.
* Functions that write files (tif / xlsx / vti / vtp) ask where to save first;
  the working directory is temporarily switched so the modules' hard-coded
  filenames land in the chosen folder, then files are renamed as requested.

Run with the same Python environment as the project:
    python Fable_UI.py
(Optional: pydicom for DICOM import.)
"""

import os
import io
import re
import sys
import math
import time
import queue
import threading
import warnings
import traceback
import contextlib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")            # figures are rendered off-screen and captured
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np               # noqa: E402

import tkinter as tk             # noqa: E402
from tkinter import ttk, filedialog, messagebox  # noqa: E402

warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", message=".*cannot be shown.*")

_ANSI_RE = re.compile(r'\x1b\[[\d;]*[A-Za-z]')

# --- Locate and import the project modules ---------------------------------
# This file lives in <project>/ui/ ; the analysis modules live in the
# <project>/logic/ package. To import them as `logic.*`, the directory that
# CONTAINS the `logic/` package (the project root) must be on sys.path.
UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(UI_DIR)
MODULE_DIR_CANDIDATES = [PROJECT_ROOT, UI_DIR]

APP_TITLE = "Porous Media Analysis"

# Output folder: next to this .py file, or next to the .exe when frozen
# (PyInstaller sets sys.frozen and sys.executable points at the bundle exe).
if getattr(sys, "frozen", False):
    _APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    _APP_BASE_DIR = UI_DIR
OUTPUT_DIR_DEFAULT = os.path.join(_APP_BASE_DIR, "UI_output")


def asset_path(name):
    """Resolve a file in assets/: PyInstaller bundle dir, project root or ui/."""
    for base in (getattr(sys, "_MEIPASS", None), _APP_BASE_DIR,
                 PROJECT_ROOT, UI_DIR):
        if base:
            p = os.path.join(base, "assets", name)
            if os.path.isfile(p):
                return p
    return None


_APP_ICON_CACHE = {}    # size -> PhotoImage (kept alive: Tk keeps weak refs)


def load_app_icon(size=None):
    """assets/Icon.png as a Tk PhotoImage, scaled to `size` px if given.
    A circular mask is applied because the artwork's baked-in background is
    opaque on the left edge (a hard straight edge would otherwise show).
    Needs a Tk root to exist. Returns None if the asset is missing."""
    key = size or 0
    if key in _APP_ICON_CACHE:
        return _APP_ICON_CACHE[key]
    path = asset_path("Icon.png")
    if not path:
        return None
    photo = None
    if size:
        try:                             # smooth resize, keeps transparency
            from PIL import Image, ImageChops, ImageDraw, ImageTk
            im = Image.open(path).convert("RGBA")
            w, h = im.size
            side = min(w, h)
            im = im.crop(((w - side) // 2, (h - side) // 2,
                          (w + side) // 2, (h + side) // 2))
            mask = Image.new("L", (side, side), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, side - 1, side - 1), fill=255)
            # circle badge, and keep the artwork's own transparency
            im.putalpha(ImageChops.multiply(im.getchannel("A"), mask))
            im = im.resize((size, size), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
        except Exception:  # noqa: BLE001
            photo = None
    if photo is None:
        try:
            photo = tk.PhotoImage(file=path)   # Tk 8.6 reads PNG natively
            if size and photo.width() > size:
                photo = photo.subsample(max(1, round(photo.width() / size)))
        except tk.TclError:
            return None
    _APP_ICON_CACHE[key] = photo
    return photo


gen = filt = met = psim = opnm = None


def load_project_modules():
    """Import the thesis modules (heavy: porespy/openpnm). Called after the
    splash screen is painted so start-up does not look frozen."""
    global gen, filt, met, psim, opnm
    for cand in MODULE_DIR_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "logic", "Generators_3D.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            break
    else:
        raise ImportError(
            "Could not find the project modules (logic/Generators_3D.py etc.).\n"
            "Looked in:\n" + "\n".join(MODULE_DIR_CANDIDATES))
    import logic.Generators_3D as _gen
    import logic.Filters_3D as _filt
    import logic.Metrics_3D as _met
    import logic.Simulations3D_porespy as _psim
    import logic.Simulations3D_openPNM as _opnm
    gen, filt, met, psim, opnm = _gen, _filt, _met, _psim, _opnm


# ===========================================================================
#  Theme
# ===========================================================================
BG       = "#12151c"   # app background
SIDE     = "#0d1017"   # sidebar
CARD     = "#1b2029"   # card background
CARD_HI  = "#222a38"   # card header hover / selected
FIELD    = "#242b38"   # entry background
BORDER   = "#2b3342"
TEXT     = "#e6e9f0"
MUTED    = "#8b93a7"
ACCENT   = "#4f9cf9"
ACCENT_D = "#3b82e0"
CRIMSON  = "#dc143c"   # app title (matches the red of the icon)
GOOD     = "#34d399"
GOOD_D   = "#14532d"
BAD      = "#f87171"
BAD_D    = "#7f1d1d"
WARN     = "#fbbf24"

FONT      = ("Segoe UI", 10)
FONT_SM   = ("Segoe UI", 9)
FONT_B    = ("Segoe UI", 10, "bold")
FONT_H1   = ("Segoe UI", 17, "bold")
FONT_H2   = ("Segoe UI", 12, "bold")
FONT_IT   = ("Segoe UI", 9, "italic")

PORE_RGB  = (253, 231, 37)   # viridis yellow  — True  = pore / void
SOLID_RGB = (48, 25, 82)     # deep violet     — False = solid


def style_app(root):
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox", fieldbackground=FIELD, background=CARD,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER,
                    lightcolor=CARD, darkcolor=CARD, selectbackground=FIELD,
                    selectforeground=TEXT)
    style.map("TCombobox",
              fieldbackground=[("readonly", FIELD)],
              foreground=[("readonly", TEXT)])
    style.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED)
    style.configure("Horizontal.TScrollbar", background=CARD, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED)
    style.configure("TProgressbar", background=ACCENT, troughcolor=FIELD,
                    bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
    style.configure("TSeparator", background=BORDER)
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_D)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)


def hover(widget, normal, hi):
    widget.bind("<Enter>", lambda e: widget.configure(bg=hi))
    widget.bind("<Leave>", lambda e: widget.configure(bg=normal))


def make_button(parent, text, command, primary=False, width=None):
    bg = ACCENT if primary else FIELD
    hi = ACCENT_D if primary else CARD_HI
    fg = "#0b1220" if primary else TEXT
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                  activebackground=hi, activeforeground=fg, relief="flat",
                  font=FONT_B if primary else FONT, bd=0, padx=14, pady=5,
                  cursor="hand2", width=width or 0,
                  disabledforeground=MUTED)
    hover(b, bg, hi)
    return b


# ===========================================================================
#  Form-field model + value parsing   (same semantics as app_gui.py)
# ===========================================================================
def F(name, ftype, default, choices=None, label=None, tooltip=None):
    return {"name": name, "type": ftype, "default": default,
            "choices": choices, "label": label or name, "tooltip": tooltip}


class FormField:
    def __init__(self, field, var):
        self.field = field
        self.var = var

    def get(self):
        f = self.field
        t = f["type"]
        if t == "bool":
            return bool(self.var.get())
        s = str(self.var.get()).strip()
        if t == "int_none" and s == "":
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
        return s


def collect_kwargs(ff):
    out = {}
    for name, field in ff.items():
        try:
            out[name] = field.get()
        except ValueError:
            raise ValueError(f"Invalid value for '{field.field['label']}'.")
    return out


def build_form(parent, fields):
    """Grid of labelled dark widgets; returns {name: FormField}."""
    ff = {}
    for i, f in enumerate(fields):
        lbl = tk.Label(parent, text=f["label"], bg=CARD, fg=MUTED, font=FONT_SM)
        lbl.grid(row=i, column=0, sticky="w", padx=(6, 10), pady=2)
        if f.get("tooltip"):
            Tooltip(lbl, f["tooltip"])
        t = f["type"]
        if t == "bool":
            var = tk.BooleanVar(value=bool(f["default"]))
            cb = tk.Checkbutton(parent, variable=var, bg=CARD,
                                activebackground=CARD, selectcolor=FIELD,
                                fg=ACCENT, bd=0, highlightthickness=0)
            cb.grid(row=i, column=1, sticky="w")
            lbl.configure(cursor="hand2")
            lbl.bind("<Button-1>", lambda e, v=var: v.set(not v.get()))
        elif t.startswith("choice"):
            var = tk.StringVar(value=str(f["default"]))
            ttk.Combobox(parent, textvariable=var, values=f["choices"],
                         state="readonly", width=16, font=FONT_SM
                         ).grid(row=i, column=1, sticky="w", padx=2, pady=1)
        else:
            var = tk.StringVar(value=str(f["default"]))
            e = tk.Entry(parent, textvariable=var, width=18, bg=FIELD, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=FONT_SM,
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT)
            e.grid(row=i, column=1, sticky="w", padx=2, pady=1)
        ff[f["name"]] = FormField(f, var)
    return ff


# ===========================================================================
#  Reusable widgets
# ===========================================================================
class Tooltip:
    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._win = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e=None):
        if self._win or not self._text:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 2
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text, background="#0b0e14", foreground=TEXT,
                 relief="solid", borderwidth=1, wraplength=320,
                 font=FONT_SM, justify="left", padx=8, pady=5).pack()

    def _hide(self, _e=None):
        if self._win:
            self._win.destroy()
            self._win = None


class ScrollableFrame(tk.Frame):
    """Vertically scrollable dark container; children go into `self.inner`.
    The mouse wheel is grabbed only while the pointer is over this frame."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, **kw)
        canvas = tk.Canvas(self, bg=BG, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(canvas, bg=BG)
        self._win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._sync_region)
        canvas.bind("<Configure>", self._canvas_resized)
        self.canvas = canvas
        self.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    def _content_fits(self):
        return self.inner.winfo_reqheight() <= self.canvas.winfo_height()

    def _sync_region(self, _e=None):
        # Anchor the region at (0,0); when the content fits the viewport
        # (e.g. every card collapsed) snap back to the top so nothing can be
        # scrolled out of view.
        self.canvas.configure(scrollregion=(0, 0, self.inner.winfo_reqwidth(),
                                            self.inner.winfo_reqheight()))
        if self._content_fits():
            self.canvas.yview_moveto(0)

    def _canvas_resized(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)
        self._sync_region()

    def _wheel(self, event):
        if self._content_fits():
            return          # nothing to scroll — ignore the wheel
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class ResultBox(tk.Frame):
    """Selectable text area with a coloured accent border for run output."""

    def __init__(self, master):
        super().__init__(master, bg=BORDER, padx=1, pady=1)
        self._text = tk.Text(self, height=4, bg="#12161f", fg=TEXT,
                             relief="flat", wrap="word", state="disabled",
                             font=FONT_SM, cursor="xterm",
                             insertbackground=TEXT)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)
        self._text.bind("<MouseWheel>", self._on_scroll)
        self._set("(no result yet)")

    def _on_scroll(self, event):
        # Scroll the box only while it still has content in that direction;
        # at its top/bottom (or when empty) let the event through so the
        # page keeps scrolling instead.
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
        self.configure(bg=GOOD if ok else BAD)
        self._text.configure(bg=GOOD_D if ok else BAD_D,
                             fg="#eafff3" if ok else "#ffecec")
        self._set(text if text else ("Done." if ok else "Error"))

    def reset(self):
        self.configure(bg=BORDER)
        self._text.configure(bg="#12161f", fg=TEXT)
        self._set("(no result yet)")


class SingleSelect:
    """Group of BooleanVars where setting one True clears the others."""

    def __init__(self):
        self.vars = []

    def make_var(self):
        v = tk.BooleanVar(value=False)

        def _enforce(*_):
            if v.get():
                for other in self.vars:
                    if other is not v:
                        other.set(False)
        v.trace_add("write", _enforce)
        self.vars.append(v)
        return v


# ===========================================================================
#  Collapsible operation card
# ===========================================================================
def build_card(parent, title, fields, mode="multi", single=None,
               order_values=None, with_result=False, est=True):
    """One collapsible card. mode: 'single' | 'multi' | 'order' | 'none'.
    Returns dict: frame, var, mode, ff, result, est_label, title, expand()."""
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    card = tk.Frame(outer, bg=CARD)
    card.pack(fill="both", expand=True)
    header = tk.Frame(card, bg=CARD)
    header.pack(fill="x", padx=8, pady=(7, 5))

    sec = {"mode": mode, "title": title, "var": None, "frame": outer}

    if mode == "single":
        var = single.make_var()
        cb = tk.Checkbutton(header, variable=var, bg=CARD, activebackground=CARD,
                            selectcolor=FIELD, fg=ACCENT, bd=0, highlightthickness=0)
        cb.pack(side="left")
        sec["var"] = var
    elif mode == "multi":
        var = tk.BooleanVar(value=False)
        tk.Checkbutton(header, variable=var, bg=CARD, activebackground=CARD,
                       selectcolor=FIELD, fg=ACCENT, bd=0,
                       highlightthickness=0).pack(side="left")
        sec["var"] = var
    elif mode == "order":
        var = tk.StringVar(value="")
        tk.Label(header, text="order", bg=CARD, fg=MUTED,
                 font=FONT_SM).pack(side="left")
        ttk.Combobox(header, textvariable=var, values=order_values,
                     state="readonly", width=3, font=FONT_SM
                     ).pack(side="left", padx=(3, 0))
        sec["var"] = var

    title_lbl = tk.Label(header, text=title, bg=CARD, fg=TEXT, font=FONT_B)
    title_lbl.pack(side="left", padx=8)

    est_lbl = None
    if est:
        est_lbl = tk.Label(header, text="", bg=CARD, fg=MUTED, font=FONT_IT)
        est_lbl.pack(side="right", padx=(4, 2))
    sec["est_label"] = est_lbl

    arrow = tk.Label(header, text="▸" if fields else "", bg=CARD, fg=MUTED,
                     font=FONT, cursor="hand2" if fields else "arrow")
    arrow.pack(side="right", padx=2)

    body = tk.Frame(card, bg=CARD)
    sec["ff"] = build_form(body, fields)
    sec["_body"] = body
    sec["_arrow"] = arrow
    sec["_expanded"] = False

    def expand(show=True):
        if not fields:
            return
        if show and not sec["_expanded"]:
            body.pack(fill="x", padx=8, pady=(0, 6), before=result_holder)
            arrow.configure(text="▾")
            sec["_expanded"] = True
        elif not show and sec["_expanded"]:
            body.pack_forget()
            arrow.configure(text="▸")
            sec["_expanded"] = False
    sec["expand"] = expand
    arrow.bind("<Button-1>", lambda e: expand(not sec["_expanded"]))

    result_holder = tk.Frame(card, bg=CARD)
    result_holder.pack(fill="x", padx=8, pady=(0, 6))
    sec["result"] = None
    if with_result:
        rb = ResultBox(result_holder)
        rb.pack(fill="x")
        sec["result"] = rb

    # Selecting a card auto-expands its parameters; title click toggles select.
    if mode in ("single", "multi"):
        title_lbl.configure(cursor="hand2")
        title_lbl.bind("<Button-1>", lambda e: sec["var"].set(not sec["var"].get()))
        sec["var"].trace_add(
            "write", lambda *_: expand(True) if sec["var"].get() else None)
    elif mode == "order":
        sec["var"].trace_add(
            "write", lambda *_: expand(True) if str(sec["var"].get()).strip() else None)
    return sec


def grid_cards(container, secs, columns):
    """Masonry layout: one packed sub-frame per column, so expanding a card
    only pushes down the cards in its own column — row neighbours are not
    stretched and no vertical gaps open up below them."""
    col_frames = []
    for c in range(columns):
        container.columnconfigure(c, weight=1, uniform="cards")
        f = tk.Frame(container, bg=BG)
        f.grid(row=0, column=c, sticky="new", padx=6)
        col_frames.append(f)
    for idx, sec in enumerate(secs):
        sec["frame"].pack(in_=col_frames[idx % columns], fill="x", pady=5)
        sec["frame"].lift()      # column frames were created after the cards


def card_selected(sec):
    return sec["mode"] in ("single", "multi") and bool(sec["var"].get())


def card_order(sec):
    s = str(sec["var"].get()).strip()
    return int(s) if s else None


# ===========================================================================
#  Time estimates  (base seconds at 1 M voxels, scaling exponent)
# ===========================================================================
_LARGE_VOXELS = 50_000_000
_TIME_ESTIMATES = {
    "CROP_VOLUME": (0.05, 1.0), "FILL_CLOSED_PORES": (0.5, 1.0),
    "TRIM_DISCONNECTED_VOXELS": (0.8, 1.0), "TRIM_FLOATING_SOLID": (0.5, 1.0),
    "TRIM_NONPERCOLATING_PATHS": (1.5, 1.0),
    "APPLY_CHORDS": (0.3, 1.0), "APPLY_CHORDS_3D": (1.0, 1.0),
    "APPLY_CHORDS_SOLID_PHASE": (0.3, 1.0), "APPLY_CHORDS_3D_SOLID_PHASE": (1.0, 1.0),
    "FIND_DISCONNECTED_VOXELS": (0.5, 1.0), "LOCAL_THICKNESS": (5.0, 1.1),
    "POROSIMETRY": (10.0, 1.1), "REGION_SIZE": (1.0, 1.0),
    "SNOW2_REGIONS": (15.0, 1.2),
    "BOND_NUMBER": (5.0, 1.1), "CHORD_COUNTS": (1.0, 1.0),
    "CHORD_COUNTS_SOLID_PHASE": (1.0, 1.0), "CHORD_LENGTH_DISTRIBUTION": (1.5, 1.0),
    "CHORD_LENGTH_DISTRIBUTION_SOLID_PHASE": (1.5, 1.0), "COORDINATION_NUMBER": (1.0, 1.0),
    "TWO_POINT_CORRELATION": (1.0, 1.0), "FIND_H": (30.0, 1.2),
    "FIND_POROSITY_THRESHOLD": (0.5, 1.0), "IS_PERCOLATING": (0.5, 1.0),
    "LINEAL_PATH_DISTRIBUTION": (3.0, 1.0), "MESH_SURFACE_AREA": (2.0, 1.0),
    "MESH_VOLUME": (2.0, 1.0), "PC_MAP_TO_PC_CURVE": (30.0, 1.2),
    "PERCOLATING_POROSITY": (0.5, 1.0), "PORE_SIZE_DISTRIBUTION": (10.0, 1.1),
    "POROSITY": (0.05, 1.0), "POROSITY_PROFILE": (0.2, 1.0),
    "RADIAL_DENSITY_DISTRIBUTION": (1.0, 1.0),
    "REGION_SURFACE_AREAS": (15.0, 1.2), "REGION_VOLUMES": (15.0, 1.2),
    "REGIONPROPS_3D": (5.0, 1.1), "REV_POROSITY": (5.0, 1.1),
    "SATN_PROFILE": (30.0, 1.2),
    "DRAINAGE": (30.0, 1.2), "IMBIBITION": (30.0, 1.2),
    "INJECTION": (45.0, 1.2), "TORTUOSITY": (20.0, 1.5),
    "Stokes flow": (18.0, 1.2), "Fickian diffusion": (18.0, 1.2),
    "Advection-diffusion": (22.0, 1.2), "Drainage": (17.0, 1.2),
    "Invasion percolation": (17.0, 1.2),
}


def _format_time(s):
    if s < 1:
        return "<1 s"
    if s < 60:
        return f"~{int(s)} s"
    if s < 3600:
        return f"~{s / 60:.0f} min"
    return f"~{s / 3600:.1f} h"


def est_time_text(title, n_voxels):
    entry = _TIME_ESTIMATES.get(title)
    if entry is None:
        return ""
    base_s, exp = entry
    return "est " + _format_time(base_s * (n_voxels / 1_000_000) ** exp)


# ===========================================================================
#  File dialogs & rename helper  (philosophy carried over from app_gui.py)
# ===========================================================================
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
    if ext == ".png":
        return [("PNG image", "*.png"), ("All files", "*.*")]
    return [("All files", "*.*")]


def ask_save_file(parent, default_name):
    path = filedialog.asksaveasfilename(
        parent=parent, initialfile=default_name,
        initialdir=OUTPUT_DIR_DEFAULT,
        defaultextension=os.path.splitext(default_name)[1] if default_name else "",
        filetypes=_filetypes_for(default_name), title="Save output file")
    if not path:
        return None, None
    return os.path.dirname(path), os.path.basename(path)


def rename_saved(out_dir, default_name, chosen_name):
    target = (chosen_name or "").strip() or default_name
    if not out_dir or not default_name or target == default_name:
        return
    old = os.path.join(out_dir, default_name)
    new = os.path.join(out_dir, target)
    if os.path.exists(old):
        try:
            os.replace(old, new)
        except OSError as e:
            print(f"Warning: could not rename {default_name!r} -> {target!r}: {e}")


# ===========================================================================
#  DICOM / RAW import helpers (ported from app_gui.py)
# ===========================================================================
def _dicom_read_force(path):
    import pydicom
    from pydicom.uid import ImplicitVRLittleEndian
    ds = pydicom.dcmread(path, force=True)
    if "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    return ds


def _dicom_voxel_size(ds):
    try:
        dy, dx = (float(v) for v in ds.PixelSpacing)
    except Exception:
        dx = dy = 1.0
    dz = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
    return dx, dy, dz


def _dicom_pixels(ds):
    arr = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr * slope + intercept


def _read_dicom_file(path):
    ds = _dicom_read_force(path)
    arr = _dicom_pixels(ds)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape}")
    return arr, _dicom_voxel_size(ds)


def _read_dicom_series(paths):
    datasets = [_dicom_read_force(p) for p in paths]
    try:
        datasets.sort(key=lambda d: int(d.InstanceNumber))
    except Exception:
        pass
    slices = [_dicom_pixels(ds) for ds in datasets]
    # each slice is (rows, cols) = (y, x); stack along z then swap to (x, y, z)
    vol = np.transpose(np.stack(slices, axis=-1), (1, 0, 2))
    return vol, _dicom_voxel_size(datasets[0])


# ===========================================================================
#  Numpy slice / PNG -> Tk PhotoImage
#  Pillow is used for smooth arbitrary scaling when available (it ships with
#  imageio/scikit-image, so it is normally present); otherwise fall back to
#  Tk's integer zoom/subsample.
# ===========================================================================
try:
    from PIL import Image as _PILImage
except Exception:  # noqa: BLE001
    _PILImage = None


_VIRIDIS_LUT = None


def _viridis_lut():
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is None:
        _VIRIDIS_LUT = (plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]
                        * 255).astype(np.uint8)
    return _VIRIDIS_LUT


def _rgb_from_plane(plane, vmin=None, vmax=None):
    """2D array -> (h, w, 3) uint8. Boolean planes use the PoreSpy colours;
    numeric planes use the viridis colormap (like the module plots), scaled
    to [vmin, vmax] (defaults: the plane's own range)."""
    h, w = plane.shape
    if plane.dtype == bool:
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[plane] = PORE_RGB
        rgb[~plane] = SOLID_RGB
        return rgb
    p = plane.astype(np.float64)
    if vmin is None or vmax is None:
        finite = p[np.isfinite(p)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        lo = lo if vmin is None else float(vmin)
        hi = hi if vmax is None else float(vmax)
    else:
        lo, hi = float(vmin), float(vmax)
    p = np.nan_to_num(p, nan=lo, posinf=hi, neginf=lo)
    rng = hi - lo
    idx = (np.clip((p - lo) / rng, 0.0, 1.0) * 255).astype(np.uint8) if rng > 0 \
        else np.zeros_like(p, dtype=np.uint8)
    return _viridis_lut()[idx]


def _photo_from_rgb(rgb):
    h, w = rgb.shape[:2]
    ppm = b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()
    return tk.PhotoImage(data=ppm)


def slice_to_photo(plane, max_w=430, max_h=None, vmin=None, vmax=None):
    """Convert a 2D array to a Tk PhotoImage scaled (up or down, keeping the
    aspect ratio) to fit inside max_w x max_h. Nearest-neighbour is used so
    voxels stay crisp."""
    if max_h is None:
        max_h = max_w
    # Draw with the origin at the bottom (like ParaView / matplotlib origin="lower")
    # so the first plane axis increases upward — gravity packings then visibly
    # settle downward instead of appearing upside-down.
    plane = np.asarray(plane)[::-1]
    rgb = _rgb_from_plane(plane, vmin, vmax)
    h, w = plane.shape
    scale = min(max_w / w, max_h / h)
    if _PILImage is not None and abs(scale - 1.0) > 1e-3:
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        im = _PILImage.fromarray(rgb).resize((tw, th), _PILImage.NEAREST)
        return _photo_from_rgb(np.asarray(im))
    img = _photo_from_rgb(rgb)
    big = max(h, w)
    target = min(max_w, max_h)
    if big < target:
        img = img.zoom(max(1, target // big))
    elif big > target:
        img = img.subsample(math.ceil(big / target))
    return img


def png_to_photo(png_bytes, target=None):
    img = tk.PhotoImage(data=png_bytes)
    if target and img.width() > target:
        img = img.subsample(math.ceil(img.width() / target))
    return img


def png_fit_photo(png_bytes, max_w, max_h):
    """PNG bytes -> PhotoImage scaled smoothly to fit max_w x max_h (keeping
    aspect). Falls back to integer subsampling without Pillow."""
    max_w, max_h = max(1, max_w), max(1, max_h)
    if _PILImage is not None:
        im = _PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = im.size
        scale = min(max_w / w, max_h / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        if (tw, th) != (w, h):
            im = im.resize((tw, th), _PILImage.LANCZOS)
        return _photo_from_rgb(np.asarray(im))
    img = tk.PhotoImage(data=png_bytes)
    f = max(math.ceil(img.width() / max_w), math.ceil(img.height() / max_h), 1)
    return img.subsample(f) if f > 1 else img


# ===========================================================================
#  Minimal .vti reader for files written by pyevtk's imageToVTK
#  (raw-appended encoding, UInt64 block headers, Fortran-ravelled arrays).
#  save_results_vtk() writes arrays in the app's (x, y, z) order directly
#  (pyevtk maps array axis 0/1/2 -> VTK X/Y/Z), so no transpose is needed on read.
# ===========================================================================
_VTK_DTYPES = {
    "Float64": "f8", "Float32": "f4",
    "Int64": "i8", "UInt64": "u8", "Int32": "i4", "UInt32": "u4",
    "Int16": "i2", "UInt16": "u2", "Int8": "i1", "UInt8": "u1",
}


def read_vti(path):
    """Return {name: 3D array in the app's (x, y, z) orientation}."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    m = re.search(rb'<AppendedData\s+encoding="raw"\s*>', data)
    if not m:
        raise ValueError("Unsupported .vti file (no raw appended data). "
                         "Only files written by this app are supported.")
    xml_part = data[:m.start()]
    raw = data[data.index(b"_", m.end()) + 1:]
    byte_order = "<"
    mbo = re.search(rb'byte_order="([^"]+)"', xml_part)
    if mbo and mbo.group(1) == b"BigEndian":
        byte_order = ">"
    mext = re.search(rb'WholeExtent="([^"]+)"', xml_part)
    if not mext:
        raise ValueError("No WholeExtent found in .vti file.")
    ext = [int(x) for x in mext.group(1).split()]
    # point data: n points per axis = extent_max - extent_min + 1
    shape = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
    arrays = {}
    for tag in re.finditer(rb"<DataArray\b[^>]*>", xml_part):
        t = tag.group(0)
        mname = re.search(rb'Name="([^"]*)"', t)
        mtype = re.search(rb'type="([^"]+)"', t)
        moff = re.search(rb'offset="\s*(\d+)\s*"', t)
        if not (mname and mtype and moff):
            continue
        np_t = _VTK_DTYPES.get(mtype.group(1).decode())
        if np_t is None:
            continue
        dtype = np.dtype(byte_order + np_t)
        off = int(moff.group(1))
        nbytes = struct.unpack_from(byte_order + "Q", raw, off)[0]
        n = nbytes // dtype.itemsize
        if n != shape[0] * shape[1] * shape[2]:
            continue                       # skip non point-data arrays
        arr = np.frombuffer(raw, dtype=dtype, count=n, offset=off + 8)
        arr = arr.reshape(shape, order="F")   # already in app (x, y, z) order
        vals = np.unique(arr)
        if vals.size <= 2 and set(vals.tolist()) <= {0.0, 1.0}:
            arr = arr.astype(bool)           # geometry-like field
        arrays[mname.group(1).decode() or f"array_{len(arrays)}"] = arr
    if not arrays:
        raise ValueError("No readable point-data arrays found in .vti file.")
    return arrays


# ===========================================================================
#  Parameter specifications  (built after the modules are imported)
# ===========================================================================
AXIS3 = ["0", "1", "2"]
AXIS_NONE = ["None", "0", "1", "2"]
XYZ = ["x", "y", "z"]
XYZ_ALL = ["x", "y", "z", "all"]

_TT_CONN = "max = corners count as connections; min = faces only"
_TT_SLICE = "Axis for the 2D visualization slice (0=x, 1=y, 2=z)"
_TT_SHOW = "Also render the 2D slice figure into the Plot Gallery"
_TT_VS = "Physical length of each voxel (m)"
_TT_XL = "Export results to Excel"
_TT_BINS = "Number of histogram bins"
_TT_LOG = "Use logarithmic x-axis"
_TT_SIGMA = "Fluid-fluid surface tension (N/m)"
_TT_THETA = "Contact angle through invading phase (degrees)"
_TT_G = "Gravitational acceleration (m/s²)"
_TT_RHO_NWP = "Non-wetting phase density (kg/m³)"
_TT_RHO_WP = "Wetting phase density (kg/m³)"
_TT_AXIS_SIM = "Inlet/outlet axis; None → all faces"
_TT_MIN_SIZE = "Minimum trapped cluster size to retain (voxels)"

SPEC = {}


def build_specs():
    """Fill SPEC with the operation tables. Must run after load_project_modules()."""
    SPEC["GEN"] = [
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

    SPEC["FILTER_GROUPS"] = [
        ("Solid altering filters", [
            ("CROP_VOLUME", filt.CROP_VOLUME, [
                F("x_min", "int_none", "0", tooltip="Start voxel along axis 0 (x), inclusive; blank = 0"),
                F("x_max", "int_none", "", tooltip="End voxel along axis 0 (x), EXCLUSIVE; blank = image extent. e.g. min 50, max 150 keeps 100 voxels"),
                F("y_min", "int_none", "0", tooltip="Start voxel along axis 1 (y), inclusive; blank = 0"),
                F("y_max", "int_none", "", tooltip="End voxel along axis 1 (y), EXCLUSIVE; blank = image extent"),
                F("z_min", "int_none", "0", tooltip="Start voxel along axis 2 (z), inclusive; blank = 0"),
                F("z_max", "int_none", "", tooltip="End voxel along axis 2 (z), EXCLUSIVE; blank = image extent. Crop 1000³+ images to ≤ ~300³ to avoid out-of-memory errors"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("FILL_CLOSED_PORES", filt.FILL_CLOSED_PORES, [
                F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("TRIM_DISCONNECTED_VOXELS", filt.TRIM_DISCONNECTED_VOXELS, [
                F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
                F("inlet_faces", "choice_str", "all", ["all", "x0", "x1", "y0", "y1", "z0", "z1"],
                  tooltip="Faces used as inlets: all, x0/x1, y0/y1, z0/z1"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("TRIM_FLOATING_SOLID", filt.TRIM_FLOATING_SOLID, [
                F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("TRIM_NONPERCOLATING_PATHS", filt.TRIM_NONPERCOLATING_PATHS, [
                F("perc_axis", "choice_int", "0", AXIS3, tooltip="Axis along which percolation is tested"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
        ]),
        ("Non-solid altering filters", [
            ("APPLY_CHORDS", filt.APPLY_CHORDS, [
                F("spacing", "int", "1", tooltip="Voxels between parallel chord lines"),
                F("axis", "choice_int", "1", AXIS3, tooltip="Axis along which chords are drawn"),
                F("trim_edges", "bool", False, tooltip="Remove chords that touch the domain boundary"),
                F("label", "bool", True, tooltip="Assign a unique label to each chord"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("APPLY_CHORDS_3D", filt.APPLY_CHORDS_3D, [
                F("spacing", "int", "2", tooltip="Voxels between parallel chord lines"),
                F("trim_edges", "bool", True, tooltip="Remove chords that touch the domain boundary"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("APPLY_CHORDS_SOLID_PHASE", filt.APPLY_CHORDS_SOLID_PHASE, [
                F("spacing", "int", "1", tooltip="Voxels between parallel chord lines"),
                F("axis", "choice_int", "1", AXIS3, tooltip="Axis along which chords are drawn"),
                F("trim_edges", "bool", False, tooltip="Remove chords that touch the domain boundary"),
                F("label", "bool", True, tooltip="Assign a unique label to each chord"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("APPLY_CHORDS_3D_SOLID_PHASE", filt.APPLY_CHORDS_3D_SOLID_PHASE, [
                F("spacing", "int", "2", tooltip="Voxels between parallel chord lines"),
                F("trim_edges", "bool", True, tooltip="Remove chords that touch the domain boundary"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("FIND_DISCONNECTED_VOXELS", filt.FIND_DISCONNECTED_VOXELS, [
                F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
                F("show_3d", "bool", True, tooltip="Render the 3D view of disconnected voxels")]),
            ("LOCAL_THICKNESS", filt.LOCAL_THICKNESS, [
                F("sizes", "int", "10", tooltip="Number of size classes for the thickness map"),
                F("method", "choice_str", "dt", ["dt", "conv", "bf", "imj"],
                  tooltip="Algorithm: dt (fastest), conv, bf, imj"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("POROSIMETRY", filt.POROSIMETRY, [
                F("sizes", "int", "15", tooltip="Number of sphere diameters to test"),
                F("method", "choice_str", "dt", ["dt", "conv", "dsi"],
                  tooltip="Algorithm: dt, conv, dsi"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW),
                F("inv_axis", "choice_int_none", "None", AXIS_NONE,
                  tooltip="Invasion axis; None → invasion from all faces")]),
            ("REGION_SIZE", filt.REGION_SIZE, [
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
            ("SNOW2_REGIONS", filt.SNOW2_REGIONS, [
                F("voxel_size", "float", "1", tooltip="Physical length of each voxel (m) — used by snow2 for pore geometry"),
                F("slice", "choice_int", "0", AXIS3, tooltip=_TT_SLICE),
                F("show_slice", "bool", True, tooltip=_TT_SHOW)]),
        ]),
    ]

    SPEC["METRICS"] = [
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
            F("srange", "list_float", "0.01,0.95", tooltip="Saturation range [min,max] to search over"),
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

    _drain_imb = [
        F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
        F("theta", "float", "180", tooltip=_TT_THETA),
        F("g", "float", "9.81", tooltip=_TT_G),
        F("rho_nwp", "float", "1000", tooltip=_TT_RHO_NWP),
        F("rho_wp", "float", "0", tooltip=_TT_RHO_WP),
        F("voxel_size", "float", "1e-4", tooltip=_TT_VS),
        F("axis", "choice_int_none", "None", AXIS_NONE, tooltip=_TT_AXIS_SIM),
        F("min_size", "int", "5", tooltip=_TT_MIN_SIZE),
        F("steps", "int_none", "50",
          tooltip="Number of pressure steps. WARNING: empty → one pass per "
                  "unique capillary pressure value; with gravity (g>0) almost "
                  "every voxel is unique, so the run takes hours instead of "
                  "minutes. Keep ~25–100 unless you really need every step"),
        F("save_excel", "bool", False, tooltip="Export capillary pressure–saturation curve to Excel"),
    ]
    SPEC["PSIM"] = [
        ("DRAINAGE", psim.DRAINAGE, list(_drain_imb)),
        ("IMBIBITION", psim.IMBIBITION, list(_drain_imb)),
        ("INJECTION", psim.INJECTION, [
            F("sigma", "float", "0.01", tooltip=_TT_SIGMA),
            F("theta", "float", "180", tooltip=_TT_THETA),
            F("g", "float", "9.81", tooltip=_TT_G),
            F("rho_nwp", "float", "1000", tooltip=_TT_RHO_NWP),
            F("rho_wp", "float", "0", tooltip=_TT_RHO_WP),
            F("voxel_size", "float", "1e-4", tooltip=_TT_VS),
            F("axis", "choice_int_none", "None", AXIS_NONE, tooltip=_TT_AXIS_SIM),
            F("min_size", "int", "5", tooltip=_TT_MIN_SIZE),
            F("conn", "choice_str", "max", ["max", "min"], tooltip=_TT_CONN),
            F("maxiter", "int", "10000", tooltip="Maximum invasion iterations"),
            F("method", "choice_str", "qbip", ["qbip", "ibip"],
              tooltip="qbip = faster (binary heap); ibip = image-based"),
            F("save_excel", "bool", False, tooltip="Export capillary pressure–saturation curve to Excel")]),
        ("TORTUOSITY", psim.TORTUOSITY, [
            F("axis", "choice_int", "0", AXIS3, tooltip="Axis along which tortuosity is calculated"),
            F("crop_size", "int_none", "", label="crop size (opt.)",
              tooltip="Crop the centre cube to this side length before solving. "
                      "tortuosity_fd needs ~22 GB for a 1000³ image; use ≤300 for large volumes")]),
    ]

    SPEC["PNM_PHASES"] = [
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

    SPEC["PNM_SIMS"] = [
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

    SPEC["ALWAYS_SAVES"] = frozenset([
        gen.BLOBS, gen.CYLINDERS, gen.OVERLAPPING_SPHERES,
        gen.POLYDISPERSE_SPHERES, gen.PSEUDO_ELECTROSTATIC_PACKING,
        gen.PSEUDO_GRAVITY_PACKING, gen.VORONOI_EDGES,
        psim.DRAINAGE, psim.IMBIBITION, psim.INJECTION,
        met.REGIONPROPS_3D,
        opnm.simulate_stokes_flow, opnm.simulate_fickian_diffusion,
        opnm.simulate_advection_diffusion, opnm.simulate_drainage,
        opnm.simulate_invasion_percolation,
    ])
    SPEC["SAVE_FILE"] = {
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
    }
    SPEC["PNM_ABBR"] = {
        "Stokes flow": "SF", "Fickian diffusion": "FD",
        "Advection-diffusion": "AD", "Drainage": "Dr",
        "Invasion percolation": "IP",
    }


def get_save_file(sec):
    """Return the filename this card's function will write, or None."""
    func = sec.get("func")
    if func in SPEC["ALWAYS_SAVES"]:
        return SPEC["SAVE_FILE"].get(func)
    if "save_excel" in sec.get("ff", {}):
        try:
            if sec["ff"]["save_excel"].get():
                return SPEC["SAVE_FILE"].get(func)
        except Exception:
            pass
    return None


# ===========================================================================
#  The application
# ===========================================================================
class FableApp:
    NAV = [
        ("home",    "⌂  Home"),
        ("generate","◆  Generate"),
        ("import",  "⇩  Import"),
        ("filters", "▤  Filters"),
        ("metrics", "∑  Metrics"),
        ("psim",    "≋  PoreSpy sims"),
        ("opnm",    "🕸  OpenPNM sims"),
        ("plots",   "📊  Plots"),
        ("log",     "≡  Log"),
    ]

    def __init__(self, root):
        self.root = root
        self.image = None             # current working image (bool volume)
        self.image_unfiltered = None  # as it left the generator / import step
        self.image_name = None
        self.pn = None                # cached OpenPNM network
        self.pn_image_id = None
        self.pn_voxel_size = None

        self.gallery = []             # [{title, png, photo, frame}]
        self._action_buttons = []
        self._est_registry = []       # (sec, title) pairs to refresh
        self._busy_count = 0
        self._busy_since = None
        self._busy_title = ""

        self.job_q = queue.Queue()
        self.res_q = queue.Queue()
        self._job_gen = 0             # bumped by Cancel: stale jobs are dropped
        self._worker_thread = None
        self._start_worker()

        self._build_layout()
        self._build_pages()
        self.show_page("home")
        self.root.after(100, self._poll_results)
        self.root.after(500, self._tick_busy)
        self.log(f"{APP_TITLE} started. Output defaults to: "
                 + OUTPUT_DIR_DEFAULT)

    # ---------------- layout ----------------------------------------------
    def _build_layout(self):
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.minsize(1080, 700)

        # Sidebar
        side = tk.Frame(self.root, bg=SIDE, width=210)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self._side_icon = load_app_icon(56)
        if self._side_icon is not None:
            tk.Label(side, image=self._side_icon, bg=SIDE).pack(pady=(18, 4))
        tk.Label(side, text=APP_TITLE, bg=SIDE, fg=CRIMSON,
                 font=("Segoe UI", 13, "bold"), wraplength=180,
                 justify="center").pack(pady=(0, 16), padx=8)

        self._nav_buttons = {}
        for key, label in self.NAV:
            b = tk.Label(side, text=label, bg=SIDE, fg=MUTED, font=FONT_B,
                         anchor="w", padx=22, pady=9, cursor="hand2")
            b.pack(fill="x")
            b.bind("<Button-1>", lambda e, k=key: self.show_page(k))
            b.bind("<Enter>", lambda e, w=b: w.configure(bg="#141925")
                   if w is not self._nav_buttons.get(self._current_page) else None)
            b.bind("<Leave>", lambda e, w=b: w.configure(bg=SIDE)
                   if w is not self._nav_buttons.get(self._current_page) else None)
            self._nav_buttons[key] = b

        tk.Frame(side, bg=SIDE).pack(fill="both", expand=True)
        tk.Label(side, text="WORKING IMAGE", bg=SIDE, fg=MUTED,
                 font=("Segoe UI", 8, "bold"), anchor="w", padx=22
                 ).pack(fill="x")
        self._img_side_prev = tk.Label(side, bg=SIDE)   # mid-slice thumbnail
        self._img_side_prev.pack(pady=(4, 0))
        self._img_side_info = tk.Label(side, text="no image", bg=SIDE, fg=MUTED,
                                       font=FONT_SM, justify="left", anchor="w",
                                       padx=22, wraplength=170)
        self._img_side_info.pack(fill="x", pady=(0, 14))

        # Main column: page container + status bar
        main = tk.Frame(self.root, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        self._page_container = tk.Frame(main, bg=BG)
        self._page_container.pack(fill="both", expand=True)

        status = tk.Frame(main, bg=SIDE, height=34)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        self._status_lbl = tk.Label(status, text="Ready", bg=SIDE, fg=MUTED,
                                    font=FONT_SM, anchor="w", padx=14)
        self._status_lbl.pack(side="left", fill="x", expand=True)
        self._status_img = tk.Label(status, text="", bg=SIDE, fg=MUTED,
                                    font=FONT_SM, anchor="e", padx=14)
        self._status_img.pack(side="right")
        self._progress = ttk.Progressbar(status, mode="indeterminate", length=140)
        self._cancel_btn = tk.Button(
            status, text="✕ Cancel", command=self._cancel_ops, bg=BAD_D,
            fg="#ffecec", activebackground=BAD, activeforeground="#ffffff",
            relief="flat", font=FONT_SM, bd=0, padx=10, pady=1, cursor="hand2")
        hover(self._cancel_btn, BAD_D, BAD)

        self.pages = {}
        self._current_page = "home"

    def _make_page(self, key, title, subtitle):
        page = tk.Frame(self._page_container, bg=BG)
        page.place(relwidth=1, relheight=1)
        head = tk.Frame(page, bg=BG)
        head.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(head, text=title, bg=BG, fg=TEXT, font=FONT_H1).pack(anchor="w")
        if subtitle:
            sub = tk.Label(head, text=subtitle, bg=BG, fg=MUTED, font=FONT,
                           wraplength=980, justify="left")
            sub.pack(anchor="w", pady=(2, 0), fill="x")
            # Re-wrap with the window so text is never cut off when the app
            # is not maximised.
            head.bind("<Configure>",
                      lambda e, s=sub: s.configure(
                          wraplength=max(300, e.width - 10)))
        self.pages[key] = page
        return page

    def show_page(self, key):
        for k, b in self._nav_buttons.items():
            b.configure(bg=SIDE, fg=MUTED)
        self._nav_buttons[key].configure(bg=CARD, fg=TEXT)
        self._current_page = key
        self.pages[key].lift()
        self._refresh_estimates()
        if key == "home":
            self._refresh_home()

    # ---------------- background worker -----------------------------------
    def _start_worker(self):
        gen = self._job_gen
        self._worker_thread = threading.Thread(
            target=self._worker, args=(gen,), daemon=True)
        self._worker_thread.start()

    def _worker(self, my_gen):
        while True:
            job = self.job_q.get()
            if job.get("gen", 0) != self._job_gen:
                continue                       # job was cancelled while queued
            buf_out, buf_err = io.StringIO(), io.StringIO()
            ok, result, err = True, None, None
            t0 = time.perf_counter()
            try:
                if job.get("out_dir"):
                    os.chdir(job["out_dir"])
                with contextlib.redirect_stdout(buf_out), \
                        contextlib.redirect_stderr(buf_err):
                    result = job["func"](**job.get("kwargs", {}))
            except Exception as e:  # noqa: BLE001 — surface every failure to the UI
                ok = False
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                tail = _ANSI_RE.sub("", buf_err.getvalue()).strip()[-400:]
                if tail:
                    err += "\n---\n" + tail
                traceback.print_exc()
            except BaseException:              # KeyboardInterrupt from Cancel
                return                         # abandon this worker thread
            finally:
                elapsed = time.perf_counter() - t0
                # restore the app's default cwd (a stale, cancelled worker
                # must not touch it — a fresh worker owns it now)
                if my_gen == self._job_gen:
                    try:
                        os.chdir(OUTPUT_DIR_DEFAULT)
                    except OSError:
                        pass
            if my_gen != self._job_gen:
                return                         # cancelled: die quietly
            figs = [] if job.get("no_figs") else self._snapshot_figures()
            out = _ANSI_RE.sub("", buf_out.getvalue()).strip()
            self.res_q.put((job, ok, result, out, err, elapsed, figs))

    def _cancel_ops(self):
        """Cancel the running operation and drop everything still queued.
        Python cannot forcibly kill a thread that is inside compiled C code
        (numpy/porespy), so the computation may keep burning CPU in the
        background for a while — but its results are discarded and the UI is
        immediately released."""
        if self._busy_count == 0:
            return
        title = self._busy_title
        self._job_gen += 1                      # everything older is stale
        while True:                             # drop queued jobs
            try:
                self.job_q.get_nowait()
            except queue.Empty:
                break
        # Ask the worker to stop as soon as it returns to Python bytecode.
        t = self._worker_thread
        if t is not None and t.is_alive():
            try:
                import ctypes
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(t.ident), ctypes.py_object(KeyboardInterrupt))
            except Exception:  # noqa: BLE001
                pass
        self._start_worker()                    # fresh worker for future jobs
        # Release the UI.
        self._busy_count = 0
        self._busy_since = None
        self._progress.stop()
        self._progress.pack_forget()
        self._cancel_btn.pack_forget()
        self._set_actions_enabled(True)
        self._status_lbl.configure(text="Ready")
        try:
            os.chdir(OUTPUT_DIR_DEFAULT)
        except OSError:
            pass
        self.log(f"✋ Cancelled: {title}. Queued operations were dropped; the "
                 "running computation may continue in the background for a "
                 "moment, but its result is discarded.", "err")

    @staticmethod
    def _snapshot_figures():
        """Capture every open figure as a PNG (for the gallery thumbnails)
        AND keep the live Figure object, so clicking a plot can open a real
        matplotlib viewer that re-renders sharply at any zoom level (the way
        the old app_gui native windows did)."""
        figs = []
        for num in list(plt.get_fignums()):
            try:
                fig = plt.figure(num)
                title = ""
                try:
                    title = fig.canvas.manager.get_window_title() or ""
                except Exception:
                    pass
                if not title or title.startswith("Figure"):
                    for ax in fig.get_axes():
                        if ax.get_title():
                            title = ax.get_title()
                            break
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110,
                            facecolor=fig.get_facecolor(),
                            bbox_inches="tight")
                figs.append((title or f"Figure {num}", buf.getvalue(), fig))
            except Exception:
                pass
        plt.close("all")     # drops pyplot's refs; our fig objects stay alive
        return figs

    def submit(self, title, func, kwargs=None, out_dir=None, on_done=None,
               no_figs=False):
        """Queue one operation for the worker thread."""
        self._busy_count += 1
        if self._busy_count == 1:
            self._busy_since = time.perf_counter()
            self._progress.pack(side="right", padx=8, pady=6)
            self._progress.start(14)
            self._cancel_btn.pack(side="right", padx=(8, 0), pady=4)
            self._set_actions_enabled(False)
        self._busy_title = title
        self.job_q.put({"title": title, "func": func, "kwargs": kwargs or {},
                        "out_dir": out_dir, "on_done": on_done,
                        "no_figs": no_figs, "gen": self._job_gen})

    def _poll_results(self):
        try:
            while True:
                job, ok, result, out, err, elapsed, figs = self.res_q.get_nowait()
                if job.get("gen", 0) != self._job_gen:
                    continue        # result of a cancelled run — discard
                self._busy_count = max(0, self._busy_count - 1)
                if self._busy_count == 0:
                    self._progress.stop()
                    self._progress.pack_forget()
                    self._cancel_btn.pack_forget()
                    self._set_actions_enabled(True)
                    self._status_lbl.configure(text="Ready")
                    self._busy_since = None
                for title, png, fobj in figs:
                    self.add_plot(f"{job['title']} — {title}"
                                  if title else job['title'], png, fig=fobj)
                stamp = f"{job['title']}  ({elapsed:.1f} s)"
                if ok:
                    self.log("✔ " + stamp + (("\n" + out) if out else ""), "ok")
                else:
                    self.log("✘ " + stamp + "\n" + (err or ""), "err")
                if job.get("on_done"):
                    try:
                        job["on_done"](ok, result, out, err, elapsed, figs)
                    except Exception:
                        traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _tick_busy(self):
        if self._busy_since is not None:
            t = time.perf_counter() - self._busy_since
            self._status_lbl.configure(
                text=f"⏳  {self._busy_title} …  {t:.0f} s")
        self.root.after(500, self._tick_busy)

    def register_action(self, btn):
        self._action_buttons.append(btn)

    def _set_actions_enabled(self, enabled):
        for b in self._action_buttons:
            try:
                b.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    # ---------------- shared state helpers ---------------------------------
    def set_image(self, im, name, as_original=True):
        self.image = im
        if as_original:
            self.image_unfiltered = im
        self.image_name = name
        self.pn = None
        self._update_image_info()
        if hasattr(self, "slice_viewer"):
            if im is not None:
                self.slice_viewer.register_volume(name or "image", im)
            self.slice_viewer.set_image(im)

    def _update_image_info(self):
        if self.image is None:
            txt = "no image"
            self._status_img.configure(text="No working image", fg=MUTED)
            self._img_side_prev.configure(image="")
            self._img_side_prev._photo = None
        else:
            try:      # small mid-slice preview so the working image is
                mid = self.image.shape[2] // 2          # always recognisable
                photo = slice_to_photo(self.image[:, :, mid], 150, 110)
                self._img_side_prev.configure(image=photo)
                self._img_side_prev._photo = photo
            except Exception:  # noqa: BLE001
                pass
            sh = self.image.shape
            total = int(np.prod(sh))
            poro = float(self.image.sum()) / total
            txt = (f"{self.image_name or 'image'}\n{sh[0]}×{sh[1]}×{sh[2]}"
                   f"  ({total/1e6:.1f} M vox)\nporosity {poro:.3f}")
            bar = (f"{self.image_name or 'image'}   {sh[0]}×{sh[1]}×{sh[2]}"
                   f"   φ={poro:.3f}")
            if total > _LARGE_VOXELS:
                bar += "   ⚠ large image — crop before heavy operations"
                self._status_img.configure(text=bar, fg=WARN)
            else:
                self._status_img.configure(text=bar, fg=MUTED)
        self._img_side_info.configure(text=txt)

    def require_image(self):
        if self.image is None:
            messagebox.showwarning(
                "No image", "No working image yet.\nGenerate or import one first.",
                parent=self.root)
            return False
        return True

    def _refresh_estimates(self):
        n = int(np.prod(self.image.shape)) if self.image is not None else 1_000_000
        for sec, title in self._est_registry:
            if sec["est_label"] is not None:
                sec["est_label"].configure(text=est_time_text(title, n))

    def _register_cards(self, secs):
        for sec in secs:
            self._est_registry.append((sec, sec["title"]))

    # ---------------- logging ----------------------------------------------
    def log(self, msg, kind="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] ", "mut")
        self._log_text.insert("end", msg.rstrip() + "\n\n", kind)
        self._log_text.configure(state="disabled")
        self._log_text.see("end")

    # ---------------- plot gallery ------------------------------------------
    def add_plot(self, title, png_bytes, fig=None):
        try:
            photo = png_to_photo(png_bytes, target=390)
        except tk.TclError:
            return
        item = {"title": title, "png": png_bytes, "photo": photo, "fig": fig}
        card = tk.Frame(self._gallery_grid, bg=BORDER, padx=1, pady=1)
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="both", expand=True)
        top = tk.Frame(inner, bg=CARD)
        top.pack(fill="x", padx=6, pady=(5, 2))
        tk.Label(top, text=title, bg=CARD, fg=TEXT, font=FONT_SM,
                 anchor="w", wraplength=330, justify="left"
                 ).pack(side="left", fill="x", expand=True)
        x = tk.Label(top, text="✕", bg=CARD, fg=MUTED, cursor="hand2", font=FONT_SM)
        x.pack(side="right")
        s = tk.Label(top, text="💾", bg=CARD, fg=MUTED, cursor="hand2", font=FONT_SM)
        s.pack(side="right", padx=6)
        img_lbl = tk.Label(inner, image=photo, bg=CARD, cursor="hand2")
        img_lbl.pack(padx=6, pady=(0, 6))
        item["frame"] = card
        x.bind("<Button-1>", lambda e: self._remove_plot(item))
        s.bind("<Button-1>", lambda e: self._save_plot(item))
        img_lbl.bind("<Button-1>", lambda e: self._zoom_plot(item))
        self.gallery.append(item)
        if len(self.gallery) > 48:                    # cap memory
            self._remove_plot(self.gallery[0])
        self._regrid_gallery()
        self._update_plots_nav()

    def _remove_plot(self, item):
        if item in self.gallery:
            self.gallery.remove(item)
            item["frame"].destroy()
            self._regrid_gallery()
            self._update_plots_nav()

    def _clear_plots(self):
        for item in list(self.gallery):
            self._remove_plot(item)

    def _regrid_gallery(self):
        cols = 3
        for c in range(cols):
            self._gallery_grid.columnconfigure(c, weight=1)
        for idx, item in enumerate(self.gallery):
            r, c = divmod(idx, cols)
            item["frame"].grid(row=r, column=c, sticky="n", padx=6, pady=6)

    def _update_plots_nav(self):
        n = len(self.gallery)
        self._nav_buttons["plots"].configure(
            text=f"📊  Plots ({n})" if n else "📊  Plots")

    def _save_plot(self, item):
        safe = re.sub(r'[^\w\-. ]', '_', item["title"])[:60] or "plot"
        path = filedialog.asksaveasfilename(
            parent=self.root, initialfile=safe + ".png",
            initialdir=OUTPUT_DIR_DEFAULT, defaultextension=".png",
            filetypes=_filetypes_for("x.png"), title="Save plot as PNG")
        if path:
            with open(path, "wb") as f:
                f.write(item["png"])
            self.log(f"Plot saved: {path}")

    def _zoom_plot(self, item):
        """Open a plot. When the live matplotlib figure is available, embed a
        real interactive canvas with the navigation toolbar (zoom-to-rect,
        pan, home, save) — vector re-rendering keeps lines sharp at any zoom,
        like the native windows of the old app_gui. Falls back to a PNG
        viewer with drag-zoom otherwise."""
        if item.get("fig") is not None:
            try:
                self._open_interactive_plot(item)
                return
            except Exception:  # noqa: BLE001 — fall back to the PNG viewer
                traceback.print_exc()
        win = tk.Toplevel(self.root)
        win.title(item["title"])
        win.configure(bg="#0b0e14")
        if _PILImage is not None:
            src = _PILImage.open(io.BytesIO(item["png"])).convert("RGB")
            sw, sh = src.size
        else:
            src = None
            probe = tk.PhotoImage(data=item["png"])
            sw, sh = probe.width(), probe.height()
        w = min(sw + 20, self.root.winfo_screenwidth() - 100)
        h = min(sh + 60, self.root.winfo_screenheight() - 140)
        win.geometry(f"{w}x{h}")
        win.minsize(320, 240)

        state = {"view": (0.0, 0.0, float(sw), float(sh)),  # source coords
                 "after": None, "photo": None,
                 "disp": (0, 0, 1.0),                        # ox, oy, scale
                 "press": None}

        bar = tk.Frame(win, bg=SIDE)
        bar.pack(side="bottom", fill="x")
        canvas = tk.Canvas(win, bg="#0b0e14", highlightthickness=0,
                           cursor="tcross" if src is not None else "arrow")
        canvas.pack(fill="both", expand=True)

        def render():
            state["after"] = None
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            if cw < 20 or ch < 20:
                return
            x0, y0, x1, y1 = state["view"]
            vw, vh = x1 - x0, y1 - y0
            scale = min((cw - 8) / vw, (ch - 8) / vh)
            tw, th = max(1, int(vw * scale)), max(1, int(vh * scale))
            try:
                if src is not None:
                    crop = src.crop((int(x0), int(y0),
                                     max(int(x0) + 1, int(round(x1))),
                                     max(int(y0) + 1, int(round(y1)))))
                    resample = (_PILImage.LANCZOS if scale < 3
                                else _PILImage.NEAREST)
                    photo = _photo_from_rgb(
                        np.asarray(crop.resize((tw, th), resample)))
                else:
                    photo = png_fit_photo(item["png"], cw - 8, ch - 8)
                    tw, th = photo.width(), photo.height()
            except tk.TclError:
                return
            ox, oy = (cw - tw) // 2, (ch - th) // 2
            state["photo"] = photo               # keep a reference
            state["disp"] = (ox, oy, scale)
            canvas.delete("img")
            canvas.create_image(ox, oy, image=photo, anchor="nw", tags="img")

        def schedule_render(_e=None):
            if state["after"] is not None:
                win.after_cancel(state["after"])
            state["after"] = win.after(60, render)

        def reset(_e=None):
            state["view"] = (0.0, 0.0, float(sw), float(sh))
            render()

        def on_press(e):
            state["press"] = (e.x, e.y)
            canvas.delete("selrect")

        def on_move(e):
            if state["press"] is None:
                return
            px, py = state["press"]
            canvas.delete("selrect")
            canvas.create_rectangle(px, py, e.x, e.y, outline=ACCENT,
                                    dash=(4, 3), width=1, tags="selrect")

        def on_release(e):
            if state["press"] is None or src is None:
                return
            px, py = state["press"]
            state["press"] = None
            canvas.delete("selrect")
            if abs(e.x - px) < 12 or abs(e.y - py) < 12:
                return                            # too small — ignore
            ox, oy, scale = state["disp"]
            vx0, vy0 = state["view"][0], state["view"][1]
            sx0 = vx0 + (min(px, e.x) - ox) / scale
            sy0 = vy0 + (min(py, e.y) - oy) / scale
            sx1 = vx0 + (max(px, e.x) - ox) / scale
            sy1 = vy0 + (max(py, e.y) - oy) / scale
            sx0 = max(0.0, min(sx0, sw - 2))
            sy0 = max(0.0, min(sy0, sh - 2))
            sx1 = max(sx0 + 2, min(sx1, float(sw)))
            sy1 = max(sy0 + 2, min(sy1, float(sh)))
            state["view"] = (sx0, sy0, sx1, sy1)
            render()

        if src is not None:
            make_button(bar, "⤺  Reset zoom", reset).pack(side="left",
                                                          padx=8, pady=4)
            tk.Label(bar, text="drag a region to zoom in · double-click to "
                               "reset", bg=SIDE, fg=MUTED,
                     font=FONT_SM).pack(side="left", padx=8)
            canvas.bind("<ButtonPress-1>", on_press)
            canvas.bind("<B1-Motion>", on_move)
            canvas.bind("<ButtonRelease-1>", on_release)
            canvas.bind("<Double-Button-1>", reset)
        canvas.bind("<Configure>", schedule_render)
        win.after(30, render)

    def _open_interactive_plot(self, item):
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        fig = item["fig"]
        win = tk.Toplevel(self.root)
        win.title(item["title"])
        win.configure(bg=BG)
        w = min(int(fig.get_figwidth() * fig.dpi) + 30,
                self.root.winfo_screenwidth() - 100)
        h = min(int(fig.get_figheight() * fig.dpi) + 90,
                self.root.winfo_screenheight() - 140)
        win.geometry(f"{max(w, 420)}x{max(h, 320)}")
        win.minsize(420, 320)
        canvas = FigureCanvasTkAgg(fig, master=win)
        try:
            toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        except TypeError:                      # older matplotlib
            toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        tk.Label(win, text="🔍 toolbar: zoom into a rectangle · pan · ⌂ reset "
                           "· 💾 save", bg=BG, fg=MUTED,
                 font=FONT_SM).pack(side="bottom", fill="x")
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

        def _sync_size():
            # Match the figure to the widget's real, post-layout size — the
            # first draw otherwise happens at the pre-layout size, leaving
            # the plot off-centre until the user manually resizes the window.
            wpx, hpx = widget.winfo_width(), widget.winfo_height()
            if wpx > 20 and hpx > 20:
                fig.set_size_inches(wpx / fig.dpi, hpx / fig.dpi,
                                    forward=False)
            canvas.draw_idle()

        win.update_idletasks()
        _sync_size()
        win.after(150, _sync_size)   # again once the window is truly mapped
    def _build_pages(self):
        self._build_log_page()      # first: other pages log during construction
        self._build_plots_page()
        self._build_home_page()
        self._build_generate_page()
        self._build_import_page()
        self._build_filters_page()
        self._build_metrics_page()
        self._build_psim_page()
        self._build_opnm_page()

    # ---- Home --------------------------------------------------------------
    def _build_home_page(self):
        page = self._make_page(
            "home", "Home",
            "Overview of the working volume. Generate or import an image, then walk "
            "the pipeline: Filters → Metrics → PoreSpy sims → OpenPNM sims.")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=10)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.slice_viewer = SliceViewer(left, self)
        self.slice_viewer.pack(fill="both", expand=True)

        right = tk.Frame(body, bg=BG, width=300)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)

        info_card = tk.Frame(right, bg=CARD, padx=14, pady=12)
        info_card.pack(fill="x")
        tk.Label(info_card, text="Working image", bg=CARD, fg=TEXT,
                 font=FONT_H2).pack(anchor="w")
        self._home_info = tk.Label(info_card, text="—", bg=CARD, fg=MUTED,
                                   font=FONT, justify="left", anchor="w")
        self._home_info.pack(anchor="w", pady=(6, 2))

        btns = tk.Frame(right, bg=BG)
        btns.pack(fill="x", pady=12)
        b1 = make_button(btns, "◆  Generate an image", lambda: self.show_page("generate"))
        b1.pack(fill="x", pady=3)
        b2 = make_button(btns, "⇩  Import an image", lambda: self.show_page("import"))
        b2.pack(fill="x", pady=3)
        b3 = make_button(btns, "💾  Save volume as .tif", self._save_volume)
        b3.pack(fill="x", pady=3)
        b4 = make_button(btns, "↺  Reset to unfiltered image", self._reset_image)
        b4.pack(fill="x", pady=3)
        for b in (b3, b4):
            self.register_action(b)

        tip = tk.Frame(right, bg=CARD, padx=14, pady=12)
        tip.pack(fill="x", pady=(4, 0))
        tk.Label(tip, text="Tips", bg=CARD, fg=TEXT, font=FONT_H2).pack(anchor="w")
        tk.Label(tip, bg=CARD, fg=MUTED, font=FONT_SM, justify="left", anchor="w",
                 wraplength=250, text=(
                     "• Yellow = pore/void, violet = solid (PoreSpy convention).\n"
                     "• Every figure a run produces lands in the Plots page.\n"
                     "• Hover any parameter name for an explanation.\n"
                     "• Large volumes (>50 M voxels): crop first in Filters.\n"
                     "• All printed output is kept in the Log page.")).pack(anchor="w", pady=(4, 0))

    def _refresh_home(self):
        if self.image is None:
            self._home_info.configure(text="No image yet.\nGenerate or import one.")
        else:
            sh = self.image.shape
            total = int(np.prod(sh))
            poro = float(self.image.sum()) / total
            filtered = (self.image_unfiltered is not None
                        and self.image is not self.image_unfiltered)
            self._home_info.configure(text=(
                f"name:  {self.image_name or '—'}\n"
                f"shape:  {sh[0]} × {sh[1]} × {sh[2]}\n"
                f"voxels:  {total/1e6:.1f} M\n"
                f"porosity:  {poro:.4f}\n"
                f"state:  {'filtered' if filtered else 'original'}"))
        self.slice_viewer.render()

    def _save_volume(self):
        if not self.require_image():
            return
        default = f"{self.image_name or 'volume'}.tif"
        out_dir, name = ask_save_file(self.root, default)
        if not out_dir:
            return
        path = os.path.join(out_dir, name or default)

        def _save():
            import imageio.v2 as iio
            arr = np.transpose(self.image.astype(np.uint8) * 255, (2, 1, 0))
            iio.volsave(path, arr)
            print(f"Volume saved: {path}")
        self.submit("Saving volume", _save, on_done=lambda ok, *a: None,
                    no_figs=True)

    def _reset_image(self):
        if self.image_unfiltered is None:
            messagebox.showinfo("Reset", "No original image stored.", parent=self.root)
            return
        self.image = self.image_unfiltered
        self.pn = None
        self._update_image_info()
        self.slice_viewer.set_image(self.image)
        self._refresh_home()
        self.log("Working image reset to the unfiltered original.")

    # ---- Generate ------------------------------------------------------------
    def _build_generate_page(self):
        page = self._make_page(
            "generate", "Generators",
            "Tick exactly one generator, set its parameters, then click Generate. "
            "You are asked where to save the .tif first; the result becomes the "
            "working image.")
        sf = ScrollableFrame(page)
        sf.pack(fill="both", expand=True, padx=12, pady=4)
        single = SingleSelect()
        self._gen_secs = []
        for title, func, fields in SPEC["GEN"]:
            sec = build_card(sf.inner, title, fields, mode="single",
                             single=single, est=False)
            sec["func"] = func
            self._gen_secs.append(sec)
        grid_cards(sf.inner, self._gen_secs, 2)

        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=18, pady=(4, 12))
        b = make_button(bar, "Generate", self._do_generate, primary=True, width=16)
        b.pack(side="left")
        self.register_action(b)
        self._gen_time = tk.Label(bar, text="", bg=BG, fg=MUTED, font=FONT_IT)
        self._gen_time.pack(side="left", padx=14)

    def _do_generate(self):
        chosen = [s for s in self._gen_secs if card_selected(s)]
        if len(chosen) != 1:
            messagebox.showwarning("Selection", "Tick exactly one generator.",
                                   parent=self.root)
            return
        sec = chosen[0]
        try:
            kwargs = collect_kwargs(sec["ff"])
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.root)
            return
        save_file = get_save_file(sec)
        out_dir, chosen_name = (ask_save_file(self.root, save_file)
                                if save_file else (None, None))
        # cancelling the Save dialog still generates (file lands in default dir)
        if out_dir is None:
            out_dir = OUTPUT_DIR_DEFAULT

        def on_done(ok, result, out, err, elapsed, figs):
            self._gen_time.configure(text=f"run time {elapsed:.1f} s")
            if ok:
                rename_saved(out_dir, save_file, chosen_name)
                self.set_image(result, sec["title"].capitalize())
                for s in self._gen_secs:
                    s["var"].set(False)
                self.show_page("plots")
            else:
                messagebox.showerror("Generation failed", err, parent=self.root)

        self.submit(f"Generating {sec['title']}", sec["func"], kwargs,
                    out_dir=out_dir, on_done=on_done)

    # ---- Import ---------------------------------------------------------------
    def _build_import_page(self):
        page = self._make_page(
            "import", "Import an image",
            "Load a 3D volume from disk. Greyscale images are denoised "
            "(NL-means) and thresholded; binary images are used directly. "
            "PoreSpy treats True/bright voxels as pore — in most CT scans "
            "bright = solid, so inversion is usually needed.")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="x", padx=18, pady=14)

        card = tk.Frame(body, bg=CARD, padx=18, pady=16)
        card.pack(anchor="w")
        tk.Label(card, text="Source", bg=CARD, fg=TEXT, font=FONT_H2).pack(anchor="w")
        tk.Label(card, bg=CARD, fg=MUTED, font=FONT, justify="left",
                 text="A single 3D .tif / .raw / .dcm file, or a folder of 2D "
                      ".tif or .dcm slices.").pack(anchor="w", pady=(2, 10))
        b1 = make_button(card, "Select 3D image file  (.tif / .raw / .dcm)",
                         self._import_file, primary=True)
        b1.pack(anchor="w", pady=3, fill="x")
        b2 = make_button(card, "Select folder of 2D slices  (.tif / .dcm)",
                         self._import_folder)
        b2.pack(anchor="w", pady=3, fill="x")
        for b in (b1, b2):
            self.register_action(b)

    def _import_file(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Select a 3D image file",
            filetypes=[("3D image", "*.tif *.tiff *.raw *.dcm"),
                       ("TIFF image", "*.tif *.tiff"),
                       ("Raw binary", "*.raw"),
                       ("DICOM", "*.dcm"),
                       ("All files", "*.*")])
        if path:
            self._do_import(path, is_folder=False)

    def _import_folder(self):
        path = filedialog.askdirectory(
            parent=self.root, title="Select a folder of 2D slices (.tif or .dcm)")
        if path:
            self._do_import(path, is_folder=True)

    def _do_import(self, path, is_folder):
        raw_shape, raw_dtype = None, None
        if not is_folder and os.path.splitext(path)[1].lower() == ".raw":
            raw_shape, raw_dtype = self._ask_raw_params(os.path.getsize(path))
            if raw_shape is None:
                return
        is_dcm_file = not is_folder and os.path.splitext(path)[1].lower() == ".dcm"
        state = {"dcm_voxel_size": None}

        def _load():
            import imageio.v2 as iio
            from skimage.io import imread_collection
            if is_folder:
                dcm_files = sorted(
                    f for f in os.listdir(path) if f.lower().endswith(".dcm"))
                if dcm_files:
                    arr, state["dcm_voxel_size"] = _read_dicom_series(
                        [os.path.join(path, f) for f in dcm_files])
                    print(f"Loaded {len(dcm_files)} DICOM slices: shape={arr.shape}")
                else:
                    seq = imread_collection(os.path.join(path, "*.tif"))
                    if len(seq) == 0:
                        raise ValueError("No .tif or .dcm slices found in the folder.")
                    # slices are (rows, cols) = (y, x); stack to z then swap to (x, y, z)
                    arr = np.transpose(np.stack(seq, axis=-1), (1, 0, 2))
                    print(f"Loaded {len(seq)} slices: shape={arr.shape}")
            elif is_dcm_file:
                arr, state["dcm_voxel_size"] = _read_dicom_file(path)
                print(f"Loaded DICOM file: shape={arr.shape}")
            elif raw_shape is not None:
                arr = np.fromfile(path, dtype=np.dtype(raw_dtype)).reshape(raw_shape)
                arr = np.transpose(arr, (2, 1, 0))
                print(f"Loaded .raw image: shape={arr.shape}, dtype={raw_dtype}")
            else:
                arr = iio.volread(path)
                arr = np.transpose(arr, (2, 1, 0))
                print(f"Loaded 3D image: shape={arr.shape}")
            return arr

        def on_loaded(ok, im3d, out, err, elapsed, figs):
            if not ok:
                messagebox.showerror("Import failed", err, parent=self.root)
                return
            if state["dcm_voxel_size"] is not None:
                dx, dy, dz = state["dcm_voxel_size"]
                messagebox.showinfo(
                    "DICOM voxel spacing",
                    f"Voxel size from DICOM header: {dx:.5g} × {dy:.5g} × {dz:.5g} mm.\n"
                    "Use this value (in metres) for 'voxel_size' fields for "
                    "physically correct results.", parent=self.root)
            self._import_stage2(path, is_folder, im3d)

        self.submit("Loading image", _load, on_done=on_loaded, no_figs=True)

    def _import_stage2(self, path, is_folder, im3d):
        base_name = (os.path.basename(os.path.normpath(path)) if is_folder
                     else os.path.splitext(os.path.basename(path))[0])
        is_tif_source = (not is_folder and
                         os.path.splitext(path)[1].lower() in (".tif", ".tiff"))
        unique = np.unique(im3d)
        already_binary = len(unique) <= 2

        if already_binary:
            im_binary = im3d.astype(bool)
            poro_as_is = float(im_binary.sum()) / im_binary.size
            poro_inv = 1.0 - poro_as_is
            messagebox.showinfo(
                "Import",
                f"Image is already binary (values {unique}).\n\n"
                f"As-is    → porosity = {poro_as_is:.3f}  (True = pore)\n"
                f"Inverted → porosity = {poro_inv:.3f}\n\n"
                "PoreSpy treats True voxels as pore/void. In most CT scans "
                "bright = solid, dark = pore, so inversion is usually needed.",
                parent=self.root)
            inverted = messagebox.askyesno(
                "Invert image", "Invert the image (swap pore and solid)?\n"
                "Say Yes if pores are dark (low values) in your scan.",
                parent=self.root)
            if inverted:
                im_binary = ~im_binary
            self._import_finalize(base_name, im_binary, already_binary=True,
                                  inverted=inverted, is_tif_source=is_tif_source)
        else:
            im_float = im3d.astype(np.float64)
            im_float = (im_float - im_float.min()) / (im_float.max() - im_float.min())

            def _denoise():
                import porespy as ps
                # nl_means_layered only iterates correctly for cubic volumes;
                # move the slice axis to 0 and call with axis=0 to sidestep it.
                im_axis0 = np.moveaxis(im_float, 2, 0)
                d = ps.filters.nl_means_layered(im_axis0, axis=0)
                d = np.moveaxis(d, 0, 2)
                return (d - d.min()) / (d.max() - d.min())

            def on_denoised(ok, denoised, out, err, elapsed, figs):
                if not ok:
                    messagebox.showerror("Import failed", err, parent=self.root)
                    return
                self._import_threshold(base_name, im3d, denoised)

            self.submit("Denoising (NL-means)", _denoise, on_done=on_denoised,
                        no_figs=True)

    def _import_threshold(self, base_name, im3d, denoised):
        # Histogram figure (rendered inline in the dialog and kept in gallery)
        fig, ax = plt.subplots(figsize=[6, 3.4])
        ax.hist(denoised.flatten(), bins=50, edgecolor="k")
        ax.set_xlabel("Voxel intensity")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of denoised image")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        hist_png = buf.getvalue()
        self.add_plot("Import — intensity histogram", hist_png)

        dlg = tk.Toplevel(self.root)
        dlg.title("Choose segmentation threshold")
        dlg.configure(bg=BG)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("640x480")
        dlg.minsize(430, 360)
        img_lbl = tk.Label(dlg, bg=BG)
        img_lbl.pack(fill="both", expand=True, padx=14, pady=(14, 6))
        hstate = {"after": None, "size": None}

        def _fit_hist():
            hstate["after"] = None
            tw = max(1, img_lbl.winfo_width() - 4)
            th = max(1, img_lbl.winfo_height() - 4)
            if hstate["size"] == (tw, th):
                return
            hstate["size"] = (tw, th)
            try:
                photo = png_fit_photo(hist_png, tw, th)
            except tk.TclError:
                return
            img_lbl.configure(image=photo)
            img_lbl._photo = photo

        def _on_hist_resize(_e=None):
            if hstate["after"] is not None:
                dlg.after_cancel(hstate["after"])
            hstate["after"] = dlg.after(60, _fit_hist)

        img_lbl.bind("<Configure>", _on_hist_resize)
        row = tk.Frame(dlg, bg=BG)
        row.pack(pady=(0, 6))
        tk.Label(row, text="Threshold [0–1]  (voxels below → pore):",
                 bg=BG, fg=TEXT, font=FONT).pack(side="left")
        tvar = tk.StringVar(value="0.5")
        tk.Entry(row, textvariable=tvar, width=8, bg=FIELD, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=FONT).pack(side="left", padx=8)
        res = {"t": None}

        def _ok():
            try:
                t = float(tvar.get())
                if not (0.0 <= t <= 1.0):
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Invalid", "Enter a number between 0 and 1.",
                                       parent=dlg)
                return
            res["t"] = t
            dlg.destroy()
        make_button(dlg, "Segment", _ok, primary=True).pack(pady=(2, 14))
        dlg.wait_window()
        if res["t"] is None:
            self.log("Import cancelled at threshold step.")
            return
        threshold = res["t"]
        im_binary = denoised < threshold   # True = pore (void)

        inverted = messagebox.askyesno(
            "Invert image", "Invert the image (swap pore and solid phases)?",
            parent=self.root)
        if inverted:
            im_binary = ~im_binary

        # Before/after comparison figure into the gallery
        mid = im3d.shape[2] // 2
        fig, ax = plt.subplots(1, 3, figsize=[13, 4.5])
        ax[0].imshow(im3d[:, :, mid], cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(denoised[:, :, mid], cmap="gray"); ax[1].set_title("Denoised"); ax[1].axis("off")
        ax[2].imshow(~im_binary[:, :, mid], cmap="gray"); ax[2].set_title(f"Binary (t={threshold})"); ax[2].axis("off")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        self.add_plot("Import — segmentation preview", buf.getvalue())

        self._import_finalize(base_name, im_binary, already_binary=False,
                              inverted=inverted, is_tif_source=False)

    def _import_finalize(self, base_name, im_binary, already_binary,
                         inverted, is_tif_source):
        skip_save = is_tif_source and already_binary and not inverted
        if not skip_save:
            if already_binary and inverted:
                default_name = f"{base_name}_inverted.tif"
            elif inverted:
                default_name = f"{base_name}_binary_inverted.tif"
            else:
                default_name = f"{base_name}_binary.tif"
            out_dir, chosen = ask_save_file(self.root, default_name)
            if out_dir:
                save_name = chosen or default_name

                def _save():
                    import imageio.v2 as iio
                    arr = np.transpose(im_binary.astype("uint8") * 255, (2, 1, 0))
                    iio.volsave(os.path.join(out_dir, save_name), arr)
                    print(f"Binary image saved: {os.path.join(out_dir, save_name)}")
                self.submit("Saving binary image", _save, no_figs=True)
        self.set_image(im_binary, base_name)
        self.log(f"Import complete: {base_name}  shape={im_binary.shape}")
        self.show_page("home")

    def _ask_raw_params(self, file_size):
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
        dlg = tk.Toplevel(self.root)
        dlg.title("Raw image parameters")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text="Raw image import", bg=BG, fg=TEXT,
                 font=FONT_H2).pack(padx=18, pady=(16, 4))
        tk.Label(dlg, text=f"File size: {file_size:,} bytes\n"
                           "Enter the volume dimensions and data type.",
                 bg=BG, fg=MUTED, justify="center").pack(padx=18, pady=(0, 10))
        frm = tk.Frame(dlg, bg=BG)
        frm.pack(fill="x", padx=18, pady=4)
        tk.Label(frm, text="Shape (slices, H, W):", bg=BG, fg=TEXT,
                 anchor="w").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 8))
        shape_var = tk.StringVar(value=suggested_shape)
        tk.Entry(frm, textvariable=shape_var, width=22, bg=FIELD, fg=TEXT,
                 insertbackground=TEXT, relief="flat").grid(row=0, column=1, pady=3)
        tk.Label(frm, text="Data type:", bg=BG, fg=TEXT,
                 anchor="w").grid(row=1, column=0, sticky="w", pady=3, padx=(0, 8))
        dtype_var = tk.StringVar(value=suggested_dtype)
        ttk.Combobox(frm, textvariable=dtype_var, state="readonly", width=20,
                     values=["uint8", "uint16", "uint32", "int16", "float32", "float64"]
                     ).grid(row=1, column=1, sticky="w", pady=3)

        def _ok():
            try:
                parts = [int(x.strip()) for x in
                         shape_var.get().replace(";", ",").split(",") if x.strip()]
                if len(parts) != 3:
                    raise ValueError("Need exactly 3 values.")
                expected = (parts[0] * parts[1] * parts[2]
                            * np.dtype(dtype_var.get()).itemsize)
                if expected != file_size:
                    if not messagebox.askyesno(
                            "Size mismatch",
                            f"Shape {parts} × {dtype_var.get()} needs {expected:,} "
                            f"bytes but file has {file_size:,} bytes.\nContinue anyway?",
                            parent=dlg):
                        return
                result["shape"] = tuple(parts)
                result["dtype"] = dtype_var.get()
                dlg.destroy()
            except ValueError as e:
                messagebox.showwarning("Invalid input",
                                       f"Enter three comma-separated integers.\n{e}",
                                       parent=dlg)
        bf = tk.Frame(dlg, bg=BG)
        bf.pack(pady=(10, 16))
        make_button(bf, "OK", _ok, primary=True, width=10).pack(side="left", padx=6)
        make_button(bf, "Cancel", dlg.destroy, width=10).pack(side="left", padx=6)
        dlg.wait_window()
        return result["shape"], result["dtype"]

    # ---- Filters ----------------------------------------------------------
    def _build_filters_page(self):
        page = self._make_page(
            "filters", "Filters",
            "Solid-altering filters chain together in the order you set (1, 2, 3, …) "
            "and permanently replace the working image — they always run first. "
            "Non-solid-altering filters are simply ticked; each independently "
            "receives the solid-altering result. Tick 'save output' on a "
            "non-solid filter to write its result as a separate .tif.")
        sf = ScrollableFrame(page)
        sf.pack(fill="both", expand=True, padx=12, pady=4)

        n_solid = len(SPEC["FILTER_GROUPS"][0][1])
        order_values = [""] + [str(i) for i in range(1, n_solid + 1)]
        self._filter_secs = []
        self._solid_secs = []
        self._nonsolid_secs = []
        for gi, (group_label, specs) in enumerate(SPEC["FILTER_GROUPS"]):
            is_nonsolid = (gi == 1)
            tk.Label(sf.inner, text=group_label, bg=BG, fg=TEXT,
                     font=FONT_H2, anchor="w").pack(fill="x", padx=10, pady=(12, 2))
            ttk.Separator(sf.inner, orient="horizontal").pack(fill="x", padx=10, pady=(0, 4))
            holder = tk.Frame(sf.inner, bg=BG)
            holder.pack(fill="x")
            group_secs = []
            for title, func, fields in specs:
                # SA filters keep the run-order combobox; NSA filters are
                # plain checkboxes (they run after the SA chain, unordered).
                sec = build_card(holder, title, fields,
                                 mode="multi" if is_nonsolid else "order",
                                 order_values=order_values)
                sec["func"] = func
                if is_nonsolid:
                    sv = tk.BooleanVar(value=False)
                    row = tk.Frame(sec["frame"].winfo_children()[0], bg=CARD)
                    row.pack(fill="x", padx=8, pady=(0, 6))
                    tk.Checkbutton(row, text="save output (.tif)", variable=sv,
                                   bg=CARD, fg=MUTED, activebackground=CARD,
                                   activeforeground=TEXT, selectcolor=FIELD,
                                   font=FONT_SM, bd=0,
                                   highlightthickness=0).pack(side="left")
                    sec["save_var"] = sv
                    self._nonsolid_secs.append(sec)
                else:
                    sec["save_var"] = None
                    self._solid_secs.append(sec)
                group_secs.append(sec)
                self._filter_secs.append(sec)
            grid_cards(holder, group_secs, 2)
        self._register_cards(self._filter_secs)

        # Same order number on two SA filters: the newest wins, other clears.
        def _resolver(this_sec):
            def _resolve(*_):
                val = this_sec["var"].get()
                if not val:
                    return
                for other in self._solid_secs:
                    if other is not this_sec and other["var"].get() == val:
                        other["var"].set("")
            return _resolve
        for sec in self._solid_secs:
            sec["var"].trace_add("write", _resolver(sec))

        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=18, pady=(4, 12))
        b = make_button(bar, "Run filters", self._do_filters, primary=True, width=16)
        b.pack(side="left")
        self.register_action(b)
        self._filt_time = tk.Label(bar, text="", bg=BG, fg=MUTED, font=FONT_IT)
        self._filt_time.pack(side="left", padx=14)

    def _do_filters(self):
        if not self.require_image():
            return
        base_image = (self.image_unfiltered
                      if self.image_unfiltered is not None else self.image)
        solid = sorted([(card_order(s), s) for s in self._solid_secs
                        if card_order(s) is not None])
        nonsolid = [s for s in self._nonsolid_secs if card_selected(s)]
        if not solid and not nonsolid:
            messagebox.showinfo(
                "No filters", "Nothing selected — nothing to do.",
                parent=self.root)
            return
        # Validate all inputs up front
        jobs = []
        for sec in [s for _, s in solid] + nonsolid:
            try:
                kwargs = collect_kwargs(sec["ff"])
            except ValueError as e:
                messagebox.showerror("Invalid input", f"{sec['title']}: {e}",
                                     parent=self.root)
                return
            jobs.append((sec, kwargs))
        state = {"base": base_image, "elapsed": 0.0}

        # ---- phase 2: save SA result, then run NSA filters ----------------
        def after_solid():
            out_dir = None
            nsa_to_save = [s for s in nonsolid
                           if s["save_var"] and s["save_var"].get()]
            if solid:
                img_base = self.image_name or "image"
                suffix = "_".join(s["title"].lower() for _, s in solid)
                default_name = f"{img_base}_{suffix}.tif"
                out_dir, chosen = ask_save_file(self.root, default_name)
                if out_dir is None:
                    self.log("Filter run: save cancelled — result discarded.")
                    return
                save_name = chosen or default_name

                def _save():
                    import imageio.v2 as iio
                    b = state["base"]
                    arr = (b.astype(np.uint8) * 255 if b.dtype == bool
                           else b.astype(np.float32))
                    iio.volsave(os.path.join(out_dir, save_name),
                                np.transpose(arr, (2, 1, 0)))
                    print(f"Saved {save_name}")
                self.submit("Saving filtered volume", _save, no_figs=True)
            elif nsa_to_save:
                out_dir = filedialog.askdirectory(
                    parent=self.root, initialdir=OUTPUT_DIR_DEFAULT,
                    title="Choose a folder for the filter output files")
                if not out_dir:
                    return
            run_nonsolid(0, out_dir)

        def run_nonsolid(idx, out_dir):
            if idx >= len(nonsolid):
                finish()
                return
            sec = nonsolid[idx]
            kwargs = next(k for s, k in jobs if s is sec)
            kwargs = dict(kwargs)
            kwargs["im"] = state["base"]

            def on_done(ok, result, out, err, elapsed, figs, sec=sec,
                        idx=idx, out_dir=out_dir):
                state["elapsed"] += elapsed
                if ok and sec["save_var"] and sec["save_var"].get() and out_dir:
                    nsa_name = f"{self.image_name or 'image'}_{sec['title'].lower()}.tif"

                    def _save(result=result, nsa_name=nsa_name):
                        import imageio.v2 as iio
                        arr = (result.astype(np.uint8) * 255
                               if result.dtype == bool
                               else result.astype(np.float32))
                        iio.volsave(os.path.join(out_dir, nsa_name),
                                    np.transpose(arr, (2, 1, 0)))
                        print(f"Saved {nsa_name}")
                    self.submit(f"Saving {nsa_name}", _save, no_figs=True)
                if not ok:
                    messagebox.showerror("Filter failed",
                                         f"{sec['title']}:\n{err}", parent=self.root)
                run_nonsolid(idx + 1, out_dir)
            self.submit(f"Filter: {sec['title']}", sec["func"], kwargs,
                        on_done=on_done)

        def finish():
            self.image = state["base"]
            self.pn = None
            self._update_image_info()
            if solid:      # keep the filtered result selectable in the viewer
                self.slice_viewer.register_volume(
                    f"{self.image_name or 'image'} (filtered)", self.image)
            self.slice_viewer.set_image(self.image)
            self._filt_time.configure(text=f"run time {state['elapsed']:.1f} s")
            for s in self._solid_secs:
                s["var"].set("")
            for s in self._nonsolid_secs:
                s["var"].set(False)
            self.show_page("plots")

        # ---- phase 1: solid-altering chain ---------------------------------
        def run_solid(idx):
            if idx >= len(solid):
                after_solid()
                return
            order, sec = solid[idx]
            kwargs = next(k for s, k in jobs if s is sec)
            kwargs = dict(kwargs)
            kwargs["im"] = state["base"]

            def on_done(ok, result, out, err, elapsed, figs, sec=sec, idx=idx):
                state["elapsed"] += elapsed
                if not ok:
                    messagebox.showerror("Filter failed",
                                         f"{sec['title']}:\n{err}", parent=self.root)
                    return
                state["base"] = result
                run_solid(idx + 1)
            self.submit(f"Filter {order}: {sec['title']}", sec["func"], kwargs,
                        on_done=on_done)

        run_solid(0)

    # ---- Metrics / PoreSpy sims (shared machinery) --------------------------
    def _build_ops_page(self, key, title, subtitle, specs, run_label):
        page = self._make_page(key, title, subtitle)
        sf = ScrollableFrame(page)
        sf.pack(fill="both", expand=True, padx=12, pady=4)
        secs = []
        for t, func, fields in specs:
            sec = build_card(sf.inner, t, fields, mode="multi", with_result=True)
            sec["func"] = func
            secs.append(sec)
        grid_cards(sf.inner, secs, 2)
        self._register_cards(secs)

        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=18, pady=(4, 12))
        time_lbl = tk.Label(bar, text="", bg=BG, fg=MUTED, font=FONT_IT)
        b = make_button(bar, run_label,
                        lambda: self._run_ops(secs, time_lbl),
                        primary=True, width=16)
        b.pack(side="left")
        self.register_action(b)
        make_button(bar, "Clear selection",
                    lambda: [s["var"].set(False) or (s["result"].reset()
                             if s["result"] else None) for s in secs]
                    ).pack(side="left", padx=8)
        time_lbl.pack(side="left", padx=8)
        return secs

    def _build_metrics_page(self):
        self._metric_secs = self._build_ops_page(
            "metrics", "Metrics",
            "Tick any number of metrics, set their parameters, then click "
            "Calculate. Each card shows the captured output (green) or the "
            "error (red); plots land in the Plots page.",
            SPEC["METRICS"], "Calculate")

    def _build_psim_page(self):
        self._psim_secs = self._build_ops_page(
            "psim", "PoreSpy simulations",
            "Voxel-based two-phase simulations (drainage / imbibition / injection "
            "save .vti files for ParaView) plus tortuosity. Tick any number, then "
            "click Simulate.",
            SPEC["PSIM"], "Simulate")

    def _run_ops(self, secs, time_lbl):
        if not self.require_image():
            return
        chosen = [s for s in secs if card_selected(s)]
        if not chosen:
            messagebox.showwarning("Selection", "Tick at least one operation.",
                                   parent=self.root)
            return
        saving = [(s, get_save_file(s)) for s in chosen
                  if get_save_file(s) is not None]
        if not saving:
            out_dir, name_map = None, {}
        elif len(saving) == 1:
            out_dir, chosen_name = ask_save_file(self.root, saving[0][1])
            if out_dir is None:
                return
            name_map = {saving[0][1]: chosen_name}
        else:
            out_dir = filedialog.askdirectory(
                parent=self.root, initialdir=OUTPUT_DIR_DEFAULT,
                title="Choose a folder for the output files")
            if not out_dir:
                return
            name_map = {f: f for _, f in saving}

        state = {"elapsed": 0.0}

        def run_one(idx):
            if idx >= len(chosen):
                time_lbl.configure(text=f"run time {state['elapsed']:.1f} s")
                for s in secs:
                    s["var"].set(False)
                return
            sec = chosen[idx]
            try:
                kwargs = collect_kwargs(sec["ff"])
            except ValueError as e:
                sec["result"].show(str(e), ok=False)
                run_one(idx + 1)
                return
            kwargs["im"] = self.image

            def on_done(ok, result, out, err, elapsed, figs, sec=sec, idx=idx):
                state["elapsed"] += elapsed
                if ok:
                    sec["result"].show(out or "Done (see Plots / saved files).", ok=True)
                    sf_name = get_save_file(sec)
                    if sf_name and out_dir and sf_name in name_map:
                        rename_saved(out_dir, sf_name, name_map[sf_name])
                        final = ((name_map[sf_name] or "").strip() or sf_name)
                        if final.lower().endswith(".vti"):
                            fp = os.path.join(out_dir, final)
                            if os.path.isfile(fp):
                                self.slice_viewer.register_vti_file(fp)
                                self.log(f"{final} can be viewed slice-by-slice "
                                         "in the Home slice viewer.")
                else:
                    sec["result"].show(err, ok=False)
                run_one(idx + 1)
            self.submit(sec["title"], sec["func"], kwargs, out_dir=out_dir,
                        on_done=on_done)
        run_one(0)

    # ---- OpenPNM ------------------------------------------------------------
    def _build_opnm_page(self):
        page = self._make_page(
            "opnm", "OpenPNM simulations",
            "A pore network is extracted from the working image (snow2) and "
            "cached. Choose exactly one phase, tick any number of simulations, "
            "then Simulate. All results are exported together into a single "
            ".vtp file for ParaView.")
        sf = ScrollableFrame(page)
        sf.pack(fill="both", expand=True, padx=12, pady=4)

        vs_row = tk.Frame(sf.inner, bg=BG)
        vs_row.pack(fill="x", padx=10, pady=(6, 0))
        vs_lbl = tk.Label(vs_row, text="Voxel size (m):", bg=BG, fg=TEXT, font=FONT)
        vs_lbl.pack(side="left")
        Tooltip(vs_lbl, "Physical length of each voxel (m) — passed to snow2 "
                        "for network geometry")
        self._opnm_vs = tk.StringVar(value="1")
        tk.Entry(vs_row, textvariable=self._opnm_vs, width=12, bg=FIELD, fg=TEXT,
                 insertbackground=TEXT, relief="flat", font=FONT,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", padx=8)

        tk.Label(sf.inner, text="Phase (pick one)", bg=BG, fg=TEXT,
                 font=FONT_H2, anchor="w").pack(fill="x", padx=10, pady=(12, 2))
        phase_holder = tk.Frame(sf.inner, bg=BG)
        phase_holder.pack(fill="x")
        single = SingleSelect()
        self._phase_secs = {}
        for name, fields in SPEC["PNM_PHASES"]:
            sec = build_card(phase_holder, name.capitalize(), fields,
                             mode="single", single=single, est=False)
            self._phase_secs[name] = sec
        grid_cards(phase_holder, list(self._phase_secs.values()), 3)

        tk.Label(sf.inner, text="Simulations", bg=BG, fg=TEXT,
                 font=FONT_H2, anchor="w").pack(fill="x", padx=10, pady=(14, 2))
        sim_holder = tk.Frame(sf.inner, bg=BG)
        sim_holder.pack(fill="x")
        self._pnm_secs = []
        for title, func, fields in SPEC["PNM_SIMS"]:
            sec = build_card(sim_holder, title, fields, mode="multi",
                             with_result=True)
            sec["func"] = func
            self._pnm_secs.append(sec)
        grid_cards(sim_holder, self._pnm_secs, 2)
        self._register_cards(self._pnm_secs)

        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=18, pady=(4, 12))
        b = make_button(bar, "Simulate", self._do_opnm, primary=True, width=16)
        b.pack(side="left")
        self.register_action(b)
        self._opnm_time = tk.Label(bar, text="", bg=BG, fg=MUTED, font=FONT_IT)
        self._opnm_time.pack(side="left", padx=14)

    def _do_opnm(self):
        if not self.require_image():
            return
        chosen_phase = [n for n, s in self._phase_secs.items() if card_selected(s)]
        if len(chosen_phase) != 1:
            messagebox.showwarning("Phase", "Tick exactly one phase.",
                                   parent=self.root)
            return
        phase_name = chosen_phase[0]
        try:
            overrides = collect_kwargs(self._phase_secs[phase_name]["ff"])
        except ValueError as e:
            messagebox.showerror("Invalid phase input", str(e), parent=self.root)
            return
        chosen = [s for s in self._pnm_secs if card_selected(s)]
        if not chosen:
            messagebox.showwarning("Selection", "Tick at least one simulation.",
                                   parent=self.root)
            return
        combined_default = ("openpnm_"
                            + "_".join(SPEC["PNM_ABBR"][s["title"]] for s in chosen)
                            + ".vtp")
        out_dir, vtk_name = ask_save_file(self.root, combined_default)
        if out_dir is None:
            return
        try:
            voxel_size = float(self._opnm_vs.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", "Voxel size must be a number.",
                                 parent=self.root)
            return

        state = {"elapsed": 0.0, "n_ok": 0}

        def run_sims(idx):
            if idx >= len(chosen):
                export()
                return
            sec = chosen[idx]
            try:
                kwargs = collect_kwargs(sec["ff"])
            except ValueError as e:
                sec["result"].show(str(e), ok=False)
                run_sims(idx + 1)
                return
            kwargs["network"] = self.pn
            kwargs["phase_overrides"] = overrides
            kwargs["save_vtk"] = False   # one combined export at the end
            func = sec["func"]
            if func in (opnm.simulate_drainage, opnm.simulate_invasion_percolation):
                kwargs["inv_phase"] = phase_name
                kwargs["def_phase"] = "air" if phase_name == "water" else "water"
            else:
                kwargs["chosen_phase"] = phase_name

            def on_done(ok, result, out, err, elapsed, figs, sec=sec, idx=idx):
                state["elapsed"] += elapsed
                if ok:
                    sec["result"].show(out or "Done.", ok=True)
                    state["n_ok"] += 1
                else:
                    sec["result"].show(err, ok=False)
                run_sims(idx + 1)
            self.submit(f"OpenPNM: {sec['title']}", func, kwargs,
                        out_dir=out_dir, on_done=on_done)

        def export():
            if not state["n_ok"]:
                self._opnm_time.configure(text=f"run time {state['elapsed']:.1f} s")
                return
            stem = os.path.splitext(vtk_name or combined_default)[0]

            def on_done(ok, result, out, err, elapsed, figs):
                state["elapsed"] += elapsed
                if ok:
                    self._opnm_time.configure(
                        text=f"run time {state['elapsed']:.1f} s   |   saved {stem}.vtp")
                else:
                    messagebox.showerror("VTK export failed", err, parent=self.root)
                for s in self._pnm_secs:
                    s["var"].set(False)
                for s in self._phase_secs.values():
                    s["var"].set(False)
            self.submit("Exporting combined VTK", opnm.export_vtk,
                        {"network": self.pn, "filename": stem},
                        out_dir=out_dir, on_done=on_done, no_figs=True)

        # Build (or reuse) the network, then run the simulations.
        if (self.pn is None or self.pn_image_id != id(self.image)
                or self.pn_voxel_size != voxel_size):
            def on_built(ok, pn, out, err, elapsed, figs):
                state["elapsed"] += elapsed
                if not ok:
                    messagebox.showerror("Network build failed", err,
                                         parent=self.root)
                    return
                self.pn = pn
                self.pn_image_id = id(self.image)
                self.pn_voxel_size = voxel_size
                run_sims(0)
            self.submit("Building pore network (snow2)", opnm.build_network,
                        {"image": self.image, "voxel_size": voxel_size},
                        out_dir=out_dir, on_done=on_built)
        else:
            run_sims(0)

    # ---- Plots page ----------------------------------------------------------
    def _build_plots_page(self):
        page = self._make_page(
            "plots", "Plot Gallery",
            "Every figure produced by a run is captured here. Click a plot to "
            "zoom, 💾 to save it as PNG, ✕ to remove it.")
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=18, pady=(0, 4))
        make_button(bar, "Clear all", self._clear_plots).pack(side="left")
        sf = ScrollableFrame(page)
        sf.pack(fill="both", expand=True, padx=12, pady=4)
        self._gallery_grid = sf.inner

    # ---- Log page --------------------------------------------------------------
    def _build_log_page(self):
        page = self._make_page(
            "log", "Session log",
            "Timestamped record of every run: captured output, errors and durations.")
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=18, pady=(0, 4))
        make_button(bar, "Clear log", self._clear_log).pack(side="left")
        holder = tk.Frame(page, bg=BORDER, padx=1, pady=1)
        holder.pack(fill="both", expand=True, padx=18, pady=(4, 14))
        self._log_text = tk.Text(holder, bg="#10141c", fg=TEXT, relief="flat",
                                 wrap="word", state="disabled", font=("Consolas", 9),
                                 insertbackground=TEXT)
        vsb = ttk.Scrollbar(holder, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._log_text.pack(side="left", fill="both", expand=True)
        self._log_text.tag_configure("info", foreground=TEXT)
        self._log_text.tag_configure("ok", foreground=GOOD)
        self._log_text.tag_configure("err", foreground=BAD)
        self._log_text.tag_configure("mut", foreground=MUTED)

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")


# ===========================================================================
#  Slice viewer
# ===========================================================================
WORKING_VIEW = "● Working image"


class SliceViewer(tk.Frame):
    """Interactive orthogonal slice viewer.

    Besides the working image it can preview any volume produced during the
    session (generator / filter snapshots) and .tif / .vti files opened from
    disk. The view selection never changes the working image used for
    processing — a notice appears whenever something else is being viewed."""

    MAX_EXTRA = 6    # in-memory volumes kept besides the working image

    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app = app
        self._photo = None
        self._volumes = {}     # name -> {kind: "array"|"vti", arr, path}
        self._order = []       # insertion order of extra volumes (for eviction)
        self._render_after = None

        top = tk.Frame(self, bg=CARD)
        top.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(top, text="Slice viewer", bg=CARD, fg=TEXT,
                 font=FONT_H2).pack(side="left")
        self.pos_lbl = tk.Label(top, text="", bg=CARD, fg=MUTED, font=FONT_SM)
        self.pos_lbl.pack(side="right")

        vrow = tk.Frame(self, bg=CARD)
        vrow.pack(fill="x", padx=12, pady=(0, 2))
        tk.Label(vrow, text="view", bg=CARD, fg=MUTED,
                 font=FONT_SM).pack(side="left")
        self.view_var = tk.StringVar(value=WORKING_VIEW)
        self.view_cb = ttk.Combobox(vrow, textvariable=self.view_var,
                                    values=[WORKING_VIEW], state="readonly",
                                    width=32, font=FONT_SM)
        self.view_cb.pack(side="left", padx=(4, 8))
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self._view_changed())
        make_button(vrow, "Open .tif / .vti…",
                    self._open_file).pack(side="left")
        tk.Label(vrow, text="axis", bg=CARD, fg=MUTED,
                 font=FONT_SM).pack(side="left", padx=(14, 4))
        self.axis_var = tk.StringVar(value="2")
        cb = ttk.Combobox(vrow, textvariable=self.axis_var,
                          values=["0", "1", "2"], state="readonly", width=3,
                          font=FONT_SM)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._axis_changed())

        self.note_lbl = tk.Label(self, text="", bg=CARD, fg=WARN,
                                 font=FONT_IT, anchor="w", padx=2)
        self.note_lbl.pack(fill="x", padx=12)

        disp = tk.Frame(self, bg="#0b0e14")
        disp.pack(fill="both", expand=True, padx=12, pady=4)
        self.canvas_lbl = tk.Label(disp, bg="#0b0e14",
                                   text="no image", fg=MUTED, font=FONT)
        self.canvas_lbl.pack(side="left", fill="both", expand=True)
        self.canvas_lbl.bind("<Configure>", self._on_canvas_resize)

        # Colorbar panel (shown only for numeric fields, hidden for chords)
        self._cbar_frame = tk.Frame(disp, bg="#0b0e14")
        self._cbar_title = tk.Label(self._cbar_frame, text="", bg="#0b0e14",
                                    fg=TEXT, font=FONT_SM, wraplength=86,
                                    justify="center")
        self._cbar_title.pack(side="top", pady=(6, 2))
        self._cbar_canvas = tk.Canvas(self._cbar_frame, width=88, bg="#0b0e14",
                                      highlightthickness=0)
        self._cbar_canvas.pack(side="top", fill="y", expand=True, padx=(6, 4))
        self._cbar_canvas.bind("<Configure>", lambda e: self._draw_cbar())
        self._cbar_visible = False
        self._view_range = None          # (vmin, vmax) of the viewed volume

        self.slider = tk.Scale(self, from_=0, to=0, orient="horizontal",
                               bg=CARD, fg=TEXT, troughcolor=FIELD,
                               highlightthickness=0, bd=0,
                               activebackground=ACCENT, showvalue=False,
                               command=lambda v: self.render())
        self.slider.pack(fill="x", padx=12, pady=(0, 10))

        legend = tk.Frame(self, bg=CARD)
        legend.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(legend, text="■", fg="#fde725", bg=CARD).pack(side="left")
        tk.Label(legend, text="pore / void", fg=MUTED, bg=CARD,
                 font=FONT_SM).pack(side="left", padx=(2, 12))
        tk.Label(legend, text="■", fg="#472d7b", bg=CARD).pack(side="left")
        tk.Label(legend, text="solid", fg=MUTED, bg=CARD,
                 font=FONT_SM).pack(side="left", padx=(2, 12))
        tk.Label(legend, text="(numeric fields use the viridis colormap)",
                 fg=MUTED, bg=CARD, font=FONT_SM).pack(side="left")

    # ---------------- volume registry --------------------------------------
    def current_array(self):
        name = self.view_var.get()
        if name == WORKING_VIEW:
            return self.app.image
        v = self._volumes.get(name)
        return None if v is None else v.get("arr")

    def _refresh_values(self):
        self.view_cb.configure(values=[WORKING_VIEW] + self._order)

    def register_volume(self, name, arr):
        """Add (or update) an in-memory volume for viewing; returns its name."""
        name = (name or "volume").strip() or "volume"
        if name == WORKING_VIEW:
            name = "volume"
        if name in self._volumes:                    # update in place
            self._volumes[name] = {"kind": "array", "arr": arr}
            self._order.remove(name)
            self._order.append(name)
        else:
            self._volumes[name] = {"kind": "array", "arr": arr}
            self._order.append(name)
            self._evict()
        self._refresh_values()
        return name

    def register_vti_file(self, path):
        """Add a .vti file entry (loaded lazily on first selection)."""
        name = os.path.basename(path)
        if name in self._volumes and self._volumes[name].get("path") == path:
            return name
        self._volumes[name] = {"kind": "vti", "arr": None, "path": path}
        if name in self._order:
            self._order.remove(name)
        self._order.append(name)
        self._evict()
        self._refresh_values()
        return name

    def _evict(self):
        while len(self._order) > self.MAX_EXTRA:
            old = self._order.pop(0)
            del self._volumes[old]
            if self.view_var.get() == old:
                self.view_var.set(WORKING_VIEW)

    def select_view(self, name):
        if name == WORKING_VIEW or name in self._volumes:
            self.view_var.set(name)
            self._view_changed()

    # ---------------- events -------------------------------------------------
    def _view_changed(self):
        name = self.view_var.get()
        if name != WORKING_VIEW:
            v = self._volumes.get(name)
            if v and v["kind"] == "vti" and v.get("arr") is None:
                self._load_vti_entry(name, v)
                return                      # re-selected when parsing is done
        self._reset_slider()
        self._update_note()
        self._update_range()
        self.render()

    # ---------------- colorbar ---------------------------------------------
    @staticmethod
    def _fmt_val(v):
        return f"{v:.4g}"

    def _update_range(self):
        """Volume-wide vmin/vmax so colours stay consistent across slices and
        match the colorbar. Chord/label fields get colours but no bar."""
        arr = self.current_array()
        name = self.view_var.get()
        if arr is None or arr.dtype == bool:
            self._view_range = None
            self._set_cbar(False)
            return
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self._view_range = None
            self._set_cbar(False)
            return
        lo, hi = float(finite.min()), float(finite.max())
        self._view_range = (lo, hi)
        if "chord" in name.lower():          # labelled chords: bar meaningless
            self._set_cbar(False)
        else:
            # Title: for .vti entries ("file.vti · field") show the field name
            title = name.split("·")[-1].strip() if "·" in name else name
            self._cbar_title.configure(text=title.replace("_", " "))
            self._set_cbar(True)
            self._draw_cbar()

    def _set_cbar(self, visible):
        if visible and not self._cbar_visible:
            self._cbar_frame.pack(side="right", fill="y")
            self._cbar_visible = True
            self._draw_cbar()
        elif not visible and self._cbar_visible:
            self._cbar_frame.pack_forget()
            self._cbar_visible = False

    def _draw_cbar(self):
        """Gradient bar with numbered major tick marks (nice round values)."""
        if not self._cbar_visible or self._view_range is None:
            return
        c = self._cbar_canvas
        h = c.winfo_height()
        if h < 40:
            return
        pad = 10                              # room for end tick labels
        bar_w, bar_h = 16, h - 2 * pad
        idx = (np.linspace(1, 0, bar_h) * 255).astype(np.uint8)   # max on top
        rgb = np.repeat(_viridis_lut()[idx][:, None, :], bar_w, axis=1)
        try:
            self._cbar_photo = _photo_from_rgb(rgb)
        except tk.TclError:
            return
        c.delete("all")
        c.create_image(0, pad, image=self._cbar_photo, anchor="nw")
        lo, hi = self._view_range
        if hi <= lo:
            return
        from matplotlib.ticker import MaxNLocator
        ticks = [t for t in MaxNLocator(nbins=6).tick_values(lo, hi)
                 if lo <= t <= hi]
        for t in ticks:
            y = pad + bar_h - (t - lo) / (hi - lo) * bar_h
            c.create_line(bar_w, y, bar_w + 5, y, fill=TEXT)
            c.create_text(bar_w + 8, y, text=self._fmt_val(t), anchor="w",
                          fill=TEXT, font=FONT_SM)

    def _load_vti_entry(self, name, v):
        path = v["path"]

        def on_done(ok, fields, out, err, *_a):
            if not ok:
                messagebox.showerror("Could not read .vti", err,
                                     parent=self.app.root)
                self.select_view(WORKING_VIEW)
                return
            base = os.path.basename(path)
            if name in self._order:          # replace file entry with fields
                self._order.remove(name)
            self._volumes.pop(name, None)
            first = None
            for fname, arr in fields.items():
                n = self.register_volume(f"{base} · {fname}", arr)
                if first is None:
                    first = n
            if first:
                self.select_view(first)
        self.app.submit(f"Reading {os.path.basename(path)}", read_vti,
                        {"path": path}, on_done=on_done, no_figs=True)

    def _open_file(self):
        path = filedialog.askopenfilename(
            parent=self.app.root, title="Open a volume to view",
            initialdir=OUTPUT_DIR_DEFAULT,
            filetypes=[("3D volume", "*.tif *.tiff *.vti"),
                       ("TIFF image", "*.tif *.tiff"),
                       ("VTK ImageData", "*.vti"),
                       ("All files", "*.*")])
        if not path:
            return
        if path.lower().endswith(".vti"):
            self.select_view(self.register_vti_file(path))
            return

        def _load():
            import imageio.v2 as iio
            arr = iio.volread(path)
            if arr.ndim != 3:
                raise ValueError(f"Not a 3D volume: shape {arr.shape}")
            arr = np.transpose(arr, (2, 1, 0))
            vals = np.unique(arr)
            if vals.size <= 2:
                arr = arr != vals.min()      # binary: bright = pore
            print(f"Loaded for viewing: shape={arr.shape}")
            return arr

        def on_done(ok, arr, out, err, *_a):
            if not ok:
                messagebox.showerror("Could not open volume", err,
                                     parent=self.app.root)
                return
            self.select_view(self.register_volume(os.path.basename(path), arr))
        self.app.submit(f"Loading {os.path.basename(path)}", _load,
                        on_done=on_done, no_figs=True)

    def _axis_changed(self):
        self._reset_slider()
        self.render()

    def _on_canvas_resize(self, _e=None):
        # Debounced re-render so the slice always fills the display area.
        if self._render_after is not None:
            self.after_cancel(self._render_after)
        self._render_after = self.after(80, self._render_now)

    def _render_now(self):
        self._render_after = None
        self.render()

    def _update_note(self):
        if self.view_var.get() == WORKING_VIEW:
            self.note_lbl.configure(text="")
        else:
            self.note_lbl.configure(
                text="⚠ preview only — processing still uses the working image "
                     "(see bottom-left)")

    def _reset_slider(self):
        arr = self.current_array()
        if arr is None:
            return
        axis = int(self.axis_var.get())
        n = arr.shape[axis]
        self.slider.configure(to=max(0, n - 1))
        self.slider.set(n // 2)

    # ---------------- API used by the app ----------------------------------
    def set_image(self, im):
        """Working image changed: switch the view back to it."""
        self.view_var.set(WORKING_VIEW)
        self._update_note()
        self._update_range()
        if im is None:
            self.canvas_lbl.configure(image="", text="no image")
            self._photo = None
            self.pos_lbl.configure(text="")
            return
        self._reset_slider()
        self.render()

    def render(self):
        im = self.current_array()
        if im is None:
            self.canvas_lbl.configure(image="", text="no image")
            self._photo = None
            self.pos_lbl.configure(text="")
            return
        axis = int(self.axis_var.get())
        idx = min(int(self.slider.get()), im.shape[axis] - 1)
        if axis == 0:
            plane = im[idx, :, :]
        elif axis == 1:
            plane = im[:, idx, :]
        else:
            plane = im[:, :, idx]
        # Fill the black display area (fall back before first layout pass).
        w = self.canvas_lbl.winfo_width() - 8
        h = self.canvas_lbl.winfo_height() - 8
        if w < 60 or h < 60:
            w = h = 430
        vmin, vmax = self._view_range if self._view_range else (None, None)
        try:
            self._photo = slice_to_photo(plane, w, h, vmin=vmin, vmax=vmax)
            self.canvas_lbl.configure(image=self._photo, text="")
            self.pos_lbl.configure(
                text=f"slice {idx + 1} / {im.shape[axis]}   "
                     f"axis {'xyz'[axis]}")
        except tk.TclError:
            pass


# ===========================================================================
#  Entry point
# ===========================================================================
def main():
    # Crisp fonts on high-DPI Windows displays
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    os.makedirs(OUTPUT_DIR_DEFAULT, exist_ok=True)

    root = tk.Tk()
    style_app(root)
    root.title(APP_TITLE)
    root.geometry("560x340")

    # Window / taskbar icon
    ico = asset_path("icon.ico")
    if ico:
        try:
            root.iconbitmap(default=ico)
        except tk.TclError:
            pass
    icon_full = load_app_icon()
    if icon_full is not None:
        try:
            root.iconphoto(True, icon_full)
        except tk.TclError:
            pass

    splash = tk.Frame(root, bg=BG)
    splash.pack(fill="both", expand=True)
    splash_icon = load_app_icon(96)
    if splash_icon is not None:
        tk.Label(splash, image=splash_icon, bg=BG).pack(pady=(42, 8))
        pady_title = (0, 2)
    else:
        pady_title = (58, 2)
    tk.Label(splash, text=APP_TITLE, bg=BG, fg=TEXT,
             font=("Segoe UI", 20, "bold")).pack(pady=pady_title)
    status = tk.Label(splash, text="Loading scientific libraries "
                                   "(porespy, openpnm)…", bg=BG, fg=MUTED,
                      font=FONT)
    status.pack(pady=(14, 0))
    root.update()

    try:
        load_project_modules()
        build_specs()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        messagebox.showerror(
            "Start-up failed",
            "Could not import the project modules.\n\n"
            f"{type(e).__name__}: {e}\n\n"
            "Make sure this file can see the folder containing "
            "Generators_3D.py etc., and that the Python environment has "
            "porespy / openpnm installed.", parent=root)
        root.destroy()
        return

    os.chdir(OUTPUT_DIR_DEFAULT)
    splash.destroy()
    FableApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
