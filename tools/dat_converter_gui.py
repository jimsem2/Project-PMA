# -*- coding: utf-8 -*-
"""
Drag-and-drop converter: ASCII .dat voxel volumes  ->  3-D .tif for the
Porous Media Analysis app.

Drop .dat files (or whole folders) onto the window and they are converted to
3-D TIFF volumes that the app imports directly — already binary, correct
porosity, no inversion needed.

This is a STANDALONE tool. It does not import or modify the application
source, so using it never requires rebuilding or updating the .exe.

Run:
    python tools\\dat_converter_gui.py
    python tools\\dat_converter_gui.py file1.dat file2.dat   (pre-loaded)

Drag-and-drop needs the tkinterdnd2 package (pip install tkinterdnd2). If it
is missing the window still works — use the 'Add files' / 'Add folder'
buttons instead.
"""
import os
import queue
import sys
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dat_to_tif import collect_dat_files, convert   # noqa: E402

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except ImportError:                                  # graceful fallback
    HAVE_DND = False

# --- theme (matches the main application) ----------------------------------
BG, CARD, FIELD = "#12151c", "#1b2029", "#242b38"
BORDER, TEXT, MUTED = "#2b3342", "#e6e9f0", "#8b93a7"
ACCENT, ACCENT_D = "#4f9cf9", "#3b82e0"
GOOD, BAD, WARN = "#34d399", "#f87171", "#fbbf24"
FONT, FONT_SM = ("Segoe UI", 10), ("Segoe UI", 9)
FONT_B, FONT_H1 = ("Segoe UI", 10, "bold"), ("Segoe UI", 16, "bold")


def asset(name):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "assets", name)
    return p if os.path.isfile(p) else None


class ConverterApp:
    def __init__(self, root, preload=()):
        self.root = root
        self.files = []                 # queued .dat paths
        self.msgs = queue.Queue()
        self.busy = False

        root.title("DAT → TIF converter")
        root.configure(bg=BG)
        root.geometry("760x600")
        root.minsize(640, 520)
        ico = asset("icon.ico")
        if ico:
            try:
                root.iconbitmap(default=ico)
            except tk.TclError:
                pass

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground=FIELD, background=CARD,
                        foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER,
                        lightcolor=CARD, darkcolor=CARD,
                        selectbackground=FIELD, selectforeground=TEXT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", FIELD)],
                  foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", FIELD)],
                  selectforeground=[("readonly", TEXT)])
        root.option_add("*TCombobox*Listbox.background", FIELD)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_D)
        root.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        style.configure("TProgressbar", background=ACCENT, troughcolor=FIELD,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)

        self._build()
        if preload:
            self._add(preload)
        self.root.after(100, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self):
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=18, pady=(14, 6))
        tk.Label(head, text="DAT → TIF converter", bg=BG, fg=TEXT,
                 font=FONT_H1).pack(anchor="w")
        tk.Label(head, bg=BG, fg=MUTED, font=FONT_SM, justify="left",
                 text=("Converts ASCII 0/1 voxel volumes into 3-D TIFF files "
                       "that Porous Media Analysis imports directly.")
                 ).pack(anchor="w", pady=(2, 0))

        # drop zone
        self.drop = tk.Frame(self.root, bg=CARD, highlightthickness=2,
                             highlightbackground=BORDER,
                             highlightcolor=BORDER)
        self.drop.pack(fill="x", padx=18, pady=(10, 8))
        msg = ("Drop .dat files or folders here"
               if HAVE_DND else "Use the buttons below to add .dat files")
        self.drop_lbl = tk.Label(self.drop, text=msg, bg=CARD, fg=TEXT,
                                 font=("Segoe UI", 13, "bold"), pady=26)
        self.drop_lbl.pack()
        sub = ("…or use the buttons below" if HAVE_DND
               else "(install tkinterdnd2 to enable drag-and-drop)")
        self.drop_sub = tk.Label(self.drop, text=sub, bg=CARD, fg=MUTED,
                                 font=FONT_SM, pady=(0))
        self.drop_sub.pack(pady=(0, 18))

        if HAVE_DND:
            for w in (self.drop, self.drop_lbl, self.drop_sub):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_drop)
                w.dnd_bind("<<DropEnter>>", self._drop_enter)
                w.dnd_bind("<<DropLeave>>", self._drop_leave)

        # buttons
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=18)
        self._btn(bar, "Add files…", self._browse_files).pack(side="left")
        self._btn(bar, "Add folder…", self._browse_folder).pack(side="left",
                                                                padx=6)
        self._btn(bar, "Clear list", self._clear).pack(side="left")
        self.convert_btn = self._btn(bar, "Convert", self._start, primary=True)
        self.convert_btn.pack(side="right")

        # options
        opt = tk.Frame(self.root, bg=CARD, padx=14, pady=12)
        opt.pack(fill="x", padx=18, pady=(10, 8))
        tk.Label(opt, text="Options", bg=CARD, fg=TEXT, font=FONT_B,
                 anchor="w").pack(fill="x")

        # -- pore character row
        r1 = tk.Frame(opt, bg=CARD)
        r1.pack(fill="x", pady=(8, 0))
        tk.Label(r1, text="Pore / void character:", bg=CARD, fg=MUTED,
                 font=FONT_SM).pack(side="left")
        self.pore_var = tk.StringVar(value="1")
        ttk.Combobox(r1, textvariable=self.pore_var, values=["1", "0"],
                     state="readonly", width=4, font=FONT_SM
                     ).pack(side="left", padx=(8, 10))

        # -- raw checkbox
        self.raw_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt, text="also write a .raw copy",
                       variable=self.raw_var, bg=CARD, fg=TEXT,
                       selectcolor=FIELD, activebackground=CARD,
                       activeforeground=TEXT, font=FONT_SM, bd=0,
                       highlightthickness=0, anchor="w"
                       ).pack(fill="x", pady=(6, 0))

        # -- output folder row
        r3 = tk.Frame(opt, bg=CARD)
        r3.pack(fill="x", pady=(6, 0))
        tk.Label(r3, text="Output folder:", bg=CARD, fg=MUTED,
                 font=FONT_SM).pack(side="left")
        self._btn(r3, "Browse…", self._browse_out).pack(side="right")
        self.out_var = tk.StringVar(value="")
        tk.Entry(r3, textvariable=self.out_var, bg=FIELD, fg=TEXT, bd=0,
                 insertbackground=TEXT, font=FONT_SM
                 ).pack(side="left", fill="x", expand=True, padx=(8, 8),
                        ipady=3)
        tk.Label(opt, bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w",
                 text="leave empty to write each .tif next to its .dat"
                 ).pack(fill="x", pady=(2, 0))

        # log
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="both", expand=True, padx=18, pady=(4, 6))
        self.log = tk.Text(wrap, bg="#0d1017", fg=TEXT, bd=0, font=FONT_SM,
                           wrap="word", height=10, padx=10, pady=8,
                           insertbackground=TEXT)
        sb = ttk.Scrollbar(wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        for tag, col in (("ok", GOOD), ("err", BAD), ("warn", WARN),
                         ("muted", MUTED), ("head", ACCENT)):
            self.log.tag_configure(tag, foreground=col)
        self.log.configure(state="disabled")

        # status
        st = tk.Frame(self.root, bg="#0d1017", height=32)
        st.pack(fill="x", side="bottom")
        st.pack_propagate(False)
        self.status = tk.Label(st, text="Ready — no files queued.",
                               bg="#0d1017", fg=MUTED, font=FONT_SM,
                               anchor="w", padx=14)
        self.status.pack(side="left", fill="x", expand=True)
        self.prog = ttk.Progressbar(st, mode="determinate", length=180)

        self._say("Drop .dat files above, choose the pore character, then "
                  "press Convert.\n", "muted")

    def _btn(self, parent, text, cmd, primary=False):
        bg = ACCENT if primary else FIELD
        hi = ACCENT_D if primary else "#2c3444"
        fg = "#0b1220" if primary else TEXT
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      activebackground=hi, activeforeground=fg, relief="flat",
                      font=FONT_B if primary else FONT_SM, bd=0, padx=14,
                      pady=6, cursor="hand2", disabledforeground=MUTED)
        b.bind("<Enter>", lambda e: b.configure(bg=hi)
               if str(b["state"]) != "disabled" else None)
        b.bind("<Leave>", lambda e: b.configure(bg=bg)
               if str(b["state"]) != "disabled" else None)
        return b

    # ------------------------------------------------------------- file list
    def _drop_enter(self, _e):
        self.drop.configure(highlightbackground=ACCENT, highlightcolor=ACCENT)
        self.drop_lbl.configure(fg=ACCENT)

    def _drop_leave(self, _e):
        self.drop.configure(highlightbackground=BORDER, highlightcolor=BORDER)
        self.drop_lbl.configure(fg=TEXT)

    def _on_drop(self, event):
        self._drop_leave(event)
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:                              # noqa: BLE001
            paths = event.data.split()
        self._add(paths)

    def _browse_files(self):
        paths = filedialog.askopenfilenames(
            title="Select .dat volumes",
            filetypes=[("ASCII voxel volumes", "*.dat"), ("All files", "*.*")])
        self._add(paths)

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select a folder containing .dat files")
        if d:
            self._add([d])

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_var.set(d)

    def _add(self, paths):
        found = collect_dat_files([p.strip("{}") for p in paths])
        new = [f for f in found if f not in self.files]
        skipped = len(paths) - len(found) if found else 0
        self.files += new
        if new:
            self._say(f"Queued {len(new)} file(s):\n", "head")
            for f in new:
                mb = os.path.getsize(f) / 1e6
                self._say(f"   • {os.path.basename(f)}  ({mb:,.1f} MB)\n")
        if not found:
            self._say("Nothing added — drop .dat files or a folder "
                      "containing them.\n", "warn")
        elif skipped > 0:
            self._say(f"({skipped} dropped item(s) had no .dat files)\n",
                      "muted")
        self._refresh_status()

    def _clear(self):
        self.files = []
        self._say("List cleared.\n", "muted")
        self._refresh_status()

    def _refresh_status(self):
        n = len(self.files)
        self.status.configure(
            text="Ready — no files queued." if not n
            else f"{n} file(s) queued — press Convert.")
        self.convert_btn.configure(state="normal" if n and not self.busy
                                   else "disabled")

    # ------------------------------------------------------------ conversion
    def _start(self):
        if self.busy or not self.files:
            return
        self.busy = True
        self.convert_btn.configure(state="disabled")
        self.prog.pack(side="right", padx=12)
        self.prog.configure(maximum=len(self.files), value=0)

        files = list(self.files)
        pore = self.pore_var.get()
        raw = self.raw_var.get()
        out_dir = self.out_var.get().strip() or None
        if out_dir and not os.path.isdir(out_dir):
            self._say(f"Output folder does not exist: {out_dir}\n", "err")
            self._finish()
            return

        def work():
            done = failed = 0
            for i, f in enumerate(files, 1):
                self.msgs.put(("status", f"Converting {os.path.basename(f)} "
                                         f"({i}/{len(files)})…"))
                try:
                    _, n, poro = convert(
                        f, pore, raw, out_dir,
                        log=lambda s: self.msgs.put(("log", s + "\n")))
                    done += 1
                except Exception as e:                 # noqa: BLE001
                    failed += 1
                    self.msgs.put(("err", f"  FAILED {os.path.basename(f)}: "
                                          f"{e}\n"))
                    traceback.print_exc()
                self.msgs.put(("prog", i))
            self.msgs.put(("done", (done, failed)))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "log":
                    self._say(payload)
                elif kind == "err":
                    self._say(payload, "err")
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "prog":
                    self.prog.configure(value=payload)
                elif kind == "done":
                    done, failed = payload
                    self._say(
                        f"\nFinished: {done} converted"
                        + (f", {failed} failed" if failed else "")
                        + ".  Import the .tif files with 'Select 3D image "
                          "file' in the app.\n",
                        "err" if failed else "ok")
                    self._finish()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish(self):
        self.busy = False
        self.prog.pack_forget()
        self._refresh_status()

    def _say(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:                                  # noqa: BLE001
        pass
    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    ConverterApp(root, preload=sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
