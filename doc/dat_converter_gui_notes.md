# Notes on `dat_converter_gui.py`

Companion explanation file for the drag-and-drop converter window in the `tools/` folder.

This is a small standalone Tkinter application that wraps `dat_to_tif.py`: you drop `.dat` files (or
folders containing them) onto the window, choose which character is the void, and press Convert. It
exists so that converting a volume does not require typing a command line.

Like the rest of `tools/`, it is **not** part of the application source — it imports only
`dat_to_tif.py`, never the app — so it can be changed freely without rebuilding the `.exe`.

## Layout

- **Drop zone** — accepts dropped files and folders; its border highlights while a drag hovers.
- **Add files… / Add folder… / Clear list** — fallback for when drag-and-drop is unavailable.
- **Options** — pore/void character (`1` or `0`), an optional `.raw` copy, and an output folder
  (empty means "write each `.tif` next to its `.dat`").
- **Log** — colour-coded transcript: queued files, per-file shape and porosity, errors, summary.
- **Status bar** — queue state, current file, and a determinate progress bar during conversion.

## Important methods

- `ConverterApp._on_drop(event)`: parses the dropped payload with `root.tk.splitlist`, which handles
  the Tcl brace-quoting used for paths containing spaces, then queues the `.dat` files found.
- `ConverterApp._add(paths)` / `_clear()`: maintain the queue, skipping duplicates and reporting when
  a dropped item contained no `.dat` files.
- `ConverterApp._start()`: validates the output folder and launches the conversion on a daemon
  thread, so the window stays responsive on large volumes.
- `ConverterApp._poll()`: drains a `queue.Queue` of `(kind, payload)` messages every 100 ms and
  applies them to the log, status bar and progress bar. This is the only place the worker's results
  touch Tk — worker threads never call Tk directly.
- `main()`: enables per-monitor DPI awareness, then creates a `TkinterDnD.Tk` root (or a plain
  `tk.Tk` if drag-and-drop is unavailable) and pre-loads any paths passed on the command line.

## Notes

- Drag-and-drop needs the optional `tkinterdnd2` package. The import is guarded: if it is missing,
  `HAVE_DND` is `False`, the drop bindings are skipped, and the window explains that the buttons
  should be used instead. The tool therefore never fails to start over a missing optional package.
- Because `sys.argv` is pre-loaded into the queue, dragging `.dat` files onto the launcher
  `tools/DAT to TIF converter.bat` in Explorer opens the window with those files already listed.
- The colour constants mirror the main application's dark theme so the two look like one product.
  They are duplicated here deliberately rather than imported, to keep the tool independent of the
  app's source.
- The `ttk.Combobox` needs `style.map(..., foreground=[("readonly", TEXT)])` in addition to
  `style.configure`; without the map the selected value is invisible on the dark background.
- Conversion errors are caught per file: one bad file is logged in red and the remaining files
  continue, with a final "N converted, M failed" summary.

## Deployment

The `tools/` folder is self-contained — nothing in it imports the application or the `logic/`
package — so it can be copied to another machine on its own. Two files support that:

- `tools/requirements-tools.txt` lists only what the converter needs (`numpy`, `imageio`,
  `tifffile`, plus optional `tkinterdnd2`), rather than the project's full scientific stack.
- `tools/DAT to TIF converter.bat` searches for an interpreter in order: the project venv
  (`..\.venv`), then `pythonw.exe` on `PATH`, then the `py` launcher. The same file therefore works
  inside the full project and in a standalone copy, and prints installation guidance if no Python
  is found.

## Typical usage

Double-click `tools/DAT to TIF converter.bat`, or run:

```
python tools\dat_converter_gui.py
python tools\dat_converter_gui.py file1.dat file2.dat    # pre-loaded
```

End-user instructions are in `tools/HOW TO USE - DAT to TIF converter.txt`; the conversion logic
itself is documented in `doc/dat_to_tif_notes.md`.
