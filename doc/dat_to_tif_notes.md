# Notes on `dat_to_tif.py`

Companion explanation file for the ASCII-volume converter in the `tools/` folder.

This module converts `.dat` files — plain text volumes containing only the characters `0` and `1`,
one character per voxel — into 3-D `.tif` volumes that the Porous Media Analysis app can import
directly. It exists because some datasets (e.g. `RockSamples/Kikkis`) are distributed in that raw
text form, which no image library reads.

`tools/` is **not** part of the application source. Nothing here is imported by the app, so using or
editing this file never requires rebuilding the `.exe` or updating the copy installed on another PC.

## Important functions

- `read_dat(path, pore_char="1")`: reads the text file, strips CR/LF bytes, validates that only
  `'0'`/`'1'` remain, infers the cube side `n` from the voxel count and returns a boolean array
  (`True` = pore). The array is indexed `[z][y][x]`.
- `convert(path, pore_char, also_raw, out_dir, log)`: converts one file and writes the `.tif`
  (optionally also a headerless `.raw`). Returns `(out_path, n, porosity)`.
- `collect_dat_files(inputs)`: expands a mixed list of files and folders into a sorted list of
  `.dat` paths. Shared with the drag-and-drop window.
- `main(argv)`: the command-line entry point (see *Typical usage*).

## Input format

The shape is implicit in the line structure, not stored in a header: a cube of side `n` has `n`
characters per line and `n*n` lines, so line index = `z*n + y` and character position = `x`. The
converter derives `n` from the total voxel count and refuses the file if it is not a perfect cube,
rather than guessing a shape and producing a silently wrong volume.

## Output convention

The `.tif` is written as `uint8` with values `0` and `255`, array ordered `[z][y][x]` — exactly the
convention the app uses when it saves its own volumes (`np.transpose(image, (2, 1, 0))`, `True*255`).
Bright (255) is the pore/void phase. The app therefore imports the file, detects it as already
binary and reports the correct porosity with "keep as-is"; no inversion step is needed.

## Notes

- `--pore-is` selects which character is the void; the default is `1` because in the Kikkis file
  names the `_1_` field means "1 is the pore value". Passing the wrong value produces a phase-swapped
  volume, recoverable either by re-converting or by choosing "inverted" at import.
- Conversion is lossless and repeatable, so the original `.dat` files should be kept.
- Reading uses `np.fromfile` on the raw bytes and filters out CR/LF, which is far faster than
  parsing the text line by line — a 300³ file (27 MB) converts in about a second.
- `.raw` output is offered as a fallback: the app reads `.raw` too, but then the shape and datatype
  must be typed manually at import, so `.tif` is preferred.

## Typical usage

```
python tools\dat_to_tif.py                          # default: RockSamples\Kikkis
python tools\dat_to_tif.py "C:\some\folder"         # every .dat in that folder
python tools\dat_to_tif.py file1.dat file2.dat      # specific files
python tools\dat_to_tif.py <input> --pore-is 0      # if 0 is the void
python tools\dat_to_tif.py <input> --raw            # also write .raw copies
```

Each `.tif` is written next to its `.dat` unless an output folder is given. End-user instructions
are in `tools/HOW TO USE - DAT to TIF converter.txt`; the graphical front-end is documented in
`doc/dat_converter_gui_notes.md`.
