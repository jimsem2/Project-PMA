# -*- coding: utf-8 -*-
"""
Convert ASCII '0'/'1' voxel .dat volumes into 3-D .tif volumes readable by
the Porous Media Analysis app.

This is a STANDALONE utility: it is not part of the application source, so
converting files here does NOT require rebuilding or updating the .exe.

Input format (auto-detected)
----------------------------
A text file containing only the characters '0' and '1' plus line breaks.
Each line holds one row of a slice, so a cube of side n has n characters per
line and n*n lines (line index = z*n + y, character index = x).

In the Kikkis dataset the '_1_' in a file name means '1 is the pore/void
value', which is the default assumed here (--pore-is 1).

Output
------
A 3-D TIFF (uint8, 0 or 255) written in exactly the convention the app uses
for its own saved volumes: bright (255) = pore/void, dark (0) = solid, with
the array stored as [z][y][x]. The app therefore imports it directly, detects
it as already binary, and reports the correct porosity with 'keep as-is' —
no inversion needed.

Usage
-----
    python tools\\dat_to_tif.py                        # convert RockSamples\\Kikkis
    python tools\\dat_to_tif.py <file-or-folder> ...   # convert specific inputs
    python tools\\dat_to_tif.py <input> --pore-is 0    # if '0' is the void
    python tools\\dat_to_tif.py <input> --raw          # also write a .raw copy

A drag-and-drop window for the same job is available in
tools\\dat_converter_gui.py.
"""
import os
import sys

import numpy as np


def read_dat(path, pore_char="1"):
    """Read an ASCII 0/1 volume file and return a boolean array (True = pore).

    The returned array is indexed [z][y][x]: z is the slice (block of lines),
    y the row within the slice and x the position along the line.
    """
    b = np.fromfile(path, dtype=np.uint8)
    b = b[(b != 13) & (b != 10)]                 # drop CR / LF

    if not ((b == ord("0")) | (b == ord("1"))).all():
        raise ValueError(
            f"{os.path.basename(path)}: file contains characters other than "
            "'0'/'1'; this converter only handles plain binary voxel dumps.")

    n = round(b.size ** (1.0 / 3.0))
    if n ** 3 != b.size:
        raise ValueError(
            f"{os.path.basename(path)}: {b.size:,} voxels is not a perfect "
            "cube, so the shape cannot be inferred automatically.")

    return b.reshape(n, n, n) == ord(pore_char)  # True = pore (void)


def convert(path, pore_char="1", also_raw=False, out_dir=None, log=print):
    """Convert one .dat file to .tif. Returns (out_path, n, porosity)."""
    pore = read_dat(path, pore_char)
    n = pore.shape[0]
    porosity = float(pore.mean())

    stem = os.path.splitext(os.path.basename(path))[0]
    dest = out_dir or os.path.dirname(os.path.abspath(path))
    out_tif = os.path.join(dest, stem + ".tif")

    arr = pore.astype(np.uint8) * 255            # 255 = pore, 0 = solid
    import imageio.v2 as iio
    iio.volsave(out_tif, arr)

    if also_raw:
        arr.tofile(os.path.join(dest, stem + ".raw"))
        log(f"    also wrote {stem}.raw  (shape {n},{n},{n} · uint8)")

    log(f"  {os.path.basename(path)}  ->  {stem}.tif")
    log(f"    shape {n}x{n}x{n}   porosity (True=pore) = {porosity:.4f}   "
        f"solid fraction = {1 - porosity:.4f}")
    return out_tif, n, porosity


def collect_dat_files(inputs):
    """Expand a list of files/folders into a sorted list of .dat files."""
    targets = []
    for a in inputs:
        if os.path.isdir(a):
            targets += [os.path.join(a, f) for f in sorted(os.listdir(a))
                        if f.lower().endswith(".dat")]
        elif os.path.isfile(a) and a.lower().endswith(".dat"):
            targets.append(a)
    return targets


def main(argv):
    pore_char = "1"
    if "--pore-is" in argv:
        i = argv.index("--pore-is")
        if i + 1 < len(argv) and argv[i + 1] in ("0", "1"):
            pore_char = argv[i + 1]
            del argv[i:i + 2]
        else:
            print("--pore-is needs a value of 0 or 1")
            return 2
    also_raw = "--raw" in argv
    args = [a for a in argv if not a.startswith("--")]

    if not args:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args = [os.path.join(here, "RockSamples", "Kikkis")]

    targets = collect_dat_files(args)
    if not targets:
        print("No .dat files found in:", ", ".join(args))
        return 1

    print(f"Converting {len(targets)} file(s); '{pore_char}' = pore/void\n")
    failed = 0
    for t in targets:
        try:
            convert(t, pore_char, also_raw)
        except Exception as e:                    # noqa: BLE001
            failed += 1
            print(f"  FAILED {os.path.basename(t)}: {e}")
    print("\nDone. Import the .tif files with 'Select 3D image file' in the app.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
