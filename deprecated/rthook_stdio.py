# PyInstaller runtime hook.
#
# In a windowed build (console=False) Python sets sys.stdout / sys.stderr /
# sys.stdin to None. Any library that writes to them directly then crashes with
# "'NoneType' object has no attribute 'write'". OpenPNM's drainage/invasion
# simulations do exactly this via their tqdm progress bars (which target
# sys.stderr), so we make sure all three standard streams are always writable.
import os
import sys

_devnull = open(os.devnull, "w")

if sys.stdout is None:
    sys.stdout = _devnull
if sys.stderr is None:
    sys.stderr = _devnull
if sys.stdin is None:
    sys.stdin = open(os.devnull, "r")
