# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect full package trees for the tricky scientific packages
porespy_d, porespy_b, porespy_h = collect_all('porespy')
openpnm_d, openpnm_b, openpnm_h = collect_all('openpnm')
imageio_d, imageio_b, imageio_h = collect_all('imageio')
skimage_d, skimage_b, skimage_h = collect_all('skimage')
edt_d, edt_b, edt_h = collect_all('edt')
numba_d, numba_b, numba_h = collect_all('numba')
llvmlite_d, llvmlite_b, llvmlite_h = collect_all('llvmlite')
# openpnm's fluid-property stack ships large .tsv/.json data files (e.g.
# chemicals/Identifiers/*.tsv) that must be bundled or drainage/invasion
# simulations crash with FileNotFoundError at runtime.
chemicals_d, chemicals_b, chemicals_h = collect_all('chemicals')
thermo_d, thermo_b, thermo_h = collect_all('thermo')
fluids_d, fluids_b, fluids_h = collect_all('fluids')

all_datas    = (porespy_d + openpnm_d + imageio_d + skimage_d + edt_d + numba_d + llvmlite_d
                + chemicals_d + thermo_d + fluids_d)
all_binaries = (porespy_b + openpnm_b + imageio_b + skimage_b + edt_b + numba_b + llvmlite_b
                + chemicals_b + thermo_b + fluids_b)
all_hidden   = (
    porespy_h + openpnm_h + imageio_h + skimage_h + edt_h + numba_h + llvmlite_h
    + chemicals_h + thermo_h + fluids_h
    + collect_submodules('scipy')
    + collect_submodules('pandas')
    + collect_submodules('h5py')
    + collect_submodules('pydicom')
    + collect_submodules('openpyxl')
    + collect_submodules('pywt')
    + collect_submodules('sklearn')
    + [
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
        'pyevtk',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.simpledialog',
    ]
)

a = Analysis(
    ['app_gui.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_stdio.py'],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PorousMediaAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PorousMediaAnalysis',
)
