# -*- mode: python ; coding: utf-8 -*-
#
# Build a single-file Windows executable of the Porous Media Analysis UI.
#
#   .venv\Scripts\pyinstaller.exe PorousMediaAnalysis.spec --noconfirm
#
# The result is  dist\PorousMediaAnalysis.exe  — a self-contained onefile
# bundle that can be copied to another PC (no Python install required).

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# The analysis modules the UI imports as `logic.*`. They live in the `logic/`
# package (which has an __init__.py). collect_submodules pulls every one of
# them into the bundle so `import logic.Generators_3D` works inside the .exe.
project_modules = collect_submodules('logic')

hidden = project_modules + [
    'imageio', 'porespy', 'openpnm', 'skimage', 'scipy', 'pandas',
]

# Several scientific packages read their own version at import time via
# importlib.metadata.version(...) (e.g. imageio, porespy). PyInstaller does not
# bundle the *.dist-info metadata by default, so those calls raise
# PackageNotFoundError and the app fails to start. copy_metadata() ships the
# metadata folders so the version lookups succeed inside the frozen .exe.
metadata_datas = []
for _pkg in ('imageio', 'porespy', 'openpnm', 'scikit-image', 'scikit-learn',
             'numba', 'llvmlite', 'numpy', 'scipy', 'pandas', 'matplotlib',
             'edt', 'pyevtk', 'PyWavelets', 'tifffile', 'networkx', 'sympy',
             'h5py', 'pydicom', 'pillow'):
    try:
        metadata_datas += copy_metadata(_pkg)
    except Exception:
        pass  # package not installed / no metadata — skip it

a = Analysis(
    ['ui\\PorousMediaAnalysis_UI.py'],
    pathex=['.'],
    binaries=[],
    datas=[('assets', 'assets')] + metadata_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PorousMediaAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\icon.ico'],
)
