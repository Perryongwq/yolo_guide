# -*- mode: python ; coding: utf-8 -*-
"""
CT600 Vision Inspection System - PyInstaller Spec File
Builds a standalone Windows executable with all dependencies.
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# Collect all data and hidden imports for key packages
datas = []
binaries = []
hiddenimports = []

# Ultralytics (YOLO) - critical for model inference
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all('ultralytics')
datas += ultralytics_datas
binaries += ultralytics_binaries
hiddenimports += ultralytics_hiddenimports

# PyTorch
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
datas += torch_datas
binaries += torch_binaries
hiddenimports += torch_hiddenimports

# Torchvision
torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')
datas += torchvision_datas
binaries += torchvision_binaries
hiddenimports += torchvision_hiddenimports

# OpenCV
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')
datas += cv2_datas
binaries += cv2_binaries
hiddenimports += cv2_hiddenimports

# FastAPI and dependencies
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('starlette')
hiddenimports += collect_submodules('pydantic')
hiddenimports += collect_submodules('multipart')

# Jinja2 templates
hiddenimports += ['jinja2.ext']

# Pandas and dependencies
hiddenimports += collect_submodules('pandas')
hiddenimports += collect_submodules('openpyxl')

# Matplotlib - required by Ultralytics
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all('matplotlib')
datas += matplotlib_datas
binaries += matplotlib_binaries
hiddenimports += matplotlib_hiddenimports

# Add application-specific data files
# Backend files
# Note: app_fastapi.py uses get_base_path() which returns the executable directory
# when frozen, so models must be at root level, not under backend/
datas += [
    ('backend/models', 'models'),  # Copy to root level to match app_fastapi.py expectations
    ('backend/calibration', 'backend/calibration'),
    ('backend/app_fastapi.py', 'backend'),
]

# Frontend files
datas += [
    ('frontend/templates', 'frontend/templates'),
    ('frontend/static', 'frontend/static'),
    ('frontend/conf', 'frontend/conf'),
    ('frontend/commons', 'frontend/commons'),
    ('frontend/configreader', 'frontend/configreader'),
    ('frontend/util', 'frontend/util'),
    ('frontend/app.py', 'frontend'),
]

# Python-JOSE - required for frontend authentication
hiddenimports += collect_submodules('jose')

# Additional hidden imports for specific modules
hiddenimports += [
    'app_fastapi',
    'app',
    'configreader.configreader',
    'commons.config_utilities',
    'util.utilfunc',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'numpy.core._dtype_ctypes',
    'pkg_resources.py2_warn',
    'PIL',
    'PIL._imaging',
    'icecream',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
    'jose',
    'jose.jwe',
    'jose.jwt',
    'jose.constants',
    'jose.exceptions',
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        '_tkinter',
        'tk',
        'tcl',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CT600_Vision_System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an icon file here if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CT600_Vision_System',
)

