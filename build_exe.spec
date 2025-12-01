# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Get the base directory where this spec file is located
BASE_DIR = Path(SPECPATH)

# Data files to include
datas = [
    ('frontend/templates', 'templates'),
    ('frontend/static', 'static'),
    ('frontend/conf', 'conf'),
    ('backend/models', 'models'),
    ('backend/calibration', 'calibration'),
]

# Hidden imports (modules PyInstaller might miss)
hiddenimports = [
    # Uvicorn
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets.websockets_impl',
    'uvicorn.loops.auto',
    'uvicorn.loops.asyncio',
    
    # FastAPI
    'fastapi',
    'fastapi.responses',
    'fastapi.staticfiles',
    'fastapi.templating',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'fastapi.exceptions',
    'starlette.middleware.sessions',
    'starlette.exceptions',
    
    # Multipart
    'multipart',
    
    # Jinja2
    'jinja2',
    'jinja2.ext',
    
    # JOSE
    'jose',
    'jose.jwt',
    'jose.jwe',
    'jose.constants',
    'cryptography',
    
    # ORJSON
    'orjson',
    
    # Ultralytics and PyTorch
    'ultralytics',
    'ultralytics.models',
    'ultralytics.utils',
    'torch',
    'torchvision',
    'torch.nn',
    'torch.cuda',
    
    # OpenCV
    'cv2',
    
    # Pandas and Excel
    'pandas',
    'openpyxl',
    
    # NumPy
    'numpy',
    
    # Other dependencies
    'aiofiles',
    'itsdangerous',
    'python_dateutil',
    'configreader',
    'icecream',
    'commons',
    'util',
]

a = Analysis(
    ['launcher.py'],
    pathex=[str(BASE_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CT600_Vision_System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console window for output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an icon file path here if you have one: 'path/to/icon.ico'
)


