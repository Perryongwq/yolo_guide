@echo off
echo ============================================================
echo Checking Build Dependencies
echo ============================================================
echo.

REM Check if Python is installed
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo    FAIL: Python is not installed or not in PATH
    echo    Please install Python 3.10+ and add it to PATH
    set ERRORS=1
) else (
    python --version
    echo    OK: Python is installed
)
echo.

REM Check PyInstaller
echo [2/6] Checking PyInstaller...
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo    FAIL: PyInstaller is not installed
    echo    Install with: pip install pyinstaller
    set ERRORS=1
) else (
    python -c "import PyInstaller; print('    OK: PyInstaller', PyInstaller.__version__)"
)
echo.

REM Check backend dependencies
echo [3/6] Checking backend dependencies...
python -c "import fastapi, uvicorn, ultralytics, torch, cv2, pandas, numpy, openpyxl" >nul 2>&1
if errorlevel 1 (
    echo    FAIL: Some backend dependencies are missing
    echo    Install with: pip install -r backend/requirements_fastapi.txt
    set ERRORS=1
) else (
    echo    OK: Backend dependencies are installed
)
echo.

REM Check frontend dependencies
echo [4/6] Checking frontend dependencies...
python -c "import fastapi, jinja2, jose, orjson, aiofiles, itsdangerous" >nul 2>&1
if errorlevel 1 (
    echo    FAIL: Some frontend dependencies are missing
    echo    Install with: pip install -r frontend/requirements.txt
    set ERRORS=1
) else (
    echo    OK: Frontend dependencies are installed
)
echo.

REM Check build files
echo [5/6] Checking build files...
if not exist "build_exe.spec" (
    echo    FAIL: build_exe.spec not found
    set ERRORS=1
) else (
    echo    OK: build_exe.spec found
)

if not exist "launcher.py" (
    echo    FAIL: launcher.py not found
    set ERRORS=1
) else (
    echo    OK: launcher.py found
)
echo.

REM Check project structure
echo [6/6] Checking project structure...
if not exist "backend" (
    echo    FAIL: backend directory not found
    set ERRORS=1
) else (
    echo    OK: backend directory found
)

if not exist "frontend" (
    echo    FAIL: frontend directory not found
    set ERRORS=1
) else (
    echo    OK: frontend directory found
)
echo.

REM Summary
echo ============================================================
if defined ERRORS (
    echo RESULT: ERRORS FOUND
    echo ============================================================
    echo.
    echo Please fix the errors above before building.
    echo.
    echo Quick fix commands:
    echo   pip install pyinstaller
    echo   pip install -r backend/requirements_fastapi.txt
    echo   pip install -r frontend/requirements.txt
    echo.
    pause
    exit /b 1
) else (
    echo RESULT: ALL CHECKS PASSED
    echo ============================================================
    echo.
    echo All dependencies are installed and ready!
    echo You can now run: build_exe.bat
    echo.
    pause
    exit /b 0
)

