@echo off
REM ============================================================================
REM CT600 AI Vision Inspection System - Build Script
REM Creates a standalone Windows executable using PyInstaller
REM Includes email alert functionality and configurable email settings
REM ============================================================================

echo.
echo ========================================================================
echo CT600 AI Vision Inspection System - Executable Build Script
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

echo [1/6] Checking Python installation...
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed or not in PATH
    pause
    exit /b 1
)

echo [2/6] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__
if exist backend\__pycache__ rmdir /s /q backend\__pycache__
if exist frontend\__pycache__ rmdir /s /q frontend\__pycache__
echo Previous builds cleaned.
echo.

echo [3/6] Installing dependencies...
echo This may take several minutes on first run...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

echo [4/6] Preparing for build...
REM Note: Runtime folders (uploads, processed, results) will be created
REM in the dist folder at the root level, not under backend/
echo Ready for build.
echo.

echo [5/6] Building executable with PyInstaller...
echo This will take 5-15 minutes depending on your system...
echo Please be patient...
echo.
pyinstaller ct600_vision.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo ERROR: PyInstaller build failed
    pause
    exit /b 1
)
echo.

echo [6/6] Finalizing distribution...
REM Copy documentation to dist folder
if exist README.md copy README.md dist\CT600_Vision_System\README.txt
if exist CAMERA_OPTIMIZATIONS.md copy CAMERA_OPTIMIZATIONS.md dist\CT600_Vision_System\
if exist DOCKER_GUIDE.md copy DOCKER_GUIDE.md dist\CT600_Vision_System\

REM Copy models folder to root level (app_fastapi.py expects models/ at executable directory level)
echo Copying models folder to distribution root...
if exist dist\CT600_Vision_System\_internal\models (
    xcopy /E /I /Y dist\CT600_Vision_System\_internal\models dist\CT600_Vision_System\models
)

REM Copy config folder to root level (app_fastapi.py expects config/ at executable directory level)
echo Copying config folder to distribution root...
if exist dist\CT600_Vision_System\_internal\config (
    xcopy /E /I /Y dist\CT600_Vision_System\_internal\config dist\CT600_Vision_System\config
)

REM Create runtime directories in dist root (app_fastapi.py uses get_base_path() which returns executable directory)
echo Creating runtime directories in distribution root...
if not exist dist\CT600_Vision_System\uploads mkdir dist\CT600_Vision_System\uploads
if not exist dist\CT600_Vision_System\processed mkdir dist\CT600_Vision_System\processed
if not exist dist\CT600_Vision_System\results mkdir dist\CT600_Vision_System\results

echo.
echo ========================================================================
echo BUILD SUCCESSFUL!
echo ========================================================================
echo.
echo Executable location: dist\CT600_Vision_System\CT600_Vision_System.exe
echo.
echo File Structure:
echo   dist\CT600_Vision_System\
echo   ├── CT600_Vision_System.exe        (Main executable)
echo   ├── models\                        (YOLO models - copied from _internal)
echo   │   ├── 03_standard_model.pt
echo   │   ├── 15standard_model.pt
echo   │   ├── 18standard_model.pt
echo   │   └── [other model files]
echo   ├── config\                        (Email configuration - REQUIRED)
echo   │   ├── email_config.json          (Must be configured before use)
echo   │   └── README.md
echo   ├── uploads\                       (Runtime data - writable)
echo   ├── processed\                     (Runtime data - writable)
echo   ├── results\                       (Runtime data - writable)
echo   └── _internal\                     (Python runtime ^& bundled code)
echo       ├── backend\
echo       │   ├── app_fastapi.py
echo       │   └── sendAlert.py
echo       └── frontend\
echo           ├── templates\
echo           ├── static\
echo           └── app.py
echo.
echo You can now:
echo   1. Navigate to: dist\CT600_Vision_System\
echo   2. Run: CT600_Vision_System.exe
echo   3. Or create a desktop shortcut to the executable
echo.
echo The entire CT600_Vision_System folder can be:
echo   - Copied to other Windows machines
echo   - Distributed via USB drive
echo   - Deployed to production systems
echo.
echo System Requirements:
echo   - Windows 10/11 (64-bit)
echo   - USB Camera (DirectShow compatible)
echo   - 4GB RAM minimum (8GB recommended)
echo   - 2GB free disk space
echo.
echo NOTE: Results will be saved to results\[machine]\
echo       NOT in _internal\ folder
echo.
echo IMPORTANT: Email Configuration
echo   - The config\email_config.json file MUST be configured before use
echo   - Update email addresses in config\email_config.json
echo   - Application will NOT start without valid email configuration
echo   - Email alerts are sent for judgement mismatches:
echo     * Human: NG ^& AI: No Good
echo     * Human: NG ^& AI: Good
echo     * Human: G ^& AI: No Good
echo.
echo Features:
echo   - Supports item types: 03type, 15type, 18type, 21type, 31type, 32type
echo   - Excel exports include: Item Type, Pitch, Y-Difference, Judgements
echo   - Email alerts include: Item Type, Pitch, Machine Number, Username
echo.
echo ========================================================================
echo.
pause

