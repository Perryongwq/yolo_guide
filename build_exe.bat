@echo off
echo ============================================================
echo Building CT600 Vision System Executable
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
echo Checking for PyInstaller...
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed
    echo Please install PyInstaller first:
    echo   pip install pyinstaller
    echo.
    echo Also ensure all dependencies are installed:
    echo   pip install -r backend/requirements_fastapi.txt
    echo   pip install -r frontend/requirements.txt
    pause
    exit /b 1
)

echo PyInstaller found. Using current Python environment.
echo.

REM Check if spec file exists
if not exist "build_exe.spec" (
    echo ERROR: build_exe.spec file not found!
    echo Please ensure build_exe.spec is in the project root.
    pause
    exit /b 1
)

REM Check if launcher.py exists
if not exist "launcher.py" (
    echo ERROR: launcher.py file not found!
    echo Please ensure launcher.py is in the project root.
    pause
    exit /b 1
)

REM Clean previous build
echo Step 1: Cleaning previous build artifacts...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

REM Build executable
echo.
echo Step 2: Building executable with PyInstaller...
echo This may take several minutes, please wait...
echo.
pyinstaller build_exe.spec --clean

if errorlevel 1 (
    echo.
    echo ERROR: Build failed! Check the output above for details.
    echo.
    echo Common issues:
    echo   - Missing dependencies. Install all requirements:
    echo       pip install -r backend/requirements_fastapi.txt
    echo       pip install -r frontend/requirements.txt
    echo   - Missing PyInstaller. Install it:
    echo       pip install pyinstaller
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Build Complete!
echo ============================================================
echo.
echo Executable location: dist\CT600_Vision_System.exe
echo.
echo The executable includes:
echo   - Backend server (port 5000) with reload enabled
echo   - Frontend server (port 5001) with reload enabled
echo   - All necessary dependencies and resources
echo.
echo To run the application, double-click CT600_Vision_System.exe
echo or run it from command line to see console output.
echo.
echo Access the application at: http://localhost:5001/vision-inspection
echo.
pause
