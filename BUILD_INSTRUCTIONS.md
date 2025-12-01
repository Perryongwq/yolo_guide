# Building CT600 Vision System Executable

This guide explains how to build the CT600 Vision Inspection System as a Windows executable (EXE) with reload functionality enabled.

## Overview

The executable bundles both the backend (FastAPI) and frontend (FastAPI) applications into a single Windows executable that:
- Starts both services automatically
- Supports hot-reload for development (automatically restarts on code changes)
- Shows console output from both services
- Allows graceful shutdown with Ctrl+C

## Prerequisites

**IMPORTANT: All dependencies must be installed in your Python environment before building!**

- **Python 3.10 or higher** installed and added to PATH
- **PyInstaller** installed: `pip install pyinstaller`
- **All backend dependencies** installed:
  ```batch
  pip install -r backend/requirements_fastapi.txt
  ```
- **All frontend dependencies** installed:
  ```batch
  pip install -r frontend/requirements.txt
  ```
- **Sufficient disk space** (approximately 2-3 GB for build artifacts)

## Quick Build

1. **Install all dependencies** (if not already installed):
   ```batch
   pip install pyinstaller
   pip install -r backend/requirements_fastapi.txt
   pip install -r frontend/requirements.txt
   ```

2. **(Optional) Check dependencies** before building:
   ```batch
   check_dependencies.bat
   ```

3. **Open Command Prompt** in the project root directory

4. **Run the build script:**
   ```batch
   build_exe.bat
   ```

5. **Wait for the build to complete** (this may take several minutes)

6. **Find your executable** at: `dist\CT600_Vision_System.exe`

## Detailed Build Steps

### Installing Dependencies First

**Before building, you must install all dependencies in your Python environment:**

```batch
REM 1. Install PyInstaller
pip install pyinstaller

REM 2. Install backend dependencies
pip install -r backend/requirements_fastapi.txt

REM 3. Install frontend dependencies
pip install -r frontend/requirements.txt
```

**Note:** You can use a virtual environment if you prefer, but it's not required by the build script. The build script uses whatever Python environment is currently active.

### Checking Dependencies (Optional but Recommended)

Before building, you can run the dependency check script to verify everything is ready:

```batch
check_dependencies.bat
```

This script will:
- Verify Python is installed
- Check if PyInstaller is installed
- Verify backend dependencies are installed
- Verify frontend dependencies are installed
- Check that build files exist

If any checks fail, the script will show you what's missing and how to fix it.

### Building the Executable

The `build_exe.bat` script simplifies the build process:

```batch
build_exe.bat
```

This script will:
1. Check if Python is installed
2. Check if PyInstaller is installed (will show error if missing)
3. Verify build files exist (`build_exe.spec`, `launcher.py`)
4. Clean previous build artifacts
5. Build the executable using `build_exe.spec`
6. Output the executable to `dist\CT600_Vision_System.exe`

### Manual Build Process

If you prefer to build manually:

```batch
REM Clean previous build
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

REM Build executable
pyinstaller build_exe.spec --clean
```

## Running the Executable

### Method 1: Double-Click
Simply double-click `CT600_Vision_System.exe` in the `dist` folder.

### Method 2: Command Line (Recommended for Development)
Run from command line to see console output:
```batch
cd dist
CT600_Vision_System.exe
```

### What Happens When You Run

1. **Backend server** starts on `http://localhost:5000`
2. **Frontend server** starts on `http://localhost:5001` (starts 3 seconds after backend)
3. **Console window** shows output from both services with prefixes:
   - `[BACKEND]` for backend logs
   - `[FRONTEND]` for frontend logs

4. **Access the application** at: `http://localhost:5001/vision-inspection`

### Stopping the Application

Press **Ctrl+C** in the console window to gracefully stop both servers.

## Reload Functionality

The executable includes **reload functionality enabled**, which means:

- ✅ **Code changes are detected automatically**
- ✅ **Servers restart when you modify:**
  - Backend Python files (`.py` files in `backend/`)
  - Frontend Python files (`.py` files in `frontend/`)
  - Template files (`.html` files in `frontend/templates/`)
  - Configuration files

- ⚠️ **Note:** For reload to work properly in an executable:
  - The executable must extract files to a temporary directory
  - Code changes need to be made in the source directories (not the extracted temp files)
  - Changes to static files (CSS, JS) may require a browser refresh

### Reload Limitations

- Reload works best when running from source code
- In an executable, reload may be limited to files that are not deeply embedded
- If reload doesn't work as expected, restart the executable

## Project Structure in Executable

The executable includes:

```
CT600_Vision_System.exe
├── Backend application (port 5000)
│   ├── FastAPI server
│   ├── YOLO models (from backend/models)
│   └── Calibration files (from backend/calibration)
│
├── Frontend application (port 5001)
│   ├── FastAPI server
│   ├── Templates (from frontend/templates)
│   ├── Static files (from frontend/static)
│   └── Configuration (from frontend/conf)
│
└── All Python dependencies
    ├── FastAPI & Uvicorn
    ├── Ultralytics & PyTorch
    ├── OpenCV
    └── Other dependencies
```

## Prerequisites Checklist

Before building, ensure you have:

- [ ] Python 3.10+ installed and in PATH
- [ ] PyInstaller installed: `pip install pyinstaller`
- [ ] Backend dependencies installed: `pip install -r backend/requirements_fastapi.txt`
- [ ] Frontend dependencies installed: `pip install -r frontend/requirements.txt`

**Quick setup command (run from project root):**
```batch
pip install pyinstaller
pip install -r backend/requirements_fastapi.txt
pip install -r frontend/requirements.txt
```

## Troubleshooting

### Build Issues

**Problem: "Python is not installed or not in PATH"**
- Solution: Install Python 3.10+ and ensure it's added to PATH
- Verify with: `python --version`

**Problem: "PyInstaller is not installed"**
- Solution: Install PyInstaller: `pip install pyinstaller`
- Verify with: `python -c "import PyInstaller"`

**Problem: "Module not found" errors during build**
- Solution: Ensure all dependencies are installed in your Python environment
- Install missing dependencies:
  ```batch
  pip install -r backend/requirements_fastapi.txt
  pip install -r frontend/requirements.txt
  ```
- Check that all dependencies are listed in `requirements_fastapi.txt` and `requirements.txt`
- Add missing modules to the `hiddenimports` list in `build_exe.spec`

**Problem: Build takes too long or hangs**
- Solution: This is normal (PyTorch and dependencies are large)
- Ensure sufficient disk space (2-3 GB free)
- The build script now skips dependency installation, so it should be faster

### Runtime Issues

**Problem: "Port 5000 or 5001 already in use"**
- Solution: Close any other instances of the application
- Or change ports in `backend/app_fastapi.py` and `frontend/app.py`

**Problem: "Failed to load model"**
- Solution: Ensure `backend/models/` contains your YOLO model files
- Check that model files are included in `build_exe.spec` datas list

**Problem: "Module not found" when running executable**
- Solution: Add the missing module to `hiddenimports` in `build_exe.spec`
- Rebuild the executable

**Problem: Reload not working**
- Solution: Reload in executables has limitations
- Try restarting the executable after code changes
- For development, consider running from source instead: `python launcher.py`

### File Path Issues

**Problem: Templates or static files not found**
- Solution: Verify paths in `build_exe.spec` datas section
- Ensure relative paths are correct

## Customization

### Changing the Executable Name

Edit `build_exe.spec`:
```python
name='Your_Custom_Name',
```

### Adding an Icon

1. Create or obtain an `.ico` file
2. Edit `build_exe.spec`:
```python
icon='path/to/your/icon.ico',
```

### Disabling Console Window

Edit `build_exe.spec`:
```python
console=False,  # Change from True to False
```

**Note:** If you disable console, you won't see logs or be able to use Ctrl+C easily.

### Excluding Files from Build

Edit `build_exe.spec` datas list to remove unnecessary files and reduce executable size.

## File Sizes

- **Executable size:** Approximately 500 MB - 2 GB (depends on PyTorch and models)
- **Build artifacts:** Additional 1-2 GB during build process

The large size is due to:
- PyTorch framework
- YOLO/Ultralytics dependencies
- OpenCV libraries
- All Python dependencies

## Advanced Usage

### Running with Custom Ports

To change ports, modify the launcher script or create a configuration file.

### Development vs Production Builds

- **Development:** Use `reload=True` (current setup) - auto-restarts on code changes
- **Production:** Change `reload=False` in both `app_fastapi.py` and `app.py` for better performance

### Creating a Distribution Package

1. Build the executable using `build_exe.bat`
2. Copy the executable to a new folder
3. Include a README with usage instructions
4. Optionally create an installer using tools like Inno Setup or NSIS

## Support

For issues or questions:
- Check the main README.md for general application information
- Review build logs in the console output
- Check that all dependencies are correctly installed

## License

Proprietary - Internal Use Only


