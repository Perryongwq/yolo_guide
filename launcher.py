"""
CT600 Vision Inspection System - Standalone Launcher
Starts both backend and frontend servers and opens browser automatically.
"""
import os
import sys
import time
import threading
import webbrowser
import logging
import signal
from pathlib import Path
import uvicorn
import requests
from contextlib import redirect_stdout, redirect_stderr
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flags for graceful shutdown
shutdown_event = threading.Event()
backend_server = None
frontend_server = None

def get_base_path():
    """Get the base path, works both in development and when frozen by PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).parent


def check_port_available(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0


def wait_for_service(url, max_attempts=30, delay=1):
    """Wait for a service to become available"""
    logger.info(f"Waiting for service at {url}...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"Service at {url} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    return False


def run_backend_server():
    """Run the backend FastAPI server"""
    try:
        base_path = get_base_path()
        
        # Determine paths based on frozen vs development mode
        if getattr(sys, 'frozen', False):
            # Frozen executable: code is in _internal/backend, data should be at executable root
            # app_fastapi.py uses get_base_path() which returns executable directory
            backend_code_path = base_path / "backend"
            # Data path is at executable directory level (not under backend/)
            backend_data_path = Path(sys.executable).parent
        else:
            # Development mode: everything in backend/
            backend_code_path = base_path / "backend"
            backend_data_path = backend_code_path
        
        # Create runtime folders in the data directory (where results will be saved)
        # app_fastapi.py expects these at BASE_PATH level (executable directory when frozen)
        os.makedirs(backend_data_path / "uploads", exist_ok=True)
        os.makedirs(backend_data_path / "processed", exist_ok=True)
        os.makedirs(backend_data_path / "results", exist_ok=True)
        
        # Add backend code path to Python path
        if str(backend_code_path) not in sys.path:
            sys.path.insert(0, str(backend_code_path))
        
        # Don't change working directory - app_fastapi.py handles paths relative to BASE_PATH
        # os.chdir(backend_data_path)  # Removed - let app_fastapi.py handle paths
        
        logger.info(f"Backend code path: {backend_code_path}")
        logger.info(f"Backend data path: {backend_data_path}")
        logger.info("Starting Backend API on http://localhost:5000")
        
        # Import backend app
        import app_fastapi
        app = app_fastapi.app
        
        config = uvicorn.Config(
            app,
            host="localhost",
            port=5000,
            log_level="info",
            access_log=False
        )
        global backend_server
        backend_server = uvicorn.Server(config)
        backend_server.run()
        
    except Exception as e:
        logger.error(f"Backend server error: {e}", exc_info=True)
        sys.exit(1)


def run_frontend_server():
    """Run the frontend FastAPI server"""
    try:
        base_path = get_base_path()
        
        # Frontend code is always in _internal/frontend (or ./frontend in dev)
        frontend_path = base_path / "frontend"
        
        # Add frontend to path
        if str(frontend_path) not in sys.path:
            sys.path.insert(0, str(frontend_path))
        
        # Change working directory to frontend (for template/static file access)
        os.chdir(frontend_path)
        
        logger.info(f"Frontend path: {frontend_path}")
        logger.info("Starting Frontend UI on http://localhost:5001")
        
        # Import frontend app
        import app as frontend_app
        app = frontend_app.app
        
        config = uvicorn.Config(
            app,
            host="localhost",
            port=5001,
            log_level="info",
            access_log=False
        )
        global frontend_server
        frontend_server = uvicorn.Server(config)
        frontend_server.run()
        
    except Exception as e:
        logger.error(f"Frontend server error: {e}", exc_info=True)
        sys.exit(1)


def open_browser(url, delay=3):
    """Open browser after a delay"""
    time.sleep(delay)
    if not shutdown_event.is_set():
        logger.info(f"Opening browser to {url}")
        webbrowser.open(url)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("\nShutdown signal received. Stopping services...")
    shutdown_event.set()
    
    # Stop servers gracefully
    if backend_server:
        backend_server.should_exit = True
    if frontend_server:
        frontend_server.should_exit = True
    
    sys.exit(0)


def main():
    """Main launcher function"""
    print("=" * 60)
    print("CT600 Vision Inspection System - Standalone Edition")
    print("=" * 60)
    print()
    
    # Check if ports are available
    if not check_port_available(5000):
        logger.error("Port 5000 is already in use. Please close the application using it.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    if not check_port_available(5001):
        logger.error("Port 5001 is already in use. Please close the application using it.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start backend server in a separate thread
        backend_thread = threading.Thread(target=run_backend_server, daemon=True)
        backend_thread.start()
        logger.info("Backend server thread started")
        
        # Wait a bit for backend to initialize
        time.sleep(2)
        
        # Start frontend server in a separate thread
        frontend_thread = threading.Thread(target=run_frontend_server, daemon=True)
        frontend_thread.start()
        logger.info("Frontend server thread started")
        
        # Wait for both services to be ready
        backend_ready = wait_for_service("http://localhost:5000/health", max_attempts=30)
        if not backend_ready:
            logger.error("Backend service failed to start within timeout period")
            sys.exit(1)
        
        frontend_ready = wait_for_service("http://localhost:5001/health", max_attempts=30)
        if not frontend_ready:
            logger.error("Frontend service failed to start within timeout period")
            sys.exit(1)
        
        print()
        print("=" * 60)
        print("✓ All services are running!")
        print("=" * 60)
        print()
        print("Backend API:        http://localhost:5000")
        print("Frontend UI:        http://localhost:5001")
        print("Vision Inspection:  http://localhost:5001/vision-inspection")
        print()
        print("The browser will open automatically...")
        print("Press Ctrl+C to stop all services")
        print("=" * 60)
        print()
        
        # Open browser in a separate thread
        browser_thread = threading.Thread(
            target=open_browser,
            args=("http://localhost:5001/login",),
            daemon=True
        )
        browser_thread.start()
        
        # Keep the main thread alive
        while not shutdown_event.is_set():
            time.sleep(1)
            
            # Check if threads are still alive
            if not backend_thread.is_alive():
                logger.error("Backend server thread died unexpectedly")
                break
            if not frontend_thread.is_alive():
                logger.error("Frontend server thread died unexpectedly")
                break
        
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
    finally:
        shutdown_event.set()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

