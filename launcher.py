import subprocess
import sys
import os
import threading
import time
import signal
from pathlib import Path

# Get the directory where the script is located
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

# Global process references for cleanup
backend_process = None
frontend_process = None

def run_backend():
    """Start the backend server with reload enabled"""
    global backend_process
    try:
        os.chdir(BACKEND_DIR)
        backend_process = subprocess.Popen(
            [sys.executable, "app_fastapi.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("=" * 60)
        print("BACKEND SERVER STARTING on http://localhost:5000")
        print("Reload mode: ENABLED")
        print("=" * 60)
        
        # Stream output
        for line in backend_process.stdout:
            print(f"[BACKEND] {line.rstrip()}")
            
    except Exception as e:
        print(f"Error starting backend: {e}")
        sys.exit(1)

def run_frontend():
    """Start the frontend server with reload enabled"""
    global frontend_process
    try:
        # Wait a bit for backend to start
        time.sleep(3)
        
        os.chdir(FRONTEND_DIR)
        frontend_process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("=" * 60)
        print("FRONTEND SERVER STARTING on http://localhost:5001")
        print("Reload mode: ENABLED")
        print("=" * 60)
        
        # Stream output
        for line in frontend_process.stdout:
            print(f"[FRONTEND] {line.rstrip()}")
            
    except Exception as e:
        print(f"Error starting frontend: {e}")
        sys.exit(1)

def cleanup():
    """Cleanup function to stop both processes"""
    print("\n" + "=" * 60)
    print("Shutting down servers...")
    print("=" * 60)
    
    if backend_process:
        try:
            backend_process.terminate()
            backend_process.wait(timeout=5)
        except:
            try:
                backend_process.kill()
            except:
                pass
    
    if frontend_process:
        try:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)
        except:
            try:
                frontend_process.kill()
            except:
                pass
    
    print("Servers stopped. Goodbye!")
    sys.exit(0)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    cleanup()

def main():
    """Main entry point"""
    # Register signal handlers
    if sys.platform == 'win32':
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("CT600 Vision Inspection System")
    print("=" * 60)
    print("\nStarting both backend and frontend servers...")
    print("Reload mode: ENABLED (code changes will auto-restart servers)")
    print("\nAccess the application at: http://localhost:5001/vision-inspection")
    print("Backend API: http://localhost:5000")
    print("\nPress Ctrl+C to stop the servers\n")
    
    # Start backend in a thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in a thread
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend_process and backend_process.poll() is not None:
                print("[ERROR] Backend process died!")
                cleanup()
            if frontend_process and frontend_process.poll() is not None:
                print("[ERROR] Frontend process died!")
                cleanup()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        cleanup()

if __name__ == "__main__":
    main()


