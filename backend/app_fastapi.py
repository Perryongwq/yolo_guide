from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import os
import sys
import cv2
import pandas as pd
import base64
from datetime import datetime
from ultralytics import YOLO
import threading
import time
import numpy as np
import logging
import uvicorn
import torch
import asyncio
import json
from sendAlert import sendErrorAlertEmail
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine if running as compiled executable or as script
def get_base_path():
    """Get base path for file storage - works for both script and exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        # Get the directory where the executable is located
        base_path = os.path.dirname(sys.executable)
        logger.info(f"Running as executable. Base path: {base_path}")
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Running as script. Base path: {base_path}")
    return base_path

# Get base path
BASE_PATH = get_base_path()

# Paths - Now always relative to BASE_PATH
UPLOAD_FOLDER = os.path.join(BASE_PATH, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_PATH, "processed")
RESULTS_FOLDER = os.path.join(BASE_PATH, "results")
EXCEL_FILE = os.path.join(PROCESSED_FOLDER, "measurement_results.xlsx")

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

logger.info(f"Directories configured:")
logger.info(f"  UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"  PROCESSED_FOLDER: {PROCESSED_FOLDER}")
logger.info(f"  RESULTS_FOLDER: {RESULTS_FOLDER}")

# Global storage for latest processing results
latest_processing_results = {
    "y_diff_microns": None,
    "judgement": None,
    "processed_timestamp": None,
    "item_type": None,
    "pitch": None,
}


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle"""
    # Startup
    logger.info("Application starting up...")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Base path: {BASE_PATH}")
    yield
    # Shutdown
    global camera
    logger.info("Application shutting down...")
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        logger.info("Camera released on shutdown")


# Initialize FastAPI app as API-only backend
app = FastAPI(
    title="CT600 Vision Guide API",
    version="1.0.0",
    description="Backend API for CT600 Vision Inspection System",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom middleware to handle large file uploads
@app.middleware("http")
async def process_file_upload(request: Request, call_next):
    # Log request size if available
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        logger.info(f"Request size: {content_length} bytes ({size_mb:.2f} MB)")

    # Set a more reasonable limit (50MB)
    max_size = 50 * 1024 * 1024  # 50MB

    if content_length and int(content_length) > max_size:
        logger.warning(f"Request exceeds size limit: {size_mb:.2f} MB")
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": f"Request too large: {size_mb:.2f} MB. Maximum allowed: 50 MB",
            },
        )

    # Continue with the request
    response = await call_next(request)
    return response


# Your existing middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(
        f"Content-Type header: {request.headers.get('Content-Type', 'Not set')}"
    )

    # Continue with the request
    response = await call_next(request)
    return response


# Note: Static files and templates removed - frontend handles all UI

# Load YOLO models dynamically based on item_type and pitch
# Get the models directory relative to BASE_PATH
MODELS_DIR = os.path.join(BASE_PATH, "models")

# Dictionary to cache loaded models (key: "item_type_pitch", e.g., "15type_standard")
loaded_models = {}

# Class names for each model type
class_names_03type = [
    "block1_edge",
    "block2_edge",
    "block1",
    "block2",
    "cal_mark",
]

class_names_15type = [
    "block1_edge15",
    "block2_edge15",
    "block1_15",
    "block2_15",
    "cal_mark",
]

class_names_18type = [
    "block1",
    "block1_edge",
    "block2",
    "block2_edge",
    "cal_mark",
]

# 21type and 31type use the same class names as 18type
class_names_21type = class_names_18type
class_names_31type = class_names_18type

# Store class names in a dictionary for easy access
class_names = {
    "03type": class_names_03type,
    "15type": class_names_15type,
    "18type": class_names_18type,
    "21type": class_names_21type,
    "31type": class_names_31type,
}

def get_model(item_type: str, pitch: str = "standard") -> YOLO:
    """
    Get or load a YOLO model based on item_type and pitch.
    Models are cached after first load for better performance.
    
    Args:
        item_type: Item type (e.g., "03type", "15type", "18type", "21type", "31type", "32type")
        pitch: Pitch type (e.g., "standard", "narrow"). Use None or "standard" for types that don't support pitch.
    
    Returns:
        YOLO model instance
    """
    # For types that don't support pitch (e.g., 32type, 03type), use item_type only
    if item_type in ["32type", "03type"] or pitch is None:
        model_key = item_type
        # For 03type, try both "03type_model.pt" and "03standard_model.pt" formats
        if item_type == "03type":
            # First try the simple format
            model_filename = "03type_model.pt"
        else:
            model_filename = "32type_model.pt"
    else:
        # For types that support pitch, construct key and filename
        model_key = f"{item_type}_{pitch}"
        # Extract number from item_type (e.g., "15type" -> "15")
        type_number = item_type.replace("type", "")
        model_filename = f"{type_number}{pitch}_model.pt"
    
    # Check if model is already loaded
    if model_key in loaded_models:
        logger.info(f"Using cached model: {model_key}")
        return loaded_models[model_key]
    
    # Load model
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # For 03type, try alternative filename if first one doesn't exist
    if item_type == "03type" and not os.path.exists(model_path):
        # Try the pitch-based format as fallback
        alternative_filename = "03standard_model.pt"
        alternative_path = os.path.join(MODELS_DIR, alternative_filename)
        if os.path.exists(alternative_path):
            logger.info(f"03type model not found at {model_path}, trying alternative: {alternative_path}")
            model_path = alternative_path
            model_filename = alternative_filename
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please ensure the model file exists."
        )
    
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Log the actual class names from the model for debugging
    if hasattr(model, 'names') and model.names:
        model_class_names = [model.names[i] for i in sorted(model.names.keys())]
        logger.info(f"Model's actual class names: {model_class_names}")
        logger.info(f"Model's class count: {len(model_class_names)}")
    
    # Cache the loaded model
    loaded_models[model_key] = model
    logger.info(f"Model cached with key: {model_key}")
    
    return model

# Constants
MICRONS_PER_PIXEL = 2.3
BLOCK1_OFFSET = 0.0
BLOCK2_OFFSET = 0.0
MEASUREMENT_OFFSET_MICRONS = (
    5.0  # Offset to apply to final measurement (e.g., +5 to correct camera calibration)
)
judgement_criteria = {"good": 10, "acceptable": 20}

# Email configuration for judgement mismatch alerts
# Load from config file - raises error if file not found or invalid
def load_email_config():
    """
    Load email configuration from config/email_config.json file.
    Raises FileNotFoundError if file not found.
    Raises ValueError if JSON is invalid or required keys are missing.
    """
    config_path = os.path.join(BASE_PATH, "config", "email_config.json")
    required_keys = ["RTO0006", "RTO0010", "RTO0013_01", "RTO0013_02", "RTO0013_03"]
    
    if not os.path.exists(config_path):
        error_msg = f"Email config file not found at {config_path}. Please create config/email_config.json with the required fields."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in email config file {config_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error reading email config file {config_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate that all required keys exist
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        error_msg = f"Missing required keys in email config: {', '.join(missing_keys)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Email config loaded successfully from: {config_path}")
    return config

# Load email configuration at startup
try:
    EMAIL_CONFIG = load_email_config()
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Failed to load email configuration: {e}")
    logger.error("Application cannot start without valid email configuration.")
    raise

def get_email_config_df():
    """Get email configuration as DataFrame for sendAlert module"""
    return pd.DataFrame([EMAIL_CONFIG])

def check_judgement_mismatch(human_judgement: str, ai_judgement: str):
    """
    Check for judgement mismatches and return error information.
    
    Conditions to alert:
    1. humanJudgement = NG & AI Judgement = No Good
    2. humanJudgement = NG & AI Judgement = Good
    3. humanJudgement = G & AI Judgement = No Good
    
    Args:
        human_judgement: Human judgement value ("G" or "NG")
        ai_judgement: AI judgement value ("Good", "Acceptable", or "No Good")
    
    Returns:
        dict with keys:
            - has_mismatch: bool - True if mismatch detected
            - error_message: str - Error message describing the mismatch, empty if no mismatch
    """
    try:
        # Normalize judgement values for comparison
        human_judgement_upper = human_judgement.upper().strip() if human_judgement else ""
        ai_judgement_normalized = ai_judgement.strip() if ai_judgement else ""
        
        # Check for the three alert conditions
        has_mismatch = False
        error_message = ""
        
        if human_judgement_upper == "NG" and ai_judgement_normalized == "No Good":
            has_mismatch = True
            error_message = "Judgement Mismatch: Human Judgement (NG) does not match AI Judgement (No Good)"
        elif human_judgement_upper == "NG" and ai_judgement_normalized == "Good":
            has_mismatch = True
            error_message = "Judgement Mismatch: Human Judgement (NG) does not match AI Judgement (Good)"
        elif human_judgement_upper == "G" and ai_judgement_normalized == "No Good":
            has_mismatch = True
            error_message = "Judgement Mismatch: Human Judgement (G) does not match AI Judgement (No Good)"
        
        return {
            "has_mismatch": has_mismatch,
            "error_message": error_message,
            "human_judgement": human_judgement,
            "ai_judgement": ai_judgement
        }
    except Exception as e:
        logger.error(f"Error in check_judgement_mismatch: {e}")
        return {
            "has_mismatch": False,
            "error_message": "",
            "human_judgement": human_judgement,
            "ai_judgement": ai_judgement
        }

def check_and_send_judgement_alert(human_judgement: str, ai_judgement: str, details: dict):
    """
    Check for judgement mismatches and send email alerts.
    
    Conditions to alert:
    1. humanJudgement = NG & AI Judgement = No Good
    2. humanJudgement = NG & AI Judgement = Good
    3. humanJudgement = G & AI Judgement = No Good
    
    Args:
        human_judgement: Human judgement value ("G" or "NG")
        ai_judgement: AI judgement value ("Good", "Acceptable", or "No Good")
        details: Dictionary with additional information (machine_number, username, y_diff, etc.)
    """
    try:
        # Log input values for debugging
        logger.info(f"[ALERT CHECK] Human Judgement: '{human_judgement}', AI Judgement: '{ai_judgement}'")
        
        # Normalize judgement values for comparison
        human_judgement_upper = human_judgement.upper().strip() if human_judgement else ""
        ai_judgement_normalized = ai_judgement.strip() if ai_judgement else ""
        
        logger.info(f"[ALERT CHECK] Normalized - Human: '{human_judgement_upper}', AI: '{ai_judgement_normalized}'")
        
        # Check for the three alert conditions
        should_alert = False
        alert_reason = ""
        
        if human_judgement_upper == "NG" and ai_judgement_normalized == "No Good":
            should_alert = True
            alert_reason = "Human Judgement: NG, AI Judgement: No Good"
            logger.info(f"[ALERT CHECK] Condition 1 matched: NG & No Good")
        elif human_judgement_upper == "NG" and ai_judgement_normalized == "Good":
            should_alert = True
            alert_reason = "Human Judgement: NG, AI Judgement: Good"
            logger.info(f"[ALERT CHECK] Condition 2 matched: NG & Good")
        elif human_judgement_upper == "G" and ai_judgement_normalized == "No Good":
            should_alert = True
            alert_reason = "Human Judgement: G, AI Judgement: No Good"
            logger.info(f"[ALERT CHECK] Condition 3 matched: G & No Good")
        else:
            logger.info(f"[ALERT CHECK] No alert condition matched. Human: '{human_judgement_upper}', AI: '{ai_judgement_normalized}'")
        
        if should_alert:
            # Prepare email details
            email_details = {
                "Alert Reason": alert_reason,
                "Machine Number": details.get("machine_number", "N/A"),
                "Username": details.get("username", "N/A"),
                "Item Type": details.get("item_type", "N/A"),
                "Pitch": details.get("pitch", "N/A"),
                "Y-Difference (microns)": details.get("y_diff", "N/A"),
                "Image Name": details.get("image_name", "N/A"),
                "Checked Date and Time": details.get("checked_datetime", "N/A"),
            }
            
            # Send email alert
            try:
                emaildf = get_email_config_df()
                logger.info(f"[ALERT] Email config DataFrame created. Config: {EMAIL_CONFIG}")
                logger.info(f"[ALERT] Email category in config: '{EMAIL_CONFIG.get('RTO0010', 'NOT FOUND')}'")
                
                error_type = f"Judgement Mismatch Alert - {alert_reason}"
                app_name = "CT600 AI Vision Inspection"
                
                logger.info(f"[ALERT] Sending judgement mismatch alert: {alert_reason}")
                logger.info(f"[ALERT] Using email category: 'judgement_alert'")
                
                sendErrorAlertEmail(
                    emaildf,
                    "judgement_alert",
                    error_type,
                    app_name,
                    email_details
                )
                logger.info(f"[ALERT] Judgement mismatch alert sent successfully")
            except Exception as email_error:
                logger.error(f"[ALERT] Error sending email: {email_error}", exc_info=True)
                raise
        else:
            logger.debug(f"No alert needed. Human: {human_judgement}, AI: {ai_judgement}")
            
    except Exception as e:
        logger.error(f"Error in check_and_send_judgement_alert: {e}")
        # Don't raise exception - we don't want email failures to break the save process

# Camera setup - Initialize as None for lazy loading
camera = None
lock = threading.Lock()
frame = None


# Helper function to initialize camera with optimized settings
def init_camera_optimized():
    """Initialize camera with DirectShow backend for faster performance on Windows"""
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for Windows
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    cam.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
    return cam


# Pydantic models for request/response
class ImageSubmission(BaseModel):
    image_url: str
    machine_number: str


class ManualSubmission(BaseModel):
    aligned_image: str
    manual_lines: list


class AlignedImageSubmission(BaseModel):
    aligned_image: str


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "Bad request. Please check your data format.",
            "error_code": 400,
            "details": str(exc),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 413:
        logger.error(f"413 error: Request entity too large")
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": "File too large. Maximum size is 50MB. Please compress your image or reduce quality.",
                "error_code": 413,
            },
        )
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "error_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": f"Server error: {str(exc)}"},
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(
        f"Content-Type header: {request.headers.get('Content-Type', 'Not set')}"
    )

    # Check content length if available
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        logger.info(f"Request size: {content_length} bytes ({size_mb:.2f} MB)")

        # Early validation for large requests
        if int(content_length) > 100 * 1024 * 1024:  # 100MB limit
            logger.warning(f"Request exceeds size limit: {size_mb:.2f} MB")
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "message": f"Request too large: {size_mb:.2f} MB. Maximum allowed: 100 MB",
                },
            )

    response = await call_next(request)
    return response


@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        global frame, camera
        while True:
            # Check if camera is available
            if camera is None or not camera.isOpened():
                logger.warning("Camera not available in video_feed, reinitializing...")
                camera = init_camera_optimized()
                if not camera.isOpened():
                    logger.error("Failed to reinitialize camera in video_feed")
                    break

            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            # Encode with lower quality for faster streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/capture")
async def capture_image():
    global frame, camera
    logger.info("Capture image endpoint called")

    if camera is None or not camera.isOpened():
        camera = init_camera_optimized()
        if not camera.isOpened():
            logger.error("Unable to access camera")
            raise HTTPException(status_code=500, detail="Unable to access the camera.")

    with lock:
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            raise HTTPException(status_code=500, detail="Failed to capture image.")

        captured_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
        success = cv2.imwrite(captured_path, frame)

        if not success or not os.path.exists(captured_path):
            logger.error("Failed to write captured image to disk")
            raise HTTPException(
                status_code=500, detail="Failed to save captured image."
            )

        # Verify file size
        file_size = os.path.getsize(captured_path)
        logger.info(
            f"Image captured and saved to {captured_path}, size: {file_size} bytes"
        )

        if file_size == 0:
            logger.error("Captured image is empty")
            raise HTTPException(status_code=500, detail="Captured image is empty.")

    # Small delay to ensure file is completely written
    await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05 for faster response

    return {
        "success": True,
        "image_path": f"/uploads/captured_image.png",
        "proceed_to_crop": True,
    }


@app.post("/reconnect_camera")
async def reconnect_camera():
    global camera
    logger.info("Reconnecting camera")

    # Run camera operations in thread pool to avoid blocking
    def _reconnect():
        global camera

        # Release existing camera if open (with timeout)
        if camera is not None:
            try:
                if camera.isOpened():
                    camera.release()
                    logger.info("Released existing camera connection")
            except Exception as e:
                logger.warning(f"Error releasing camera (ignoring): {e}")
            camera = None

        # Small delay to ensure camera is fully released
        time.sleep(0.05)

        # Initialize new camera connection with optimized settings
        new_camera = init_camera_optimized()

        if new_camera.isOpened():
            logger.info("Camera reconnected successfully")
            return new_camera
        else:
            logger.error("Failed to open camera")
            return None

    # Execute in thread pool
    loop = asyncio.get_event_loop()
    new_cam = await loop.run_in_executor(None, _reconnect)

    if new_cam is None:
        logger.error("Failed to reconnect camera")
        camera = None
        raise HTTPException(status_code=500, detail="Failed to reconnect camera.")

    camera = new_cam
    return {"success": True, "message": "Camera reconnected successfully!"}


@app.post("/disconnect_camera")
async def disconnect_camera():
    global camera
    logger.info("Disconnecting camera")

    # Run camera release in thread pool for faster response
    def _disconnect():
        global camera
        if camera is not None:
            try:
                if camera.isOpened():
                    camera.release()
                    logger.info("Camera disconnected successfully")
            except Exception as e:
                logger.warning(f"Error disconnecting camera (ignoring): {e}")
            camera = None
            return True
        else:
            logger.info("Camera already disconnected")
            camera = None
            return False

    loop = asyncio.get_event_loop()
    was_open = await loop.run_in_executor(None, _disconnect)

    if was_open:
        return {"success": True, "message": "Camera disconnected successfully!"}
    else:
        return {"success": True, "message": "Camera was already disconnected."}


@app.get("/uploads/{filename}")
async def uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"Serving file: {file_path}, size: {os.path.getsize(file_path)} bytes")

    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/processed/{filename}")
async def processed_file(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"Processed file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(
        f"Serving processed file: {file_path}, size: {os.path.getsize(file_path)} bytes"
    )

    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/results/{machine_number}/{filename}")
async def results_file(machine_number: str, filename: str):
    """Serve saved result images from results folder"""
    file_path = os.path.join(RESULTS_FOLDER, machine_number, filename)
    if not os.path.exists(file_path):
        logger.error(f"Result file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(
        f"Serving result file: {file_path}, size: {os.path.getsize(file_path)} bytes"
    )

    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/save-image")
async def save_image(request: Request):
    logger.info(f"Save-image route called")

    try:
        data = await request.json()
        if not data:
            logger.error("No JSON data received")
            raise HTTPException(status_code=400, detail="No data received.")

        image_url = data.get("image_url")
        machine_number = data.get("machine_number", "").strip().lower()
        username = data.get("username", "Unknown")  # Get username from request
        human_judgement = data.get("human_judgement", "")  # Get human judgement from request

        # Validate machine number format
        if (
            not machine_number.startswith("ct")
            or not machine_number[2:].isdigit()
            or len(machine_number) != 5
        ):
            logger.error(f"Invalid machine number format: {machine_number}")
            raise HTTPException(
                status_code=400,
                detail="Invalid machine number format. Use 'ct' followed by 3 digits.",
            )

        if not image_url:
            logger.error("Image URL not provided")
            raise HTTPException(status_code=400, detail="Image URL not provided.")

        # Decode base64 image with size validation
        try:
            if "," in image_url:
                header, encoded = image_url.split(",", 1)
            else:
                encoded = image_url

            # Check base64 size before decoding
            encoded_size_mb = (
                len(encoded) * 3 / 4 / (1024 * 1024)
            )  # Approximate decoded size
            logger.info(f"Base64 encoded size: ~{encoded_size_mb:.2f} MB")

            if encoded_size_mb > 40:  # Leave some buffer from 50MB limit
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large: ~{encoded_size_mb:.2f} MB. Please compress the image.",
                )

            decoded_image = base64.b64decode(encoded)
            logger.info(f"Image decoded successfully, size: {len(decoded_image)} bytes")

        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            raise HTTPException(
                status_code=400, detail=f"Image decoding failed: {str(e)}"
            )

        # Save to results folder - using absolute path from BASE_PATH
        result_folder = os.path.join(RESULTS_FOLDER, machine_number)
        os.makedirs(result_folder, exist_ok=True)
        logger.info(f"Result folder created/verified: {result_folder}")

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{machine_number}_{timestamp}.png"
        image_path = os.path.join(result_folder, image_filename)

        # Save image
        with open(image_path, "wb") as f:
            f.write(decoded_image)

        logger.info(f"Image saved to: {image_path}")

        # Get the latest processing results
        y_diff = latest_processing_results.get("y_diff_microns", "N/A")
        judgement = latest_processing_results.get("judgement", "N/A")
        item_type = latest_processing_results.get("item_type", "N/A")
        pitch = latest_processing_results.get("pitch", "N/A")

        # Prepare new row with all required information
        new_row = {
            "Username": username,
            "Image Name": image_filename,
            "Item Type": item_type,
            "Pitch": pitch,
            "Y-Difference (microns)": y_diff,
            "AI Judgement": judgement,
            "Checked Date and Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Machine Number": machine_number.upper(),
            "Human Judgement": human_judgement,
            "Saved Path": image_path,
            "Microns per Pixel": latest_processing_results.get("microns_per_pixel", "N/A"),
        }

        # Create machine-specific Excel file
        excel_filename = f"{machine_number}_results.xlsx"
        excel_path = os.path.join(result_folder, excel_filename)

        try:
            if os.path.exists(excel_path):
                # Append to existing Excel file
                df_existing = pd.read_excel(excel_path)
                df_updated = pd.concat(
                    [df_existing, pd.DataFrame([new_row])], ignore_index=True
                )
                logger.info(f"Appending to existing Excel file: {excel_path}")
            else:
                # Create new Excel file
                df_updated = pd.DataFrame([new_row])
                logger.info(f"Creating new Excel file: {excel_path}")

            df_updated.to_excel(excel_path, index=False)
            logger.info(f"Excel file updated successfully: {excel_path}")

        except Exception as e:
            logger.error(f"Failed to update Excel file: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save to Excel: {str(e)}"
            )

        # Check for judgement mismatches and send email alerts
        judgement_mismatch_info = None
        try:
            logger.info(f"[SAVE-IMAGE] Checking alert conditions - Human: '{human_judgement}', AI: '{judgement}'")
            alert_details = {
                "machine_number": machine_number.upper(),
                "username": username,
                "item_type": item_type,
                "pitch": pitch,
                "y_diff": y_diff,
                "image_name": image_filename,
                "checked_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            check_and_send_judgement_alert(human_judgement, judgement, alert_details)
            
            # Check for judgement mismatch to include in response
            judgement_mismatch_info = check_judgement_mismatch(human_judgement, judgement)
        except Exception as e:
            logger.error(f"Failed to send judgement alert: {e}", exc_info=True)
            # Don't fail the save operation if alert fails
            # Still check for mismatch even if alert failed
            try:
                judgement_mismatch_info = check_judgement_mismatch(human_judgement, judgement)
            except Exception as check_error:
                logger.error(f"Failed to check judgement mismatch: {check_error}")

        # Prepare response
        response_data = {
            "success": True,
            "message": "Image and result saved!",
            "excel_path": excel_path,
            "image_path": image_path,
            "judgement": judgement,
            "human_judgement": human_judgement,
        }
        
        # Add judgement mismatch information if there is a mismatch
        if judgement_mismatch_info and judgement_mismatch_info.get("has_mismatch", False):
            response_data["judgement_mismatch"] = {
                "has_error": True,
                "error_message": judgement_mismatch_info.get("error_message", ""),
            }
        else:
            response_data["judgement_mismatch"] = {
                "has_error": False,
                "error_message": "",
            }
        
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in save_image: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/manual-submit")
async def manual_submit(request: Request):
    logger.info(f"Manual-submit route called")

    try:
        data = await request.json()
        if not data:
            logger.error("No JSON data received in manual-submit")
            raise HTTPException(status_code=400, detail="No data received.")

        aligned_image_data = data.get("aligned_image")
        manual_lines = data.get("manual_lines")

        if not aligned_image_data or not manual_lines or len(manual_lines) != 2:
            logger.error("Missing or invalid data in manual-submit")
            raise HTTPException(status_code=400, detail="Missing or invalid data.")

        # Decode image with size validation
        try:
            if "," in aligned_image_data:
                header, encoded = aligned_image_data.split(",", 1)
            else:
                encoded = aligned_image_data

            # Check size before decoding
            encoded_size_mb = len(encoded) * 3 / 4 / (1024 * 1024)
            logger.info(f"Manual submit image size: ~{encoded_size_mb:.2f} MB")

            image_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

        except Exception as e:
            logger.error(f"Image processing failed in manual-submit: {e}")
            raise HTTPException(
                status_code=400, detail=f"Image processing failed: {str(e)}"
            )

        microns_per_pixel = MICRONS_PER_PIXEL
        h, w = image.shape[:2]
        y1 = int(manual_lines[0]["y1"])  # block1_edge y
        y2 = int(manual_lines[1]["y1"])  # block2_edge y
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        # Draw straight (non-slanted) lines across the image width
        cv2.line(image, (0, y1), (w - 1, y1), (255, 0, 0), 2)  # block1_edge
        cv2.line(image, (0, y2), (w - 1, y2), (0, 255, 255), 2)  # block2_edge

        # Signed difference: block1_edge - block2_edge (no abs here)
        y_diff_pixels = y1 - y2
        y_diff_microns = (
            y_diff_pixels * microns_per_pixel
        ) + MEASUREMENT_OFFSET_MICRONS

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Keep judgement on magnitude (unchanged logic)
        if y_diff_microns < 0:
            judgement = "Good"
            judgement_color = (0, 255, 0)
        elif 0 <= y_diff_microns <= 10:
            judgement = "Acceptable"
            judgement_color = (0, 165, 255)
        else:
            judgement = "No Good"
            judgement_color = (0, 0, 255)

        # Annotations
        # Position text above the blue line (block1_edge, which is y1)
        text_y = y1 - 80  # 80 pixels above block1_edge
        text_x = w // 2 + 250  # Shifted to the right
        cv2.putText(
            image,
            f"{y_diff_microns:.2f} microns (b1-b2)",
            (text_x - 50, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Judgement: {judgement}",
            (text_x - 50, text_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            judgement_color,
            2,
        )
        cv2.putText(
            image,
            f"Checked on: {current_datetime}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            image,
            "Manual Judgement",
            (text_x - 120, mid_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"{microns_per_pixel:.2f} um/pixel",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        processed_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
        cv2.imwrite(processed_path, image)
        logger.info(f"Processed image saved to: {processed_path}")

        # Store results in global variable for later use when saving
        global latest_processing_results
        latest_processing_results = {
            "y_diff_microns": round(y_diff_microns, 2),
            "judgement": judgement,
            "processed_timestamp": current_datetime,
            "item_type": "Manual",  # Manual submission doesn't have item_type
            "pitch": "N/A",  # Manual submission doesn't have pitch
        }
        logger.info(f"Stored processing results: {latest_processing_results}")

        # Save to Excel
        try:
            df = pd.DataFrame(
                [
                    {
                        "Image Name": "manual_drawn_image.png",
                        "Y-Difference (microns)": y_diff_microns,
                        "Judgement": judgement,
                        "Checked Date and Time": current_datetime,
                        "Microns per Pixel": "Manual Entry",
                    }
                ]
            )
            df.to_excel(EXCEL_FILE, index=False)
            logger.info("Manual judgement saved to Excel")
        except Exception as e:
            logger.error(f"Failed to save manual judgement to Excel: {e}")

        return {"processed_image_url": f"/processed/processed_image.png"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in manual_submit: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/test-upload")
async def test_upload(request: Request):
    """Test endpoint to debug upload issues"""
    logger.info(f"Test upload called")

    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            image_data = data.get("aligned_image", "")
            size_mb = len(image_data) * 3 / 4 / (1024 * 1024)
            logger.info(f"JSON data received, image size: ~{size_mb:.2f} MB")
            return {
                "success": True,
                "message": f"JSON upload test successful, size: {size_mb:.2f} MB",
            }
        else:
            form_data = await request.form()
            form_image = form_data.get("aligned_image", "")
            size_mb = len(str(form_image)) * 3 / 4 / (1024 * 1024)
            logger.info(f"Form data received, image size: ~{size_mb:.2f} MB")
            return {
                "success": True,
                "message": f"Form upload test successful, size: {size_mb:.2f} MB",
            }
    except Exception as e:
        logger.error(f"Test upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "CT600 Vision Guide API",
        "version": "1.0.0",
        "status": "running",
        "base_path": BASE_PATH,
        "frontend_url": "http://localhost:5001/vision-inspection",
        "endpoints": {
            "health": "/health",
            "video_feed": "/video_feed",
            "capture": "/capture",
            "reconnect_camera": "/reconnect_camera",
            "process_image": "/",
            "manual_submit": "/manual-submit",
            "save_image": "/save-image",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "CT600 Vision API",
        "timestamp": datetime.now().isoformat(),
        "base_path": BASE_PATH,
        "folders": {
            "uploads": UPLOAD_FOLDER,
            "processed": PROCESSED_FOLDER,
            "results": RESULTS_FOLDER,
        },
    }


@app.post("/")
async def index_post(request: Request):
    logger.info(f"Index POST route called")

    try:
        # Clear previous files
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        # Try to get the aligned_image, item_type, and pitch from JSON or form data
        aligned_image_data = None
        item_type = "15type"  # Default to 15type for backward compatibility
        pitch = "standard"  # Default to standard for backward compatibility
        try:
            # First try JSON data
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                json_data = await request.json()
                if json_data:
                    aligned_image_data = json_data.get("aligned_image")
                    item_type = json_data.get("item_type", "15type")
                    pitch = json_data.get("pitch", "standard")
                    logger.info(f"Successfully retrieved aligned_image from JSON data, item_type: {item_type}, pitch: {pitch}")
            else:
                # Fallback to form data
                form_data = await request.form()
                aligned_image_data = form_data.get("aligned_image")
                item_type = form_data.get("item_type", "15type")
                pitch = form_data.get("pitch", "standard")
                logger.info(f"Successfully retrieved aligned_image from form data, item_type: {item_type}, pitch: {pitch}")
        except Exception as parse_error:
            logger.error(f"Failed to parse request data: {parse_error}")
            # If both fail, try to get raw data
            try:
                raw_data = await request.body()
                raw_text = raw_data.decode("utf-8")
                logger.info(f"Retrieved raw data, length: {len(raw_text)}")
                # Try to extract base64 data from raw form data
                if "data:image" in raw_text:
                    import re

                    base64_match = re.search(
                        r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", raw_text
                    )
                    if base64_match:
                        aligned_image_data = (
                            f"data:image/png;base64,{base64_match.group(1)}"
                        )
                        logger.info("Extracted base64 data from raw data")
            except Exception as raw_error:
                logger.error(f"Failed to parse raw data: {raw_error}")

        if not aligned_image_data:
            logger.error("No image data provided in index POST")
            raise HTTPException(status_code=400, detail="No image data provided.")

        # Decode and validate image
        try:
            if "," in aligned_image_data:
                header, encoded = aligned_image_data.split(",", 1)
            else:
                encoded = aligned_image_data

            # Size check
            encoded_size_mb = len(encoded) * 3 / 4 / (1024 * 1024)
            logger.info(f"Processing image size: ~{encoded_size_mb:.2f} MB")

            decoded_image = base64.b64decode(encoded)
            nparr = np.frombuffer(decoded_image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

        except Exception as e:
            logger.error(f"Failed to decode image in index: {e}")
            raise HTTPException(
                status_code=400, detail=f"Failed to decode image: {str(e)}"
            )

        # Validate item_type and get the appropriate model and class names
        # For types that don't support pitch (e.g., 32type, 03type), ignore pitch parameter
        if item_type in ["32type", "03type"]:
            pitch = None  # These types don't use pitch
        
        # Validate pitch for types that require it
        if item_type in ["15type", "18type", "21type", "31type"]:
            if pitch not in ["standard", "narrow"]:
                logger.warning(f"Invalid pitch: {pitch} for {item_type}, defaulting to standard")
                pitch = "standard"
        
        # Validate item_type and get class names
        if item_type not in class_names:
            logger.warning(f"Invalid item_type: {item_type}, defaulting to 15type")
            item_type = "15type"
            pitch = "standard"
        
        # Get the model using lazy loading
        try:
            # For types that don't use pitch (32type, 03type), don't pass pitch (or pass None)
            if item_type in ["32type", "03type"]:
                selected_model = get_model(item_type, None)
            else:
                selected_model = get_model(item_type, pitch)
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise HTTPException(status_code=404, detail=f"Model file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
        # Get class names - prefer model's actual class names, fallback to our mapping
        selected_class_names = class_names.get(item_type, [])
        
        # Get actual class names from the model if available
        if hasattr(selected_model, 'names') and selected_model.names:
            model_class_names = [selected_model.names[i] for i in sorted(selected_model.names.keys())]
            logger.info(f"Model's actual class names: {model_class_names}")
            
            # Check if our mapping matches the model's class names
            if selected_class_names and selected_class_names != model_class_names:
                logger.warning(f"Class name mismatch for {item_type}!")
                logger.warning(f"  Expected (from mapping): {selected_class_names}")
                logger.warning(f"  Actual (from model): {model_class_names}")
                logger.warning(f"  Using model's actual class names for processing")
                # Use model's actual class names
                selected_class_names = model_class_names
            elif not selected_class_names:
                # No mapping found, use model's class names
                logger.info(f"No class name mapping for {item_type}, using model's class names")
                selected_class_names = model_class_names
        else:
            logger.warning(f"Model does not have class names attribute, using mapping: {selected_class_names}")
        
        logger.info(f"Using model: {item_type} with pitch: {pitch}, class names: {selected_class_names}")

        # YOLO prediction
        results = selected_model.predict(source=image, conf=0.25, save=False)

        block1_edge_y = block2_edge_y = None
        block1_box_y = block2_box_y = None
        calibration_marker_width_px = None
        microns_per_pixel = MICRONS_PER_PIXEL

        # First pass: Find calibration marker to calculate microns_per_pixel (matching PyQt5 code)
        for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
            x_center, y_center, width, height = box
            label = selected_class_names[int(cls.item())]
            
            if label == "cal_mark":
                # Convert to scalar if it's a numpy array/tensor (matching PyQt5 code)
                if hasattr(width, 'item'):
                    calibration_marker_width_px = width.item()
                else:
                    calibration_marker_width_px = float(width)
                break  # Found calibration marker

        # Calculate microns per pixel from calibration marker if detected (matching PyQt5 code)
        if calibration_marker_width_px:
            microns_per_pixel = 1000.0 / calibration_marker_width_px
            logger.info(
                f"[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px, microns/px = {microns_per_pixel:.2f}"
            )
            print(
                f"[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px, microns/px = {microns_per_pixel:.2f}"
            )

            if microns_per_pixel > 10:
                logger.warning(
                    "Microns per pixel too high, suggesting focus adjustment"
                )
                return {
                    "error": True,
                    "error_message": "Fail to capture work guide and insertion guide",
                    "reason": "Microns per pixel too high, suggesting focus adjustment",
                }
        else:
            logger.warning(f"cal_mark not detected, using default: {microns_per_pixel:.2f} um/pixel")
            return {
                "error": True,
                "error_message": "Fail to capture work guide and insertion guide",
                "reason": "cal_mark not detected",
            }

        # Second pass: Calculate edge positions using calibrated microns_per_pixel (matching PyQt5 code)
        for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
            x_center, y_center, width, height = box
            label = selected_class_names[int(cls.item())]
            logger.info(f"[DEBUG] cls: {cls}, index: {int(cls.item())}, label: {label}")

            # Handle all item types: 03type, 15type, 18type, 21type, and 31type
            # Check for block1_edge variants: "block1_edge", "block1_edge15"
            if label in ["block1_edge", "block1_edge15"]:
                edge_y = int(y_center + height / 2)
                # Apply offset and store edge position using calibrated microns_per_pixel (matching PyQt5 code)
                block1_edge_y = edge_y + (BLOCK1_OFFSET / microns_per_pixel)
                # Draw longer line from x_center - 300 to x_center + 300 (matching PyQt5 code)
                cv2.line(
                    image,
                    (int(x_center - 300), edge_y),
                    (int(x_center + 300), edge_y),
                    (255, 0, 0),  # Red in BGR (matching PyQt5 code)
                    2,
                )

            # Check for block2_edge variants: "block2_edge", "block2_edge15"
            elif label in ["block2_edge", "block2_edge15"]:
                edge_y = int(y_center + height / 2)
                # Apply offset and store edge position using calibrated microns_per_pixel (matching PyQt5 code)
                block2_edge_y = edge_y + (BLOCK2_OFFSET / microns_per_pixel)
                # For 31type, shift the yellow line 100 pixels to the left
                line_x_center = int(x_center - 100) if item_type == "31type" else int(x_center)
                # Draw longer line from line_x_center - 300 to line_x_center + 300 (matching PyQt5 code)
                cv2.line(
                    image,
                    (line_x_center - 300, edge_y),
                    (line_x_center + 300, edge_y),
                    (0, 255, 255),  # Yellow in BGR (matching PyQt5 code)
                    2,
                )

            # Check for block1 body variants: "block1", "block1_15" (but not "block1_edge")
            elif label in ["block1", "block1_15"]:
                block1_box_y = int(y_center + height / 2)

            # Check for block2 body variants: "block2", "block2_15" (but not "block2_edge")
            elif label in ["block2", "block2_15"]:
                block2_box_y = int(y_center + height / 2)

        # Check if both edge positions are available
        if block1_edge_y is None or block2_edge_y is None:
            logger.warning("One or more edges not detected")
            return {
                "error": True,
                "error_message": "Fail to capture work guide and insertion guide",
                "reason": "One or more edges not detected",
            }

        # Calculate measurements
        y_diff_pixels = block1_edge_y - block2_edge_y
        y_diff_microns = (
            y_diff_pixels * microns_per_pixel
        ) + MEASUREMENT_OFFSET_MICRONS
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Judgement logic
        if y_diff_microns < judgement_criteria["good"]:
            judgement = "Good"
            judgement_color = (0, 255, 0)
        elif y_diff_microns < judgement_criteria["acceptable"]:
            judgement = "Acceptable"
            judgement_color = (0, 165, 255)
        else:
            judgement = "No Good"
            judgement_color = (0, 0, 255)

        # Add annotations
        # Display microns per pixel on image (top left) - matching PyQt5 format
        cal_text = f"{microns_per_pixel:.2f} um/pixel"
        cv2.putText(
            image,
            cal_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # Yellow in BGR (matching PyQt5 code)
            2,
        )
        
        # Calculate position for text (between the two edges) - matching PyQt5 code
        text_x = image.shape[1] // 2 + 250
        text_y = int((block1_edge_y + block2_edge_y) / 2)
        
        # Draw Y-difference measurement first (at top) - shifted up by 100 pixels
        diff_text = f"{y_diff_microns:.2f} microns"
        cv2.putText(
            image,
            diff_text,
            (text_x - 100, text_y - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        
        # Draw judgment text below Y-difference - matching PyQt5 code
        judgment_text = f"Judgment: {judgement}"
        cv2.putText(
            image,
            judgment_text,
            (text_x - 100, text_y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            judgement_color,
            2,
        )
        cv2.putText(
            image,
            f"Checked on: {current_datetime}",
            (10, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
        )

        # Save processed image
        processed_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
        cv2.imwrite(processed_path, image)
        logger.info(f"Processed image saved to: {processed_path}")

        # Store results in global variable for later use when saving
        global latest_processing_results
        latest_processing_results = {
            "y_diff_microns": round(y_diff_microns, 2),
            "judgement": judgement,
            "processed_timestamp": current_datetime,
            "microns_per_pixel": round(microns_per_pixel, 2),
            "calibration_marker_width_px": round(calibration_marker_width_px, 2),
            "item_type": item_type,
            "pitch": pitch,
        }
        logger.info(f"Stored processing results: {latest_processing_results}")

        # Save results to Excel
        try:
            results_df = pd.DataFrame(
                [
                    {
                        "Image Name": "aligned_image.png",
                        "Y-Difference (microns)": y_diff_microns,
                        "Judgement": judgement,
                        "Checked Date and Time": current_datetime,
                        "Microns per Pixel": round(microns_per_pixel, 2),
                    }
                ]
            )
            results_df.to_excel(EXCEL_FILE, index=False)
            logger.info("Results saved to Excel")
        except Exception as e:
            logger.error(f"Failed to save results to Excel: {e}")

        return {
            "processed_image_url": f"/processed/processed_image.png",
            "show_final_confirm_popup": True,
            "final_message": "Please confirm visually for guide position.",
            "microns_per_pixel": round(microns_per_pixel, 2),
            "calibration_marker_width_px": round(calibration_marker_width_px, 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in index POST: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    logger.info(f"Base path: {BASE_PATH}")
    logger.info(f"Results folder: {RESULTS_FOLDER}")
    uvicorn.run(
        "app_fastapi:app", host="localhost", port=5000, log_level="info", reload=True
    )