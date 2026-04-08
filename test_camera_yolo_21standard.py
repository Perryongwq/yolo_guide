"""
Test script to capture images using USB camera and process with YOLO model 21standard
to determine microns_per_pixel.

Based on backend/app_fastapi.py
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time

# Constants from app_fastapi.py
MICRONS_PER_PIXEL = 2.3  # Default value
BLOCK1_OFFSET = 0.0
BLOCK2_OFFSET = 0.0
MEASUREMENT_OFFSET_MICRONS = 5.0
judgement_criteria = {"good": 10, "acceptable": 20}

# Get base path
def get_base_path():
    """Get base path for file storage - works for both script and exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

BASE_PATH = get_base_path()
BACKEND_PATH = os.path.join(BASE_PATH, "backend")
MODELS_DIR = os.path.join(BACKEND_PATH, "models")

# Model path for 21standard
MODEL_21STANDARD_PATH = os.path.join(MODELS_DIR, "21standard_model.pt")

# Output directory
OUTPUT_DIR = os.path.join(BASE_PATH, "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names for 21type (same as 18type)
class_names_21type = [
    "block1",
    "block1_edge",
    "block2",
    "block2_edge",
    "cal_mark",
]


def init_camera_optimized():
    """Initialize camera with DirectShow backend for faster performance on Windows"""
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for Windows
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    cam.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
    return cam


def capture_image(camera, output_path):
    """
    Capture a single image from the camera.
    
    Args:
        camera: cv2.VideoCapture object
        output_path: Path to save the captured image
    
    Returns:
        tuple: (success: bool, image: np.ndarray or None)
    """
    if camera is None or not camera.isOpened():
        print("Error: Camera is not opened")
        return False, None
    
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame from camera")
        return False, None
    
    # Save the captured image
    success = cv2.imwrite(output_path, frame)
    if not success:
        print(f"Error: Failed to save image to {output_path}")
        return False, None
    
    print(f"Image captured and saved to: {output_path}")
    print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]} (WxH)")
    return True, frame


def process_image_with_yolo(image, model, class_names):
    """
    Process image with YOLO model to determine microns_per_pixel.
    
    Args:
        image: Input image as numpy array
        model: YOLO model instance
        class_names: List of class names
    
    Returns:
        dict: Processing results including microns_per_pixel
    """
    print("\n" + "="*60)
    print("Processing image with YOLO model 21standard")
    print("="*60)
    
    # YOLO prediction
    print("\nRunning YOLO prediction...")
    results = model.predict(source=image, conf=0.25, save=False)
    
    calibration_marker_width_px = None
    microns_per_pixel = MICRONS_PER_PIXEL
    
    # First pass: Find calibration marker to calculate microns_per_pixel
    print("\nSearching for calibration marker (cal_mark)...")
    for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
        x_center, y_center, width, height = box
        label = class_names[int(cls.item())]
        
        print(f"  Detected: {label} at ({x_center:.1f}, {y_center:.1f}), size: {width:.1f}x{height:.1f}")
        
        if label == "cal_mark":
            # Convert to scalar if it's a numpy array/tensor
            if hasattr(width, 'item'):
                calibration_marker_width_px = width.item()
            else:
                calibration_marker_width_px = float(width)
            print(f"  -> Found calibration marker! Width: {calibration_marker_width_px:.2f}px")
            break
    
    # Calculate microns per pixel from calibration marker if detected
    if calibration_marker_width_px:
        microns_per_pixel = 1000.0 / calibration_marker_width_px
        print(f"\n[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px")
        print(f"[Calibration] microns/px = {microns_per_pixel:.2f}")
        
        if microns_per_pixel > 10:
            print("Warning: Microns per pixel too high, suggesting focus adjustment")
            return {
                "success": False,
                "error": True,
                "error_message": "Microns per pixel too high, suggesting focus adjustment",
                "microns_per_pixel": round(microns_per_pixel, 2),
                "calibration_marker_width_px": round(calibration_marker_width_px, 2),
            }
    else:
        print("\nWarning: cal_mark not detected, using default microns_per_pixel")
        print(f"Default microns_per_pixel = {microns_per_pixel:.2f}")
        return {
            "success": False,
            "error": True,
            "error_message": "cal_mark not detected",
            "microns_per_pixel": round(microns_per_pixel, 2),
            "calibration_marker_width_px": None,
        }
    
    # Second pass: Process other detections for visualization
    print("\nProcessing other detections...")
    block1_edge_y = block2_edge_y = None
    block1_box_y = block2_box_y = None
    
    for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
        x_center, y_center, width, height = box
        label = class_names[int(cls.item())]
        
        # Check for block1_edge
        if label == "block1_edge":
            edge_y = int(y_center + height / 2)
            block1_edge_y = edge_y + (BLOCK1_OFFSET / microns_per_pixel)
            print(f"  -> block1_edge detected at y={block1_edge_y:.2f}")
            # Draw line
            cv2.line(
                image,
                (int(x_center - 300), edge_y),
                (int(x_center + 300), edge_y),
                (255, 0, 0),  # Blue in BGR
                2,
            )
        
        # Check for block2_edge
        elif label == "block2_edge":
            edge_y = int(y_center + height / 2)
            block2_edge_y = edge_y + (BLOCK2_OFFSET / microns_per_pixel)
            print(f"  -> block2_edge detected at y={block2_edge_y:.2f}")
            # Draw line
            cv2.line(
                image,
                (int(x_center - 300), edge_y),
                (int(x_center + 300), edge_y),
                (0, 255, 255),  # Cyan in BGR
                2,
            )
        
        # Check for block1 body
        elif label == "block1":
            block1_box_y = int(y_center + height / 2)
            print(f"  -> block1 detected at y={block1_box_y:.2f}")
        
        # Check for block2 body
        elif label == "block2":
            block2_box_y = int(y_center + height / 2)
            print(f"  -> block2 detected at y={block2_box_y:.2f}")
    
    # Calculate measurements if edges are detected
    y_diff_microns = None
    judgement = None
    
    if block1_edge_y is not None and block2_edge_y is not None:
        y_diff_pixels = block1_edge_y - block2_edge_y
        y_diff_microns = (y_diff_pixels * microns_per_pixel) + MEASUREMENT_OFFSET_MICRONS
        
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
        
        print(f"\n[Measurement] Y-difference: {y_diff_microns:.2f} microns")
        print(f"[Measurement] Judgement: {judgement}")
        
        # Add annotations to image
        text_x = image.shape[1] // 2 + 250
        text_y = int((block1_edge_y + block2_edge_y) / 2)
        
        cv2.putText(
            image,
            f"{y_diff_microns:.2f} microns",
            (text_x - 100, text_y - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Judgment: {judgement}",
            (text_x - 100, text_y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            judgement_color,
            2,
        )
    
    # Add calibration info to image
    cal_text = f"{microns_per_pixel:.2f} um/pixel"
    cv2.putText(
        image,
        cal_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),  # Yellow in BGR
        2,
    )
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        image,
        f"Checked on: {current_datetime}",
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
    )
    
    return {
        "success": True,
        "error": False,
        "microns_per_pixel": round(microns_per_pixel, 2),
        "calibration_marker_width_px": round(calibration_marker_width_px, 2),
        "y_diff_microns": round(y_diff_microns, 2) if y_diff_microns is not None else None,
        "judgement": judgement,
        "block1_edge_y": round(block1_edge_y, 2) if block1_edge_y is not None else None,
        "block2_edge_y": round(block2_edge_y, 2) if block2_edge_y is not None else None,
        "processed_image": image,
    }


def main():
    """Main entry point for test script."""
    print("\n" + "="*60)
    print("CT600 Vision Guide - Camera Test Script")
    print("Model: 21standard")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_21STANDARD_PATH):
        print(f"\nError: Model file not found at {MODEL_21STANDARD_PATH}")
        print("Please ensure the 21standard_model.pt file exists in the models directory.")
        return
    
    # Load YOLO model
    print(f"\nLoading YOLO model from: {MODEL_21STANDARD_PATH}")
    try:
        model = YOLO(MODEL_21STANDARD_PATH)
        print("Model loaded successfully!")
        
        # Log model class names if available
        if hasattr(model, 'names') and model.names:
            model_class_names = [model.names[i] for i in sorted(model.names.keys())]
            print(f"Model class names: {model_class_names}")
            print(f"Model class count: {len(model_class_names)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    print("\nInitializing USB camera...")
    camera = init_camera_optimized()
    
    if not camera.isOpened():
        print("Error: Unable to access camera")
        print("Please ensure a USB camera is connected and not being used by another application.")
        return
    
    print("Camera initialized successfully!")
    print("Camera settings:")
    print(f"  Width: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS: {camera.get(cv2.CAP_PROP_FPS)}")
    
    try:
        # Give camera a moment to stabilize
        print("\nWaiting for camera to stabilize...")
        time.sleep(1)
        
        # Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured_image_path = os.path.join(OUTPUT_DIR, f"captured_{timestamp}.png")
        
        print("\nCapturing image...")
        success, image = capture_image(camera, captured_image_path)
        
        if not success or image is None:
            print("Failed to capture image")
            return
        
        # Process image with YOLO
        results = process_image_with_yolo(image, model, class_names_21type)
        
        # Save processed image
        if results.get("success") and results.get("processed_image") is not None:
            processed_image_path = os.path.join(OUTPUT_DIR, f"processed_{timestamp}.png")
            cv2.imwrite(processed_image_path, results["processed_image"])
            print(f"\nProcessed image saved to: {processed_image_path}")
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Success: {results.get('success', False)}")
        print(f"Microns per Pixel: {results.get('microns_per_pixel', 'N/A')}")
        print(f"Calibration Marker Width (px): {results.get('calibration_marker_width_px', 'N/A')}")
        
        if results.get("error"):
            print(f"Error: {results.get('error_message', 'Unknown error')}")
        else:
            print(f"Y-Difference (microns): {results.get('y_diff_microns', 'N/A')}")
            print(f"Judgement: {results.get('judgement', 'N/A')}")
            print(f"Block1 Edge Y: {results.get('block1_edge_y', 'N/A')}")
            print(f"Block2 Edge Y: {results.get('block2_edge_y', 'N/A')}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release camera
        if camera is not None and camera.isOpened():
            camera.release()
            print("\nCamera released")


if __name__ == "__main__":
    main()
