"""
Test script to process the test_image.png using CT600 Vision Guide
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Constants from app_fastapi.py
MICRONS_PER_PIXEL = 2.3
BLOCK1_OFFSET = 0.0
BLOCK2_OFFSET = 0.0
MEASUREMENT_OFFSET_MICRONS = 5.0
judgment_criteria = {"good": 10, "acceptable": 20}

# Custom display time as requested
DISPLAY_DATETIME = "2026-02-03 20:05:04"
# DISPLAY_DATETIME = "2026-02-03 20:09:46"
# Get base path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Paths
BACKEND_PATH = os.path.join(BASE_PATH, "backend")
MODELS_DIR = os.path.join(BACKEND_PATH, "models")
MODEL_15TYPE_PATH = os.path.join(MODELS_DIR, "15standard_model.pt")
MODEL_18TYPE_PATH = os.path.join(MODELS_DIR, "18standard_model_10_02_26_M.pt")
testimages_path = os.path.join(BASE_PATH, "testimages")
TEST_IMAGE_PATH = os.path.join(testimages_path, "0008.png")
OUTPUT_PATH = os.path.join(BASE_PATH, "test_output6.png")

# Class names for each model type
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


def process_image(image_path: str, item_type: str = "15type") -> dict:
    """
    Process an image using the YOLO model and return results.
    
    Args:
        image_path: Path to the image file
        item_type: Type of model to use ("15type" or "18type")
    
    Returns:
        Dictionary containing processing results
    """
    print(f"\n{'='*60}")
    print(f"CT600 Vision Guide - Test Image Processing")
    print(f"{'='*60}")
    print(f"Image path: {image_path}")
    print(f"Model type: {item_type}")
    print(f"Display Time: {DISPLAY_DATETIME}")
    print(f"{'='*60}\n")
    
    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return {"error": True, "error_message": "Test image not found"}
    
    # Select model path and class names based on item_type
    if item_type == "18type":
        model_path = MODEL_18TYPE_PATH
        selected_class_names = class_names_18type
    else:
        model_path = MODEL_15TYPE_PATH
        selected_class_names = class_names_15type
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return {"error": True, "error_message": f"Model not found at {model_path}"}
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to load image")
        return {"error": True, "error_message": "Failed to load image"}
    
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} (WxH)")
    
    # Get image dimensions for line extension
    image_height, image_width = image.shape[:2]
    
    # YOLO prediction
    print("\nRunning YOLO prediction...")
    results = model.predict(source=image, conf=0.25, save=False)
    
    block1_edge_y = block2_edge_y = None
    block1_box_y = block2_box_y = None
    calibration_marker_width_px = None
    microns_per_pixel = MICRONS_PER_PIXEL
    
    print("\nDetected objects:")
    print("-" * 40)
    
    for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
        x_center, y_center, width, height = box
        label = selected_class_names[int(cls.item())]
        print(f"  - {label}: center=({x_center:.1f}, {y_center:.1f}), size=({width:.1f}x{height:.1f})")
        
        # Handle all item types: 15type and 18type
        # Check for block1_edge variants: "block1_edge", "block1_edge15"
        if label in ["block1_edge", "block1_edge15"]:
            edge_y = int(y_center + height / 2)
            block1_edge_y = edge_y + (BLOCK1_OFFSET / microns_per_pixel)
            # Extend edge1 line more to the right
            cv2.line(
                image,
                (int(x_center - 150), edge_y),
                (min(int(x_center + 400), image_width - 1), edge_y),
                (255, 0, 0),
                2,
            )
            print(f"    -> block1_edge_y set to {block1_edge_y:.2f}")
        
        # Check for block2_edge variants: "block2_edge", "block2_edge15"
        elif label in ["block2_edge", "block2_edge15"]:
            edge_y = int(y_center + height / 2)
            block2_edge_y = edge_y + (BLOCK2_OFFSET / microns_per_pixel)
            # Extend edge2 line more to the left
            cv2.line(
                image,
                (max(int(x_center - 400), 0), edge_y),
                (int(x_center + 150), edge_y),
                (0, 255, 255),
                2,
            )
            print(f"    -> block2_edge_y set to {block2_edge_y:.2f}")
        
        # Check for block1 body variants: "block1", "block1_15" (but not "block1_edge")
        elif label in ["block1", "block1_15"]:
            block1_box_y = int(y_center + height / 2)
        
        # Check for block2 body variants: "block2", "block2_15" (but not "block2_edge")
        elif label in ["block2", "block2_15"]:
            block2_box_y = int(y_center + height / 2)
        
        elif label == "cal_mark":
            calibration_marker_width_px = width.item()
            print(f"    -> calibration marker width: {calibration_marker_width_px:.2f}px")
    
    print("-" * 40)
    
    # Calibration check
    if calibration_marker_width_px:
        microns_per_pixel = 1000.0 / calibration_marker_width_px
        print(f"\n[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px")
        print(f"[Calibration] microns/pixel = {microns_per_pixel:.2f}")
        
        if microns_per_pixel > 10:
            print("Warning: Microns per pixel too high, suggesting focus adjustment")
            return {
                "error": True,
                "error_message": "Fail to capture work guide and insertion guide",
                "reason": "Microns per pixel too high, suggesting focus adjustment",
            }
    else:
        print("\nWarning: cal_mark not detected")
        # Continue anyway for testing purposes with default value
        microns_per_pixel = MICRONS_PER_PIXEL
        print(f"Continuing with default microns_per_pixel = {microns_per_pixel:.2f}")
    
    # Check if both edge positions are available
    if block1_edge_y is None or block2_edge_y is None:
        print("\nWarning: One or more edges not detected")
        if block1_edge_y is None:
            edge_label = "block1_edge" if item_type == "18type" else "block1_edge15"
            print(f"  - {edge_label} not detected")
        if block2_edge_y is None:
            edge_label = "block2_edge" if item_type == "18type" else "block2_edge15"
            print(f"  - {edge_label} not detected")
        
        # Still save the processed image with whatever detections were made
        cv2.putText(
            image,
            f"Checked on: {DISPLAY_DATETIME}",
            (10, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            image,
            f"Microns/px:{microns_per_pixel:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),  # Yellow in BGR
            2,
        )
        cv2.putText(
            image,
            "Detection incomplete - edges not found",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        
        cv2.imwrite(OUTPUT_PATH, image)
        print(f"\nPartial processed image saved to: {OUTPUT_PATH}")
        
        return {
            "error": True,
            "error_message": "Fail to capture work guide and insertion guide",
            "reason": "One or more edges not detected",
        }
    
    # Calculate measurements
    y_diff_pixels = block1_edge_y - block2_edge_y
    y_diff_microns = (y_diff_pixels * microns_per_pixel) + MEASUREMENT_OFFSET_MICRONS
    
    print(f"\n[Measurement Results]")
    print(f"  block1_edge_y: {block1_edge_y:.2f}")
    print(f"  block2_edge_y: {block2_edge_y:.2f}")
    print(f"  Y-difference (pixels): {y_diff_pixels:.2f}")
    print(f"  Y-difference (microns): {y_diff_microns:.2f}")
    
    # Judgment logic
    if y_diff_microns < judgment_criteria["good"]:
        judgment = "Good"
        judgment_color = (0, 255, 0)
    elif y_diff_microns < judgment_criteria["acceptable"]:
        judgment = "Acceptable"
        judgment_color = (0, 165, 255)
    else:
        judgment = "No Good"
        judgment_color = (0, 0, 255)
    
    print(f"  Judgment: {judgment}")
    
    # Add annotations
    text_x = image.shape[1] // 2 + 250
    # Position text above the blue line (block1_edge)
    text_y = int(block1_edge_y - 80)  # 80 pixels above block1_edge
    
    cv2.putText(
        image,
        f"{y_diff_microns:.2f} microns",
        (text_x - 100, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"Judgment: {judgment}",
        (text_x - 100, text_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        judgment_color,
        2,
    )
    cv2.putText(
        image,
        f"Checked on: {DISPLAY_DATETIME}",
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        image,
        f"Microns/px:{microns_per_pixel:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),  # Yellow in BGR
        2,
    )
    
    # Save processed image
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"\nProcessed image saved to: {OUTPUT_PATH}")
    
    return {
        "success": True,
        "y_diff_microns": round(y_diff_microns, 2),
        "judgment": judgment,
        "processed_timestamp": DISPLAY_DATETIME,
        "output_path": OUTPUT_PATH,
    }


def main():
    """Main entry point for test script."""
    print("\n" + "="*60)
    print("CT600 Vision Guide - Image Processing Test")
    print("="*60)
    
    # Test with 18type model by default
    item_type = "18type"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["15type", "18type"]:
            item_type = sys.argv[1]
        else:
            print(f"Unknown item type: {sys.argv[1]}")
            print("Usage: python test_process_image.py [15type|18type]")
            sys.exit(1)
    
    result = process_image(TEST_IMAGE_PATH, item_type)
    
    print("\n" + "="*60)
    print("Processing Result:")
    print("="*60)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    main()

