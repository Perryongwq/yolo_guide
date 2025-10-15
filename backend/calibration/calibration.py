from flask import Flask, render_template, jsonify, Response
import cv2
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variables for video capture
camera = None
frame = None

# Pre-saved calibration image (replace with the actual path to your image)
CALIBRATION_IMAGE_PATH = "calibration_marks.png"


# Start the video feed
@app.route("/video_feed")
def video_feed():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Initialize the camera
        if not camera.isOpened():
            return Response("Unable to access the camera.", status=500)

    def generate_frames():
        global frame
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Add a green rectangle as the FOV guide
            height, width, _ = frame.shape
            guide_color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame, (100, 50), (width - 100, height - 50), guide_color, thickness)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Capture image and perform calibration
@app.route("/calibrate", methods=["POST"])
def calibrate():
    global frame, camera

    if camera is None or not camera.isOpened():
        return jsonify({"success": False, "message": "Camera is not connected. Please connect the camera."})

    # Capture the frame
    ret, captured_frame = camera.read()
    if not ret:
        return jsonify({"success": False, "message": "Failed to capture image for calibration."})

    # Load the pre-saved calibration image
    if not os.path.exists(CALIBRATION_IMAGE_PATH):
        return jsonify({"success": False, "message": "Calibration image not found."})
    
    calibration_image = cv2.imread(CALIBRATION_IMAGE_PATH, cv2.IMREAD_COLOR)

    # Perform calibration (compare captured frame with calibration image)
    # Example: Calculate microns per pixel based on known square sizes in the calibration image
    calibration_factor = calculate_calibration_factor(captured_frame, calibration_image)

    if calibration_factor is None:
        return jsonify({"success": False, "message": "Calibration failed. Ensure proper focus and alignment."})

    # Save the calibration factor for integration into the main program
    with open("calibration_factor.txt", "w") as f:
        f.write(str(calibration_factor))

    return jsonify({"success": True, "message": "Calibration successful.", "calibration_factor": calibration_factor})


def calculate_calibration_factor(captured_frame, calibration_image):
    """
    Compare the captured frame with the calibration image and calculate the calibration factor.
    """
    # This function can use OpenCV techniques to compare images and extract pixel-to-micron scaling.
    # Example: Detect known square marks in the calibration image and measure their pixel dimensions.

    # Placeholder logic:
    known_micron_width = 500  # Example: Known width of a square in microns
    pixel_width = 100  # Replace with logic to detect and measure square width in pixels
    calibration_factor = known_micron_width / pixel_width  # Microns per pixel

    return calibration_factor


# Disconnect the camera
@app.route("/disconnect", methods=["POST"])
def disconnect():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
    return jsonify({"success": True, "message": "Camera disconnected."})


if __name__ == "__main__":
    app.run(debug=False)
