"""
main.py

This is the main script for the Doctor Strange Filter project. It uses computer vision
to detect hand gestures from a webcam feed and overlays magical circle effects similar
to those seen in the Marvel character Doctor Strange's spells.

The program works by:
1. Capturing video from the webcam
2. Using MediaPipe to detect hand landmarks in real-time
3. Analyzing hand pose to determine gesture state
4. Drawing connecting lines between fingers for partial gestures
5. Overlaying rotating magical circles when hand is fully open

Key gesture states:
- Closed fist: No effects
- Partially open (ratio 0.5-1.3): Draws connecting lines between fingers
- Fully open (ratio >= 1.3): Displays rotating magic circles

Dependencies:
- cv2 (OpenCV): For video capture, image processing, and display
- mediapipe: For hand landmark detection
- json: For loading configuration settings
- functions: Custom utility functions for position extraction, distance calculation,
             line drawing, and image overlay

Configuration is loaded from config.json, which contains camera settings,
overlay parameters, and visual effect configurations.
"""

import cv2 as cv
import mediapipe as mp
import json
from functions import position_data, calculate_distance, draw_line, overlay_image

def load_config(path: str = "config.json") -> dict:
    """
    Loads configuration settings from a JSON file.

    The config.json file contains all customizable parameters for the application,
    including camera settings (device ID, resolution), overlay image paths,
    visual effect parameters (colors, sizes, rotation speeds), and key bindings.

    Args:
        path (str): Path to the configuration JSON file. Defaults to "config.json".

    Returns:
        dict: Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the JSON is malformed.
    """
    with open(path, "r") as file:
        return json.load(file)

def limit_value(val: int, min_val: int, max_val: int) -> int:
    """
    Clamps a value within specified minimum and maximum bounds.

    This utility function ensures that calculated positions and sizes don't
    exceed frame boundaries or go below zero, preventing errors when
    overlaying images or drawing elements.

    Args:
        val (int): The value to clamp.
        min_val (int): The minimum allowed value.
        max_val (int): The maximum allowed value.

    Returns:
        int: The clamped value, guaranteed to be between min_val and max_val.
    """
    return max(min(val, max_val), min_val)

def initialize_camera(config: dict) -> cv.VideoCapture:
    """
    Initializes the webcam with specified configuration parameters.

    This function sets up the video capture device using OpenCV's VideoCapture.
    It configures the camera resolution and device ID as specified in the config.
    Proper camera initialization is crucial for consistent performance and
    accurate hand detection.

    Args:
        config (dict): Configuration dictionary containing camera settings.
                      Expected keys: camera.device_id, camera.width, camera.height

    Returns:
        cv.VideoCapture: Configured video capture object ready for frame reading.

    Raises:
        RuntimeError: If the webcam cannot be opened (e.g., device not found,
                     permissions issue, or already in use).
    """
    # Create VideoCapture object with specified device ID (usually 0 for default camera)
    cap = cv.VideoCapture(config["camera"]["device_id"])

    # Set camera resolution - important for performance and detection accuracy
    cap.set(cv.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    # Verify camera opened successfully
    if not cap.isOpened():
        raise RuntimeError("Failed to open the webcam.")
    return cap

def load_images(config: dict) -> tuple:
    """
    Loads the magical circle overlay images from disk.

    The Doctor Strange effect uses two concentric rotating circles:
    - Inner circle: The smaller, faster-rotating circle
    - Outer circle: The larger, slower-rotating circle

    Images must be in PNG format with transparency (alpha channel) to allow
    proper blending with the video feed. They are loaded using OpenCV's
    IMREAD_UNCHANGED flag to preserve the alpha channel.

    Args:
        config (dict): Configuration dictionary containing overlay image paths.
                      Expected keys: overlay.inner_circle_path, overlay.outer_circle_path

    Returns:
        tuple: A tuple containing (inner_circle, outer_circle) as numpy arrays
               with shape (height, width, 4) for RGBA format.

    Raises:
        FileNotFoundError: If either image file cannot be loaded (file missing,
                          corrupted, or invalid format).
    """
    # Load images with alpha channel preserved (-1 flag = IMREAD_UNCHANGED)
    inner_circle = cv.imread(config["overlay"]["inner_circle_path"], -1)
    outer_circle = cv.imread(config["overlay"]["outer_circle_path"], -1)

    # Verify both images loaded successfully
    if inner_circle is None or outer_circle is None:
        raise FileNotFoundError("Failed to load one or more overlay images.")
    return inner_circle, outer_circle
def process_frame(frame, hands, config, inner_circle, outer_circle, deg):
    """
    Processes a single video frame, detects hands, and applies visual effects.

    This is the core processing function that handles the computer vision pipeline:
    1. Converts frame to RGB for MediaPipe processing
    2. Detects hand landmarks using MediaPipe Hands
    3. Analyzes hand pose to determine gesture state
    4. Applies appropriate visual effects based on hand openness

    The function implements three gesture states based on a ratio calculation:
    - Closed/partially closed hand: No effects
    - Moderately open (ratio 0.5-1.3): Draws connecting lines between fingers
    - Fully open (ratio >= 1.3): Displays rotating magical circles

    Args:
        frame (np.ndarray): Current video frame in BGR format.
        hands (mp.solutions.hands.Hands): MediaPipe Hands object for detection.
        config (dict): Configuration dictionary with visual effect parameters.
        inner_circle (np.ndarray): Inner magical circle image (RGBA).
        outer_circle (np.ndarray): Outer magical circle image (RGBA).
        deg (int): Current rotation angle for the circles (accumulates over frames).

    Returns:
        tuple: (processed_frame, updated_deg) where processed_frame has effects applied
               and updated_deg is the new rotation angle for the next frame.
    """
    # Get frame dimensions for coordinate scaling
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe (MediaPipe expects RGB input)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Process each detected hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert normalized coordinates (0-1) to pixel coordinates
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Extract key hand positions using our utility function
            (wrist, thumb_tip, index_mcp, index_tip,
             middle_mcp, middle_tip, ring_tip, pinky_tip) = position_data(lm_list)

            # Calculate distances for gesture analysis
            index_wrist_distance = calculate_distance(wrist, index_mcp)
            index_pinky_distance = calculate_distance(index_tip, pinky_tip)

            # Calculate hand openness ratio (higher ratio = more open hand)
            ratio = index_pinky_distance / index_wrist_distance

            # Gesture state 1: Moderately open hand - draw connecting lines
            if 0.5 < ratio < 1.3:
                fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                # Draw lines from wrist to each fingertip
                for finger in fingers:
                    frame = draw_line(frame, wrist, finger,
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])
                # Draw lines between consecutive fingertips
                for i in range(len(fingers) - 1):
                    frame = draw_line(frame, fingers[i], fingers[i + 1],
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])

            # Gesture state 2: Fully open hand - display rotating magic circles
            elif ratio >= 1.3:
                # Position circles at the middle finger MCP (center of palm)
                center_x, center_y = middle_mcp

                # Calculate circle diameter based on hand size
                diameter = round(index_wrist_distance * config["overlay"]["shield_size_multiplier"])

                # Calculate top-left corner for overlay, ensuring it stays within frame bounds
                x1 = limit_value(center_x - diameter // 2, 0, w)
                y1 = limit_value(center_y - diameter // 2, 0, h)

                # Ensure diameter doesn't exceed frame boundaries
                diameter = min(diameter, w - x1, h - y1)

                # Update rotation angle for animation effect
                deg = (deg + config["overlay"]["rotation_degree_increment"]) % 360

                # Create rotation matrices for outer and inner circles (opposite directions)
                M1 = cv.getRotationMatrix2D((outer_circle.shape[1] // 2, outer_circle.shape[0] // 2), deg, 1.0)
                M2 = cv.getRotationMatrix2D((inner_circle.shape[1] // 2, inner_circle.shape[0] // 2), -deg, 1.0)

                # Apply rotations to the circle images
                rotated_outer = cv.warpAffine(outer_circle, M1, (outer_circle.shape[1], outer_circle.shape[0]))
                rotated_inner = cv.warpAffine(inner_circle, M2, (inner_circle.shape[1], inner_circle.shape[0]))

                # Overlay the rotated circles onto the frame
                frame = overlay_image(rotated_outer, frame, x1, y1, (diameter, diameter))
                frame = overlay_image(rotated_inner, frame, x1, y1, (diameter, diameter))

    return frame, deg

def main():
    """
    Main application loop that runs the Doctor Strange Filter.

    This function orchestrates the entire application flow:
    1. Loads configuration and initializes all components
    2. Sets up video capture, image loading, and hand detection
    3. Runs the main processing loop until user quits
    4. Ensures proper cleanup of resources

    The loop processes frames in real-time, applying hand detection and
    visual effects. It maintains a rotation angle accumulator for the
    magical circles animation.

    Key components initialized:
    - Configuration loading from JSON
    - Camera setup with specified resolution
    - Loading of overlay images (inner/outer circles)
    - MediaPipe Hands model for gesture detection
    - Rotation angle tracking for circle animation

    The application can be terminated by pressing the quit key (configured in config.json).
    """
    # Initialize all components
    config = load_config()
    cap = initialize_camera(config)
    inner_circle, outer_circle = load_images(config)

    # Initialize MediaPipe Hands for gesture detection
    hands = mp.solutions.hands.Hands()

    # Initialize rotation angle for circle animation
    deg = 0

    try:
        # Main processing loop
        while cap.isOpened():
            # Capture frame from webcam
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            # Flip frame horizontally for mirror effect (natural for webcam)
            frame = cv.flip(frame, 1)

            # Process the frame and apply visual effects
            frame, deg = process_frame(frame, hands, config, inner_circle, outer_circle, deg)

            # Display the processed frame
            cv.imshow("Image", frame)

            # Check for quit key press (non-blocking wait)
            if cv.waitKey(1) == ord(config["keybindings"]["quit_key"]):
                break
    finally:
        # Ensure proper cleanup of resources
        cap.release()
        cv.destroyAllWindows()

# Entry point for the application
# This ensures the script runs only when executed directly, not when imported as a module
if __name__ == "__main__":
    main()
