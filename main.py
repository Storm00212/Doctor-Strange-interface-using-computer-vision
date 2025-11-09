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
    # Extract frame dimensions (height, width, channels) for coordinate conversion
    h, w, _ = frame.shape

    # Convert from OpenCV's BGR color space to RGB for MediaPipe compatibility
    # MediaPipe's hand detection model requires RGB input format
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Run MediaPipe hand detection on the RGB frame
    # This returns landmark coordinates for any detected hands
    results = hands.process(rgb_frame)

    # Check if any hands were detected in the frame
    if results.multi_hand_landmarks:
        # Process each detected hand (supports multiple hands)
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert MediaPipe's normalized coordinates (0.0-1.0) to pixel coordinates
            # MediaPipe returns relative positions, so we scale by frame dimensions
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Extract the 8 key landmark positions we need for gesture analysis
            # This includes wrist, fingertips, and palm joint positions
            (wrist, thumb_tip, index_mcp, index_tip,
             middle_mcp, middle_tip, ring_tip, pinky_tip) = position_data(lm_list)

            # Calculate key distances for determining hand gesture state:
            # 1. Distance from wrist to index MCP (measures hand size/proximity)
            index_wrist_distance = calculate_distance(wrist, index_mcp)

            # 2. Distance from index tip to pinky tip (measures finger spread)
            index_pinky_distance = calculate_distance(index_tip, pinky_tip)

            # Calculate the gesture ratio: finger spread relative to hand size
            # Higher ratio indicates more open/spread fingers
            # This ratio determines which visual effect to apply
            ratio = index_pinky_distance / index_wrist_distance

            # Determine gesture state based on the calculated ratio and apply effects

            # Gesture State 1: Moderately Open Hand (0.5 < ratio < 1.3)
            # This represents a hand that's partially open, like making a "stop" gesture
            # Effect: Draw glowing lines connecting the wrist to fingertips and between fingertips
            if 0.5 < ratio < 1.3:
                # Create a list of all fingertip positions for easy iteration
                fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]

                # Draw radial lines from wrist center to each fingertip
                # This creates a "starburst" or "web" pattern emanating from the palm
                for finger in fingers:
                    frame = draw_line(frame, wrist, finger,
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])

                # Draw connecting lines between consecutive fingertips
                # This creates a "finger web" effect across the open fingers
                for i in range(len(fingers) - 1):
                    frame = draw_line(frame, fingers[i], fingers[i + 1],
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])

            # Gesture State 2: Fully Open Hand (ratio >= 1.3)
            # This represents a hand that's fully open and spread wide
            # Effect: Display animated rotating magical circles (the signature Doctor Strange effect)
            elif ratio >= 1.3:
                # Use the middle finger MCP as the center point for the circles
                # This provides a stable center that's in the middle of the palm
                center_x, center_y = middle_mcp

                # Calculate the diameter of the magic circles based on hand size
                # Larger hands get larger circles, scaled by the configured multiplier
                diameter = round(index_wrist_distance * config["overlay"]["shield_size_multiplier"])

                # Calculate the top-left corner position for overlay placement
                # Clamp values to ensure the overlay stays within frame boundaries
                x1 = limit_value(center_x - diameter // 2, 0, w)
                y1 = limit_value(center_y - diameter // 2, 0, h)

                # Final boundary check: ensure diameter fits within the frame
                # This prevents the overlay from extending beyond frame edges
                diameter = min(diameter, w - x1, h - y1)

                # Update the rotation angle for the animation effect
                # Accumulate rotation and wrap around at 360 degrees
                deg = (deg + config["overlay"]["rotation_degree_increment"]) % 360

                # Create affine transformation matrices for rotating the circles
                # Outer circle rotates clockwise, inner circle rotates counter-clockwise
                # Rotation center is at the image center (width//2, height//2)
                M1 = cv.getRotationMatrix2D((outer_circle.shape[1] // 2, outer_circle.shape[0] // 2), deg, 1.0)
                M2 = cv.getRotationMatrix2D((inner_circle.shape[1] // 2, inner_circle.shape[0] // 2), -deg, 1.0)

                # Apply the rotation transformations to create animated circle images
                rotated_outer = cv.warpAffine(outer_circle, M1, (outer_circle.shape[1], outer_circle.shape[0]))
                rotated_inner = cv.warpAffine(inner_circle, M2, (inner_circle.shape[1], inner_circle.shape[0]))

                # Overlay both rotated circles onto the frame at the calculated position
                # The circles are scaled to the calculated diameter and blended with transparency
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
        # Enter the main real-time processing loop
        # This loop runs continuously until the user quits or an error occurs
        while cap.isOpened():
            # Step 1: Capture a single frame from the webcam
            # success is True if frame was captured successfully
            success, frame = cap.read()

            # If frame capture failed, break out of the loop
            # This could happen if camera is disconnected or busy
            if not success:
                print("Failed to capture frame.")
                break

            # Step 2: Flip the frame horizontally to create a mirror effect
            # This makes the video feel more natural, like looking in a mirror
            # cv.flip(frame, 1) flips along the y-axis (horizontal flip)
            frame = cv.flip(frame, 1)

            # Step 3: Process the frame through our computer vision pipeline
            # This applies hand detection and visual effects based on gesture
            # Returns the processed frame and updated rotation angle
            frame, deg = process_frame(frame, hands, config, inner_circle, outer_circle, deg)

            # Step 4: Display the processed frame in a window
            # "Image" is the window title, frame contains the visual effects
            cv.imshow("Image", frame)

            # Step 5: Check for user input to quit the application
            # cv.waitKey(1) waits 1ms for a key press (non-blocking)
            # Compare against the configured quit key (usually 'q' or ESC)
            if cv.waitKey(1) == ord(config["keybindings"]["quit_key"]):
                break

    finally:
        # Cleanup section: Always executed whether loop exits normally or due to error
        # Release the video capture device to free camera resources
        cap.release()

        # Close all OpenCV windows to clean up the display
        cv.destroyAllWindows()

# Entry point for the application
# This ensures the script runs only when executed directly, not when imported as a module
if __name__ == "__main__":
    main()
