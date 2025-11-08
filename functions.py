"""
functions.py

This module contains utility functions for the Doctor Strange Filter project.
It includes functions for extracting hand landmark positions, calculating distances,
drawing lines on frames, and overlaying images with transparency.

Dependencies:
- cv2 (OpenCV): For image processing and drawing operations.
- numpy: For numerical computations, especially distance calculations.
- typing: For type hints to improve code readability and maintainability.

Constants:
- LINE_COLOR: Default color for drawing lines (orange-like hue).
- WHITE_COLOR: White color used for inner line in draw_line function.
"""

import cv2 as cv
import numpy as np
from typing import List, Tuple

# Define color constants for drawing operations
LINE_COLOR = (0, 140, 255)  # Orange-like color for outer lines
WHITE_COLOR = (255, 255, 255)  # White color for inner lines

def position_data(lmlist: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Extracts and returns key fingertip and hand positions from the landmark list.

    This function takes the list of 21 hand landmarks detected by MediaPipe and extracts
    the coordinates of specific points that are crucial for gesture recognition and
    overlay positioning. The selected landmarks include:
    - Wrist (index 0)
    - Thumb tip (index 4)
    - Index finger MCP (metacarpophalangeal joint, index 5)
    - Index finger tip (index 8)
    - Middle finger MCP (index 9)
    - Middle finger tip (index 12)
    - Ring finger tip (index 16)
    - Pinky finger tip (index 20)

    These points are used to calculate distances, determine hand openness, and position
    the magic circle overlays.

    Args:
        lmlist (List[Tuple[int, int]]): List of 21 tuples containing (x, y) coordinates
                                        of hand keypoints, scaled to frame dimensions.

    Returns:
        List[Tuple[int, int]]: List of 8 tuples containing coordinates of key hand positions
                               in the order: [wrist, thumb_tip, index_mcp, index_tip,
                               middle_mcp, middle_tip, ring_tip, pinky_tip].

    Raises:
        ValueError: If the landmark list contains fewer than 21 points.
    """
    if len(lmlist) < 21:
        raise ValueError("Landmark list must contain at least 21 points.")

    # Indices correspond to MediaPipe hand landmark model:
    # 0: wrist, 4: thumb tip, 5: index MCP, 8: index tip,
    # 9: middle MCP, 12: middle tip, 16: ring tip, 20: pinky tip
    keys = [0, 4, 5, 8, 9, 12, 16, 20]
    return [lmlist[i] for i in keys]

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculates the Euclidean distance between two points using the Euclidean formula.

    The Euclidean distance is calculated as sqrt((x2 - x1)^2 + (y2 - y1)^2).
    This function uses NumPy's linalg.norm for efficient computation, which is
    optimized for performance compared to manual calculation.

    This distance metric is used in the main script to:
    - Measure the openness of the hand (distance between index tip and pinky tip)
    - Determine the size of the magic circle overlay (based on index-wrist distance)
    - Assess hand gesture states for triggering different visual effects

    Args:
        p1 (Tuple[int, int]): First point coordinates (x1, y1).
        p2 (Tuple[int, int]): Second point coordinates (x2, y2).

    Returns:
        float: Euclidean distance between the two points in pixels.
    """
    # Use numpy's vectorized norm calculation for better performance
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_line(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int] = LINE_COLOR,
    thickness: int = 5
) -> np.ndarray:
    """
    Draws a stylized line between two points on the frame with an inner white line.

    This function creates a visual effect similar to energy beams or magical connections
    by drawing two concentric lines: an outer colored line and an inner white line.
    This creates a glowing effect that enhances the mystical appearance of the filter.

    The line drawing is used in the main script when the hand is in a partially open
    state (ratio between 0.5 and 1.3), creating connecting lines from the wrist to
    each fingertip and between consecutive fingertips, forming a "web" pattern.

    Args:
        frame (np.ndarray): The video frame/image to draw on (BGR format).
        p1 (Tuple[int, int]): Starting point coordinates (x1, y1).
        p2 (Tuple[int, int]): Ending point coordinates (x2, y2).
        color (Tuple[int, int, int]): BGR color tuple for the outer line.
                                     Defaults to LINE_COLOR (orange-like).
        thickness (int): Thickness of the outer line in pixels. Defaults to 5.

    Returns:
        np.ndarray: The modified frame with the drawn line effect.
    """
    # Draw the outer colored line
    cv.line(frame, p1, p2, color, thickness)
    # Draw the inner white line for glowing effect (half thickness, minimum 1)
    cv.line(frame, p1, p2, WHITE_COLOR, max(1, thickness // 2))
    return frame

def overlay_image(
    target_img: np.ndarray,
    frame: np.ndarray,
    x: int, y: int,
    size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Overlays a transparent image onto a frame using alpha blending.

    This function performs alpha blending to composite a semi-transparent or fully
    transparent image (with alpha channel) onto the video frame. It's crucial for
    creating the magical effect where the Doctor Strange circles appear to float
    over the hand without completely obscuring the background.

    The process involves:
    1. Resizing the target image if a size is specified
    2. Separating the RGBA channels (Red, Green, Blue, Alpha)
    3. Creating a mask from the alpha channel to determine transparency
    4. Using bitwise operations to blend the foreground and background

    This technique allows for realistic overlay effects where the magic circles
    can have transparent backgrounds and varying opacity levels.

    Args:
        target_img (np.ndarray): Source image to overlay, must have 4 channels (RGBA).
                                Loaded with cv.IMREAD_UNCHANGED flag.
        frame (np.ndarray): Destination frame/image where overlay will be applied (BGR).
        x (int): X-coordinate of top-left corner for overlay placement.
        y (int): Y-coordinate of top-left corner for overlay placement.
        size (Tuple[int, int], optional): Desired size (width, height) to resize
                                         target_img. If None, uses original size.

    Returns:
        np.ndarray: Modified frame with the overlaid image blended in.

    Raises:
        ValueError: If target image doesn't have 4 channels or overlay exceeds frame bounds.
        cv.error: If resizing operation fails.
    """
    # Resize the target image if a specific size is requested
    if size:
        try:
            target_img = cv.resize(target_img, size)
        except cv.error as e:
            raise ValueError(f"Error resizing the target image: {e}")

    # Ensure the target image has an alpha channel (4th dimension)
    if target_img.shape[-1] != 4:
        raise ValueError("Target image must have 4 channels (RGBA).")

    # Split the image into its RGBA components
    b, g, r, a = cv.split(target_img)
    # Reconstruct the color image without alpha
    overlay_color = cv.merge((b, g, r))
    # Create a smooth mask from the alpha channel using median blur to reduce noise
    mask = cv.medianBlur(a, 5)

    # Get dimensions of the overlay
    h, w, _ = overlay_color.shape

    # Check if overlay fits within frame boundaries
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        raise ValueError("Overlay exceeds frame boundaries.")

    # Define the region of interest (ROI) on the frame
    roi = frame[y:y + h, x:x + w]

    # Create the background: original ROI where mask is inverted (transparent areas)
    img1_bg = cv.bitwise_and(roi, roi, mask=cv.bitwise_not(mask))
    # Create the foreground: overlay color where mask is applied (opaque areas)
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Combine background and foreground using addition (alpha blending)
    frame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

    return frame
