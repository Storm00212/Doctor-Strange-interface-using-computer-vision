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

    # Define the specific landmark indices we need for gesture analysis
    # MediaPipe hand model has 21 landmarks total (0-20)
    # We select key points for wrist, fingertips, and palm positions
    keys = [0, 4, 5, 8, 9, 12, 16, 20]  # Specific indices for our analysis

    # Extract and return the coordinates for these key landmarks
    # This creates a list of (x, y) tuples in the order specified above
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
    # Convert tuples to numpy arrays for vectorized operations
    # Subtract the arrays to get the difference vector (dx, dy)
    # Use numpy's linalg.norm to compute Euclidean distance: sqrt(dx^2 + dy^2)
    # This is more efficient and numerically stable than manual calculation
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
    # First, draw the outer line with the specified color and full thickness
    # This creates the main visible line of the magical connection
    cv.line(frame, p1, p2, color, thickness)

    # Then draw a thinner inner white line on top for a glowing effect
    # The white line is centered on the colored line, creating a highlight
    # max(1, thickness // 2) ensures minimum thickness of 1 pixel
    cv.line(frame, p1, p2, WHITE_COLOR, max(1, thickness // 2))

    # Return the modified frame with the glowing line effect
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
    # Step 1: Resize the overlay image if a specific size is provided
    # This allows dynamic scaling of the magical circles based on hand size
    if size:
        try:
            # Use OpenCV's resize function with default interpolation
            target_img = cv.resize(target_img, size)
        except cv.error as e:
            # Handle any resizing errors (e.g., invalid size parameters)
            raise ValueError(f"Error resizing the target image: {e}")

    # Step 2: Validate that the image has an alpha channel
    # Alpha channel is crucial for transparency effects
    if target_img.shape[-1] != 4:
        raise ValueError("Target image must have 4 channels (RGBA).")

    # Step 3: Separate the image into its color and alpha components
    # Split into Blue, Green, Red, and Alpha channels
    b, g, r, a = cv.split(target_img)

    # Step 4: Reconstruct the color image (BGR format for OpenCV)
    overlay_color = cv.merge((b, g, r))

    # Step 5: Create a smooth transparency mask from the alpha channel
    # Median blur reduces noise in the alpha channel for cleaner edges
    mask = cv.medianBlur(a, 5)

    # Step 6: Get the dimensions of the overlay image
    h, w, _ = overlay_color.shape

    # Step 7: Boundary check - ensure overlay doesn't exceed frame dimensions
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        raise ValueError("Overlay exceeds frame boundaries.")

    # Step 8: Define the region of interest (ROI) on the destination frame
    roi = frame[y:y + h, x:x + w]

    # Step 9: Alpha blending process
    # Create background layer: original pixels where overlay is transparent
    img1_bg = cv.bitwise_and(roi, roi, mask=cv.bitwise_not(mask))

    # Create foreground layer: overlay pixels where overlay is opaque
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Step 10: Combine the layers using addition (perfect alpha blending)
    # This creates the final composite where transparent areas show background
    frame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

    # Return the frame with the overlay applied
    return frame
