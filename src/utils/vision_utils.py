"""Vision utilities for coordinate translation and image processing"""

import base64
import io
from pathlib import Path
from typing import Tuple, Union

from PIL import Image
import numpy as np


# Normalized coordinate scale (as specified in the document)
NORMALIZED_SCALE = 1000


def normalize_bbox(
    bbox: Tuple[int, int, int, int],
    viewport_width: int,
    viewport_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert pixel coordinates to normalized 0-1000 scale.

    Args:
        bbox: Tuple of (x1, y1, x2, y2) in pixel coordinates
        viewport_width: Width of the viewport in pixels
        viewport_height: Height of the viewport in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in normalized 0-1000 scale
    """
    x1, y1, x2, y2 = bbox
    return (
        int((x1 / viewport_width) * NORMALIZED_SCALE),
        int((y1 / viewport_height) * NORMALIZED_SCALE),
        int((x2 / viewport_width) * NORMALIZED_SCALE),
        int((y2 / viewport_height) * NORMALIZED_SCALE),
    )


def denormalize_bbox(
    bbox: Tuple[int, int, int, int],
    viewport_width: int,
    viewport_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert normalized 0-1000 scale coordinates to pixel coordinates.

    As per specification:
    pixel_x = (bbox[0] / 1000) * viewport_size['width']
    pixel_y = (bbox[1] / 1000) * viewport_size['height']

    Args:
        bbox: Tuple of (x1, y1, x2, y2) in normalized 0-1000 scale
        viewport_width: Width of the viewport in pixels
        viewport_height: Height of the viewport in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    return (
        int((x1 / NORMALIZED_SCALE) * viewport_width),
        int((y1 / NORMALIZED_SCALE) * viewport_height),
        int((x2 / NORMALIZED_SCALE) * viewport_width),
        int((y2 / NORMALIZED_SCALE) * viewport_height),
    )


def calculate_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: Tuple of (x1, y1, x2, y2)

    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def validate_bbox(
    bbox: Tuple[int, int, int, int],
    max_val: int = NORMALIZED_SCALE
) -> bool:
    """
    Validate that a bounding box is well-formed.

    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        max_val: Maximum allowed value for coordinates

    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    # Check all values are within bounds
    if not all(0 <= v <= max_val for v in bbox):
        return False

    # Check that x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        return False

    return True


def resize_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image to target dimensions.

    Args:
        image: PIL Image to resize
        target_width: Target width in pixels
        target_height: Target height in pixels
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def image_to_base64(
    image: Union[Image.Image, bytes, Path, str],
    format: str = "PNG"
) -> str:
    """
    Convert image to base64 string for model input.

    Args:
        image: PIL Image, bytes, or path to image file
        format: Output format (PNG or JPEG)

    Returns:
        Base64 encoded string
    """
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string back to PIL Image.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        PIL Image
    """
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))


def calculate_image_diff(
    image1: Image.Image,
    image2: Image.Image,
    threshold: float = 0.1
) -> float:
    """
    Calculate the percentage difference between two images.
    Used by the Verifier to detect if an action caused visible change.

    Args:
        image1: First PIL Image (before action)
        image2: Second PIL Image (after action)
        threshold: Minimum pixel difference to count as changed

    Returns:
        Percentage of pixels that changed (0.0 to 100.0)
    """
    # Ensure same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Convert to numpy arrays
    arr1 = np.array(image1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.array(image2.convert("RGB"), dtype=np.float32) / 255.0

    # Calculate per-pixel difference
    diff = np.abs(arr1 - arr2)

    # Average across color channels
    diff_gray = np.mean(diff, axis=2)

    # Count pixels above threshold
    changed_pixels = np.sum(diff_gray > threshold)
    total_pixels = diff_gray.size

    return (changed_pixels / total_pixels) * 100.0


def draw_bbox_on_image(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    color: str = "red",
    width: int = 3,
    normalized: bool = True
) -> Image.Image:
    """
    Draw a bounding box on an image (for debugging/visualization).

    Args:
        image: PIL Image
        bbox: Tuple of (x1, y1, x2, y2)
        color: Box color
        width: Line width
        normalized: Whether bbox is in 0-1000 scale

    Returns:
        PIL Image with bbox drawn
    """
    from PIL import ImageDraw

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    if normalized:
        bbox = denormalize_bbox(bbox, image.width, image.height)

    draw.rectangle(bbox, outline=color, width=width)

    return img_copy


def crop_to_bbox(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    normalized: bool = True,
    padding: int = 10
) -> Image.Image:
    """
    Crop image to bounding box with optional padding.

    Args:
        image: PIL Image
        bbox: Tuple of (x1, y1, x2, y2)
        normalized: Whether bbox is in 0-1000 scale
        padding: Pixels to add around the crop

    Returns:
        Cropped PIL Image
    """
    if normalized:
        bbox = denormalize_bbox(bbox, image.width, image.height)

    x1, y1, x2, y2 = bbox

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    return image.crop((x1, y1, x2, y2))


def get_screenshot_path(
    screenshots_dir: Path,
    task_id: str,
    step_number: int,
    suffix: str = ""
) -> Path:
    """
    Generate a standardized screenshot file path.

    Args:
        screenshots_dir: Base directory for screenshots
        task_id: Unique task identifier
        step_number: Current step number
        suffix: Optional suffix (e.g., "before", "after")

    Returns:
        Path for the screenshot file
    """
    filename = f"step_{step_number}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".png"

    task_dir = screenshots_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    return task_dir / filename
