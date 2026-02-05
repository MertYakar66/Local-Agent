"""Screenshot management utilities for the autonomous browser agent"""

import asyncio
import io
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from loguru import logger

from src.utils.config import config
from src.utils.vision_utils import image_to_base64, draw_bbox_on_image


class ScreenshotManager:
    """
    Manages screenshot capture, storage, and processing.

    Resource Budget (from specification):
    - Screenshot Buffer: 50 recent screenshots at 1280x720 = ~500MB
    """

    def __init__(
        self,
        screenshots_dir: Optional[Path] = None,
        max_buffer_size: int = 50,
        compression: str = "PNG",
    ):
        self.screenshots_dir = screenshots_dir or config.screenshots_dir
        self.max_buffer_size = max_buffer_size
        self.compression = compression

        # Circular buffer for recent screenshots (memory optimization)
        self._screenshot_buffer: deque[Tuple[str, bytes]] = deque(maxlen=max_buffer_size)

        # Index for quick lookup
        self._screenshot_index: Dict[str, Path] = {}

    def _generate_screenshot_id(
        self,
        task_id: str,
        step_number: int,
        suffix: str = ""
    ) -> str:
        """Generate unique screenshot identifier"""
        base = f"{task_id}_step{step_number}"
        if suffix:
            base += f"_{suffix}"
        return base

    def _get_save_path(
        self,
        screenshot_id: str,
        task_id: str
    ) -> Path:
        """Get file path for saving screenshot"""
        task_dir = self.screenshots_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        extension = ".png" if self.compression == "PNG" else ".jpg"
        return task_dir / f"{screenshot_id}{extension}"

    async def capture_and_save(
        self,
        screenshot_bytes: bytes,
        task_id: str,
        step_number: int,
        suffix: str = "",
        add_to_buffer: bool = True,
    ) -> Path:
        """
        Save screenshot bytes to file and optionally add to buffer.

        Args:
            screenshot_bytes: Raw PNG bytes from Playwright
            task_id: Current task identifier
            step_number: Current step number
            suffix: Optional suffix (e.g., "before", "after")
            add_to_buffer: Whether to add to memory buffer

        Returns:
            Path to saved screenshot
        """
        screenshot_id = self._generate_screenshot_id(task_id, step_number, suffix)
        save_path = self._get_save_path(screenshot_id, task_id)

        # Process image if needed (compression, resize)
        if self.compression == "JPEG":
            image = Image.open(io.BytesIO(screenshot_bytes))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            screenshot_bytes = buffer.getvalue()

        # Save to file
        save_path.write_bytes(screenshot_bytes)

        # Add to buffer if requested
        if add_to_buffer:
            self._screenshot_buffer.append((screenshot_id, screenshot_bytes))

        # Update index
        self._screenshot_index[screenshot_id] = save_path

        logger.debug(
            f"Screenshot saved: {save_path.name} "
            f"({len(screenshot_bytes) / 1024:.1f} KB)"
        )

        return save_path

    def get_from_buffer(self, screenshot_id: str) -> Optional[bytes]:
        """Get screenshot bytes from buffer by ID"""
        for sid, data in self._screenshot_buffer:
            if sid == screenshot_id:
                return data
        return None

    def get_recent_screenshots(self, count: int = 5) -> List[Tuple[str, bytes]]:
        """Get most recent screenshots from buffer"""
        return list(self._screenshot_buffer)[-count:]

    def load_from_disk(self, screenshot_id: str) -> Optional[bytes]:
        """Load screenshot from disk by ID"""
        if screenshot_id in self._screenshot_index:
            path = self._screenshot_index[screenshot_id]
            if path.exists():
                return path.read_bytes()
        return None

    def get_screenshot_for_model(
        self,
        screenshot_bytes: bytes,
        format: str = "PNG"
    ) -> str:
        """
        Convert screenshot to base64 for model input.

        Args:
            screenshot_bytes: Raw image bytes
            format: Image format (PNG or JPEG)

        Returns:
            Base64 encoded string for model input
        """
        return image_to_base64(screenshot_bytes, format=format)

    async def create_annotated_screenshot(
        self,
        screenshot_bytes: bytes,
        bbox: Tuple[int, int, int, int],
        color: str = "red",
        save_path: Optional[Path] = None,
    ) -> bytes:
        """
        Create screenshot with bounding box annotation.

        Useful for HITL approval display.

        Args:
            screenshot_bytes: Original screenshot
            bbox: Bounding box to draw (normalized 0-1000)
            color: Box color
            save_path: Optional path to save annotated image

        Returns:
            Annotated image bytes
        """
        image = Image.open(io.BytesIO(screenshot_bytes))
        annotated = draw_bbox_on_image(image, bbox, color=color, normalized=True)

        # Convert back to bytes
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        annotated_bytes = buffer.getvalue()

        if save_path:
            save_path.write_bytes(annotated_bytes)

        return annotated_bytes

    async def cleanup_old_screenshots(
        self,
        max_age_days: int = 30,
        max_count: int = 1000,
    ) -> int:
        """
        Clean up old screenshots to save disk space.

        Args:
            max_age_days: Delete screenshots older than this
            max_count: Keep at most this many screenshots

        Returns:
            Number of screenshots deleted
        """
        deleted_count = 0
        all_screenshots = []

        # Collect all screenshot files
        for task_dir in self.screenshots_dir.iterdir():
            if task_dir.is_dir():
                for screenshot_file in task_dir.glob("*.png"):
                    stat = screenshot_file.stat()
                    all_screenshots.append((screenshot_file, stat.st_mtime))
                for screenshot_file in task_dir.glob("*.jpg"):
                    stat = screenshot_file.stat()
                    all_screenshots.append((screenshot_file, stat.st_mtime))

        # Sort by modification time (oldest first)
        all_screenshots.sort(key=lambda x: x[1])

        now = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 60 * 60

        # Delete old files
        for screenshot_path, mtime in all_screenshots:
            age_seconds = now - mtime

            should_delete = (
                age_seconds > max_age_seconds or
                len(all_screenshots) - deleted_count > max_count
            )

            if should_delete:
                try:
                    screenshot_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old screenshot: {screenshot_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {screenshot_path}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old screenshots")

        return deleted_count

    async def compress_old_screenshots(
        self,
        min_age_days: int = 7,
    ) -> int:
        """
        Convert old PNG screenshots to JPEG to save space.

        Args:
            min_age_days: Only compress screenshots older than this

        Returns:
            Number of screenshots compressed
        """
        compressed_count = 0
        now = datetime.now().timestamp()
        min_age_seconds = min_age_days * 24 * 60 * 60

        for task_dir in self.screenshots_dir.iterdir():
            if not task_dir.is_dir():
                continue

            for png_file in task_dir.glob("*.png"):
                stat = png_file.stat()
                age_seconds = now - stat.st_mtime

                if age_seconds < min_age_seconds:
                    continue

                try:
                    # Load PNG
                    image = Image.open(png_file)

                    # Save as JPEG
                    jpg_path = png_file.with_suffix(".jpg")
                    image.convert("RGB").save(jpg_path, format="JPEG", quality=85)

                    # Remove original PNG
                    png_file.unlink()

                    compressed_count += 1
                    logger.debug(f"Compressed: {png_file.name} -> {jpg_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to compress {png_file}: {e}")

        if compressed_count > 0:
            logger.info(f"Compressed {compressed_count} screenshots")

        return compressed_count

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics for screenshots"""
        total_size = 0
        file_count = 0
        task_count = 0

        for task_dir in self.screenshots_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_count += 1

            for screenshot_file in task_dir.iterdir():
                if screenshot_file.is_file():
                    file_count += 1
                    total_size += screenshot_file.stat().st_size

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "task_count": task_count,
            "buffer_size": len(self._screenshot_buffer),
            "buffer_max": self.max_buffer_size,
        }

    def clear_buffer(self) -> None:
        """Clear the screenshot buffer"""
        self._screenshot_buffer.clear()
        logger.debug("Screenshot buffer cleared")
