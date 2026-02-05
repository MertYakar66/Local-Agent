"""Browser action execution for the autonomous browser agent

Enhanced with security hardening:
- Input sanitization for all fill actions
- URL validation for navigation
- Rate limiting to prevent abuse
- Emergency stop mechanism
- Security audit logging
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout
from loguru import logger

from src.utils.config import config
from src.utils.vision_utils import denormalize_bbox, calculate_center
from src.utils.error_handling import (
    ActionError,
    RetryConfig,
    with_retry,
)
from src.utils.security import (
    SecurityLevel,
    validate_action,
    sanitize_input,
    get_emergency_stop,
    get_rate_limiter,
    get_security_log,
)


class ActionType(Enum):
    """Types of browser actions"""
    CLICK = "click"
    FILL = "fill"
    SCROLL = "scroll"
    EXTRACT = "extract"
    WAIT = "wait"
    HOVER = "hover"
    SELECT = "select"
    PRESS_KEY = "press_key"


@dataclass
class ActionResult:
    """Result of an action execution"""
    success: bool
    action_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    page_url_after: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "success": self.success,
            "action_type": self.action_type,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "extracted_data": self.extracted_data,
            "page_url_after": self.page_url_after,
            "details": self.details,
        }


# Sensitive action patterns from specification
SENSITIVE_PATTERNS = [
    r"buy|purchase|pay|checkout|confirm.*order",
    r"delete|remove|cancel",
    r"submit.*payment|enter.*credit.*card",
    r"apply.*now|sign.*contract",
]

SENSITIVE_URLS = [
    r".*/checkout",
    r".*/payment",
    r".*/cart",
    r".*/delete",
    r".*/admin",
]


def is_sensitive_action(action: Dict[str, Any], current_url: str) -> bool:
    """
    Check if action requires human approval.

    Returns True if action matches sensitive patterns.
    """
    # Check action description
    action_text = action.get("target_element", {}).get("description", "").lower()
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, action_text, re.IGNORECASE):
            return True

    # Check URL
    for pattern in SENSITIVE_URLS:
        if re.search(pattern, current_url, re.IGNORECASE):
            return True

    return False


class ActionExecutor:
    """
    Executes browser actions via Playwright API.

    Translates Qwen's bounding boxes (0-1000 scale) to pixel coordinates
    and executes click/fill/scroll actions with retry logic.
    """

    def __init__(
        self,
        page: Page,
        viewport_width: int = config.viewport_width,
        viewport_height: int = config.viewport_height,
    ):
        self.page = page
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def _bbox_to_pixels(
        self,
        bbox: List[int]
    ) -> Tuple[int, int, int, int]:
        """Convert normalized 0-1000 bbox to pixel coordinates"""
        return denormalize_bbox(
            tuple(bbox),
            self.viewport_width,
            self.viewport_height,
        )

    def _get_click_point(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center point for clicking from bbox"""
        pixel_bbox = self._bbox_to_pixels(bbox)
        return calculate_center(pixel_bbox)

    async def execute(self, action: Dict[str, Any]) -> ActionResult:
        """
        Execute an action based on Vision-Actor output.

        Args:
            action: Action dict from Vision-Actor containing:
                - action_type: click, fill, scroll, extract, wait
                - target_element: {description, bbox, confidence}
                - value: text to type (for fill actions)
                - reasoning: why this action

        Returns:
            ActionResult with success status and details

        Security:
            - Validates action before execution
            - Checks emergency stop
            - Enforces rate limits
            - Sanitizes inputs
            - Logs security events
        """
        start_time = datetime.now()
        action_type = action.get("action_type", "").lower()
        current_url = self.page.url
        security_log = get_security_log()
        emergency_stop = get_emergency_stop()
        rate_limiter = get_rate_limiter()

        logger.info(f"Executing action: {action_type}")
        logger.debug(f"Action details: {action}")

        # Security Check 1: Emergency stop
        if emergency_stop.is_stopped():
            error_msg = f"Emergency stop active: {emergency_stop.get_status()['reason']}"
            logger.error(error_msg)
            security_log.log_blocked_action(action_type, "Emergency stop active", current_url)
            return ActionResult(
                success=False,
                action_type=action_type,
                error_message=error_msg,
                page_url_after=current_url,
                duration_ms=0.0,
            )

        # Security Check 2: URL blocked
        if emergency_stop.is_url_blocked(current_url):
            error_msg = f"URL is blocked: {current_url}"
            logger.error(error_msg)
            security_log.log_blocked_action(action_type, "URL blocked", current_url)
            return ActionResult(
                success=False,
                action_type=action_type,
                error_message=error_msg,
                page_url_after=current_url,
                duration_ms=0.0,
            )

        # Security Check 3: Action blocked
        if emergency_stop.is_action_blocked(action_type):
            error_msg = f"Action type is blocked: {action_type}"
            logger.error(error_msg)
            security_log.log_blocked_action(action_type, "Action type blocked", current_url)
            return ActionResult(
                success=False,
                action_type=action_type,
                error_message=error_msg,
                page_url_after=current_url,
                duration_ms=0.0,
            )

        # Security Check 4: Rate limiting
        allowed, rate_error = rate_limiter.is_action_allowed("browser_actions")
        if not allowed:
            error_msg = f"Rate limit exceeded: {rate_error}"
            logger.warning(error_msg)
            security_log.log_blocked_action(action_type, rate_error, current_url)
            return ActionResult(
                success=False,
                action_type=action_type,
                error_message=error_msg,
                page_url_after=current_url,
                duration_ms=0.0,
            )

        # Security Check 5: Validate action
        target_element = action.get("target_element", {})
        value = action.get("value")
        is_valid, validation_msg, security_level = validate_action(
            action_type,
            target_element,
            current_url,
            value,
        )

        if not is_valid:
            error_msg = f"Action validation failed: {validation_msg}"
            logger.error(error_msg)
            security_log.log_blocked_action(action_type, validation_msg, current_url)
            return ActionResult(
                success=False,
                action_type=action_type,
                error_message=error_msg,
                page_url_after=current_url,
                duration_ms=0.0,
            )

        # Log high/critical security level actions
        if security_level in (SecurityLevel.HIGH, SecurityLevel.CRITICAL):
            logger.warning(f"[SECURITY] {security_level.value} action: {action_type} at {current_url}")

        try:
            result = await self._dispatch_action(action_type, action)
            result.page_url_after = self.page.url
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        except Exception as e:
            error_msg = f"Action '{action_type}' failed: {str(e)}"
            logger.error(error_msg)

            return ActionResult(
                success=False,
                action_type=action_type,
                error_message=error_msg,
                page_url_after=self.page.url,
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def _dispatch_action(
        self,
        action_type: str,
        action: Dict[str, Any]
    ) -> ActionResult:
        """Route to appropriate action handler"""
        handlers = {
            "click": self._execute_click,
            "fill": self._execute_fill,
            "scroll": self._execute_scroll,
            "extract": self._execute_extract,
            "wait": self._execute_wait,
            "hover": self._execute_hover,
            "select": self._execute_select,
            "press_key": self._execute_press_key,
        }

        handler = handlers.get(action_type)
        if not handler:
            raise ActionError(
                f"Unknown action type: {action_type}",
                action_type=action_type,
            )

        return await handler(action)

    @with_retry(RetryConfig(max_retries=3, base_delay=2.0))
    async def _execute_click(self, action: Dict[str, Any]) -> ActionResult:
        """
        Execute a click action.

        Coordinate Translation (from specification):
        pixel_x = (bbox[0] / 1000) * viewport_size['width']
        pixel_y = (bbox[1] / 1000) * viewport_size['height']
        """
        target = action.get("target_element", {})
        bbox = target.get("bbox", [])

        if not bbox or len(bbox) < 4:
            raise ActionError(
                "Invalid bounding box for click",
                action_type="click",
                target_element=target.get("description"),
            )

        # Get click coordinates
        click_x, click_y = self._get_click_point(bbox)

        logger.info(
            f"Clicking at ({click_x}, {click_y}) - "
            f"Element: {target.get('description', 'unknown')}"
        )

        # Wait for any animations
        await asyncio.sleep(0.1)

        # Perform click
        await self.page.mouse.click(click_x, click_y)

        # Wait for potential navigation or DOM updates
        await asyncio.sleep(0.3)

        return ActionResult(
            success=True,
            action_type="click",
            details={
                "coordinates": {"x": click_x, "y": click_y},
                "bbox": bbox,
                "element_description": target.get("description"),
            },
        )

    @with_retry(RetryConfig(max_retries=3, base_delay=2.0))
    async def _execute_fill(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a fill/type action with input sanitization"""
        target = action.get("target_element", {})
        bbox = target.get("bbox", [])
        value = action.get("value", "")

        if not bbox or len(bbox) < 4:
            raise ActionError(
                "Invalid bounding box for fill",
                action_type="fill",
                target_element=target.get("description"),
            )

        if not value:
            raise ActionError(
                "No value provided for fill action",
                action_type="fill",
            )

        # SECURITY: Sanitize input value to prevent injection
        original_value = value
        value = sanitize_input(value)

        if value != original_value:
            logger.warning(f"[SECURITY] Input was sanitized (potential injection blocked)")
            get_security_log().log_blocked_action(
                "fill",
                "Input sanitized - potential injection",
                self.page.url,
                {"original_length": len(original_value), "sanitized_length": len(value)},
            )

        # Click to focus the input
        click_x, click_y = self._get_click_point(bbox)

        logger.info(
            f"Filling input at ({click_x}, {click_y}) with '{value[:20]}...'"
        )

        # Click to focus
        await self.page.mouse.click(click_x, click_y)
        await asyncio.sleep(0.1)

        # Clear existing content and type new value
        await self.page.keyboard.press("Control+A")
        await self.page.keyboard.type(value, delay=50)  # Human-like typing speed

        return ActionResult(
            success=True,
            action_type="fill",
            details={
                "coordinates": {"x": click_x, "y": click_y},
                "bbox": bbox,
                "value_length": len(value),
                "element_description": target.get("description"),
                "was_sanitized": value != original_value,
            },
        )

    async def _execute_scroll(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a scroll action"""
        target = action.get("target_element", {})
        value = action.get("value", "down")  # up, down, left, right

        # Parse scroll direction and amount
        scroll_amount = 300  # Default scroll amount in pixels

        if isinstance(value, str):
            direction = value.lower()
        else:
            direction = "down"

        logger.info(f"Scrolling {direction} by {scroll_amount}px")

        if direction == "up":
            await self.page.evaluate(f"window.scrollBy(0, -{scroll_amount})")
        elif direction == "down":
            await self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        elif direction == "left":
            await self.page.evaluate(f"window.scrollBy(-{scroll_amount}, 0)")
        elif direction == "right":
            await self.page.evaluate(f"window.scrollBy({scroll_amount}, 0)")
        else:
            # Try to scroll to element by bbox
            if target.get("bbox"):
                click_x, click_y = self._get_click_point(target["bbox"])
                await self.page.evaluate(
                    f"window.scrollTo({click_x}, {click_y - 200})"
                )

        await asyncio.sleep(0.3)

        return ActionResult(
            success=True,
            action_type="scroll",
            details={
                "direction": direction,
                "amount": scroll_amount,
            },
        )

    async def _execute_extract(self, action: Dict[str, Any]) -> ActionResult:
        """
        Execute a data extraction action.

        Extracts text content from specified element or region.
        """
        target = action.get("target_element", {})
        description = target.get("description", "")

        logger.info(f"Extracting: {description}")

        extracted_data = {}

        # Try to extract based on description
        if "price" in description.lower():
            # Look for price patterns
            text_content = await self.page.evaluate("""
                () => {
                    const priceElements = document.querySelectorAll(
                        '[class*="price"], [class*="cost"], [data-price], .a-price'
                    );
                    const prices = [];
                    priceElements.forEach(el => {
                        const text = el.textContent.trim();
                        if (text && /\\$[\\d,]+\\.?\\d*/.test(text)) {
                            prices.push(text);
                        }
                    });
                    return prices;
                }
            """)
            extracted_data["prices"] = text_content

        elif "title" in description.lower() or "heading" in description.lower():
            # Extract titles/headings
            text_content = await self.page.evaluate("""
                () => {
                    const headings = document.querySelectorAll('h1, h2, h3, [class*="title"]');
                    return Array.from(headings).map(h => h.textContent.trim()).filter(t => t);
                }
            """)
            extracted_data["headings"] = text_content

        else:
            # General text extraction from visible area or bbox
            if target.get("bbox"):
                bbox = target["bbox"]
                pixel_bbox = self._bbox_to_pixels(bbox)
                x1, y1, x2, y2 = pixel_bbox

                text_content = await self.page.evaluate(f"""
                    () => {{
                        const elements = document.elementsFromPoint({(x1+x2)//2}, {(y1+y2)//2});
                        return elements.map(el => el.textContent).filter(t => t).join(' ').trim();
                    }}
                """)
                extracted_data["text"] = text_content[:1000]  # Limit length
            else:
                # Extract all visible text
                text_content = await self.page.evaluate("""
                    () => document.body.innerText.substring(0, 2000)
                """)
                extracted_data["text"] = text_content

        return ActionResult(
            success=True,
            action_type="extract",
            extracted_data=extracted_data,
            details={
                "description": description,
            },
        )

    async def _execute_wait(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a wait action"""
        value = action.get("value", 1.0)

        # Parse wait time
        if isinstance(value, str):
            try:
                wait_time = float(value)
            except ValueError:
                wait_time = 1.0
        else:
            wait_time = float(value) if value else 1.0

        # Cap at reasonable maximum
        wait_time = min(wait_time, 30.0)

        logger.info(f"Waiting for {wait_time} seconds")
        await asyncio.sleep(wait_time)

        return ActionResult(
            success=True,
            action_type="wait",
            details={"wait_time": wait_time},
        )

    async def _execute_hover(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a hover action"""
        target = action.get("target_element", {})
        bbox = target.get("bbox", [])

        if not bbox or len(bbox) < 4:
            raise ActionError(
                "Invalid bounding box for hover",
                action_type="hover",
                target_element=target.get("description"),
            )

        hover_x, hover_y = self._get_click_point(bbox)

        logger.info(f"Hovering at ({hover_x}, {hover_y})")
        await self.page.mouse.move(hover_x, hover_y)
        await asyncio.sleep(0.5)  # Wait for hover effects

        return ActionResult(
            success=True,
            action_type="hover",
            details={
                "coordinates": {"x": hover_x, "y": hover_y},
            },
        )

    async def _execute_select(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a select (dropdown) action"""
        target = action.get("target_element", {})
        bbox = target.get("bbox", [])
        value = action.get("value", "")

        if not value:
            raise ActionError(
                "No value provided for select action",
                action_type="select",
            )

        # First click to open dropdown
        if bbox and len(bbox) >= 4:
            click_x, click_y = self._get_click_point(bbox)
            await self.page.mouse.click(click_x, click_y)
            await asyncio.sleep(0.3)

        # Type the option or use keyboard navigation
        await self.page.keyboard.type(value)
        await self.page.keyboard.press("Enter")

        return ActionResult(
            success=True,
            action_type="select",
            details={
                "selected_value": value,
            },
        )

    async def _execute_press_key(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a keyboard key press action"""
        value = action.get("value", "Enter")

        logger.info(f"Pressing key: {value}")
        await self.page.keyboard.press(value)

        return ActionResult(
            success=True,
            action_type="press_key",
            details={"key": value},
        )

    async def scroll_element_into_view(self, bbox: List[int]) -> bool:
        """
        Scroll to make element visible.

        Used for error recovery when element is not in viewport.
        """
        try:
            center_x, center_y = self._get_click_point(bbox)

            await self.page.evaluate(f"""
                window.scrollTo({{
                    top: {center_y} - window.innerHeight / 2,
                    behavior: 'smooth'
                }})
            """)

            await asyncio.sleep(0.5)
            return True

        except Exception as e:
            logger.warning(f"Failed to scroll element into view: {e}")
            return False

    async def wait_for_page_stable(self, timeout_ms: int = 5000) -> bool:
        """Wait for page to become stable (no pending requests)"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
            return True
        except PlaywrightTimeout:
            logger.warning("Page did not become stable in time")
            return False
