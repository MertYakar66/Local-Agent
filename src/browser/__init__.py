"""Browser automation module for the autonomous browser agent"""

from src.browser.tab_manager import TabManager, TabState
from src.browser.action_executor import ActionExecutor, ActionResult
from src.browser.screenshot_utils import ScreenshotManager

__all__ = [
    "TabManager",
    "TabState",
    "ActionExecutor",
    "ActionResult",
    "ScreenshotManager",
]
