"""Multi-tab browser management for the autonomous browser agent"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from loguru import logger

from src.utils.config import config


class TabStatus(Enum):
    """Status of a browser tab"""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class TabState:
    """State information for a browser tab"""
    tab_id: int
    page: Page
    current_url: str = ""
    title: str = ""
    status: TabStatus = TabStatus.LOADING
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None
    screenshot_path: Optional[Path] = None
    error_message: Optional[str] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging"""
        return {
            "tab_id": self.tab_id,
            "current_url": self.current_url,
            "title": self.title,
            "status": self.status.value,
            "last_action": self.last_action,
            "last_action_time": self.last_action_time.isoformat() if self.last_action_time else None,
            "screenshot_path": str(self.screenshot_path) if self.screenshot_path else None,
            "error_message": self.error_message,
        }


class TabManager:
    """
    Manages multiple Playwright tabs with state tracking.

    Designed for parallel browsing on AMD Ryzen 9800X3D:
    - Run up to 10 Playwright tabs simultaneously
    - Handle async I/O for screenshot processing and tab switching
    - Each tab uses ~600MB RAM → 10 tabs = 6GB
    """

    def __init__(
        self,
        viewport_width: int = config.viewport_width,
        viewport_height: int = config.viewport_height,
        headless: bool = config.headless,
        max_tabs: int = config.max_tabs,
    ):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.max_tabs = max_tabs

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

        self.tabs: Dict[int, TabState] = {}
        self._active_tab_id: Optional[int] = None

    async def initialize(self) -> None:
        """Initialize Playwright and browser"""
        logger.info("Initializing browser...")

        self._playwright = await async_playwright().start()

        # Launch Chromium with optimized settings
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-dev-shm-usage",  # Prevent memory exploits
                "--disable-blink-features=AutomationControlled",  # Less bot detection
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-infobars",
            ],
        )

        # Create browser context with custom settings
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
            bypass_csp=False,  # Keep CSP for security
        )

        logger.info(
            f"Browser initialized: {self.viewport_width}x{self.viewport_height}, "
            f"headless={self.headless}"
        )

    async def close(self) -> None:
        """Close browser and cleanup"""
        logger.info("Closing browser...")

        # Close all tabs
        for tab_id in list(self.tabs.keys()):
            await self.close_tab(tab_id)

        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        logger.info("Browser closed")

    async def create_tab(self, tab_id: int, initial_url: Optional[str] = None) -> TabState:
        """
        Create new tab and optionally navigate to URL.

        Args:
            tab_id: Unique identifier for this tab (0-9)
            initial_url: Optional URL to navigate to

        Returns:
            TabState for the new tab
        """
        if not self._context:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        if len(self.tabs) >= self.max_tabs:
            raise RuntimeError(f"Maximum tabs ({self.max_tabs}) reached")

        if tab_id in self.tabs:
            logger.warning(f"Tab {tab_id} already exists, returning existing tab")
            return self.tabs[tab_id]

        logger.info(f"Creating tab {tab_id}")

        page = await self._context.new_page()

        # Set up event handlers
        page.on("load", lambda: self._on_page_load(tab_id))
        page.on("crash", lambda: self._on_page_crash(tab_id))

        tab_state = TabState(
            tab_id=tab_id,
            page=page,
            status=TabStatus.READY,
        )
        self.tabs[tab_id] = tab_state

        if self._active_tab_id is None:
            self._active_tab_id = tab_id

        # Navigate if URL provided
        if initial_url:
            await self.navigate(tab_id, initial_url)

        return tab_state

    def _on_page_load(self, tab_id: int) -> None:
        """Handle page load event"""
        if tab_id in self.tabs:
            self.tabs[tab_id].status = TabStatus.READY
            logger.debug(f"Tab {tab_id} loaded")

    def _on_page_crash(self, tab_id: int) -> None:
        """Handle page crash event"""
        if tab_id in self.tabs:
            self.tabs[tab_id].status = TabStatus.ERROR
            self.tabs[tab_id].error_message = "Page crashed"
            logger.error(f"Tab {tab_id} crashed")

    async def close_tab(self, tab_id: int) -> None:
        """Close a specific tab"""
        if tab_id not in self.tabs:
            logger.warning(f"Tab {tab_id} not found")
            return

        tab_state = self.tabs[tab_id]
        try:
            await tab_state.page.close()
        except Exception as e:
            logger.warning(f"Error closing tab {tab_id}: {e}")

        tab_state.status = TabStatus.CLOSED
        del self.tabs[tab_id]

        if self._active_tab_id == tab_id:
            self._active_tab_id = next(iter(self.tabs.keys()), None)

        logger.info(f"Closed tab {tab_id}")

    async def switch_to_tab(self, tab_id: int) -> TabState:
        """
        Bring tab to foreground for action execution.

        Important: Some sites detect inactive tabs and pause scripts.
        """
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")

        self._active_tab_id = tab_id
        tab_state = self.tabs[tab_id]

        # Bring page to front (important for some sites)
        await tab_state.page.bring_to_front()

        logger.debug(f"Switched to tab {tab_id}: {tab_state.current_url}")
        return tab_state

    async def navigate(
        self,
        tab_id: int,
        url: str,
        wait_until: str = "networkidle"
    ) -> TabState:
        """
        Navigate a tab to a URL.

        Args:
            tab_id: Tab to navigate
            url: URL to navigate to
            wait_until: Playwright wait condition

        Returns:
            Updated TabState
        """
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")

        tab_state = self.tabs[tab_id]
        tab_state.status = TabStatus.LOADING

        # Add https:// if not present
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        logger.info(f"Tab {tab_id}: Navigating to {url}")

        try:
            await tab_state.page.goto(
                url,
                wait_until=wait_until,
                timeout=config.page_timeout_ms,
            )

            # Update state
            tab_state.current_url = tab_state.page.url
            tab_state.title = await tab_state.page.title()
            tab_state.status = TabStatus.READY
            tab_state.last_action = f"navigate to {url}"
            tab_state.last_action_time = datetime.now()

            logger.info(f"Tab {tab_id}: Loaded '{tab_state.title}'")

        except Exception as e:
            tab_state.status = TabStatus.ERROR
            tab_state.error_message = str(e)
            logger.error(f"Tab {tab_id}: Navigation failed - {e}")
            raise

        return tab_state

    async def get_tab_screenshot(
        self,
        tab_id: int,
        full_page: bool = False
    ) -> bytes:
        """
        Capture screenshot at configured resolution.

        Args:
            tab_id: Tab to capture
            full_page: Whether to capture full scrollable page

        Returns:
            Screenshot as PNG bytes
        """
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")

        tab_state = self.tabs[tab_id]

        # Capture screenshot
        screenshot_bytes = await tab_state.page.screenshot(
            type="png",
            full_page=full_page,
        )

        logger.debug(f"Tab {tab_id}: Screenshot captured ({len(screenshot_bytes)} bytes)")
        return screenshot_bytes

    async def save_tab_screenshot(
        self,
        tab_id: int,
        save_path: Path,
        full_page: bool = False
    ) -> Path:
        """
        Capture and save screenshot to file.

        Args:
            tab_id: Tab to capture
            save_path: Path to save screenshot
            full_page: Whether to capture full page

        Returns:
            Path to saved screenshot
        """
        screenshot_bytes = await self.get_tab_screenshot(tab_id, full_page)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(screenshot_bytes)

        self.tabs[tab_id].screenshot_path = save_path

        logger.debug(f"Tab {tab_id}: Screenshot saved to {save_path}")
        return save_path

    async def execute_action_on_tab(
        self,
        tab_id: int,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route action to correct tab based on plan.

        Args:
            tab_id: Tab to execute action on
            action: Action dict from Vision-Actor

        Returns:
            Execution result dict
        """
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")

        # Ensure we're on the correct tab
        await self.switch_to_tab(tab_id)

        # Import here to avoid circular imports
        from src.browser.action_executor import ActionExecutor

        executor = ActionExecutor(
            self.tabs[tab_id].page,
            self.viewport_width,
            self.viewport_height,
        )

        result = await executor.execute(action)

        # Update tab state
        tab_state = self.tabs[tab_id]
        tab_state.last_action = action.get("action_type", "unknown")
        tab_state.last_action_time = datetime.now()
        tab_state.current_url = tab_state.page.url

        return result

    async def get_tab_state(self, tab_id: int) -> TabState:
        """Get current state of a tab"""
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")

        tab_state = self.tabs[tab_id]

        # Refresh URL and title
        tab_state.current_url = tab_state.page.url
        try:
            tab_state.title = await tab_state.page.title()
        except Exception:
            pass

        return tab_state

    async def get_all_tab_states(self) -> Dict[int, Dict[str, Any]]:
        """Get state of all tabs as dictionaries"""
        states = {}
        for tab_id in self.tabs:
            await self.get_tab_state(tab_id)
            states[tab_id] = self.tabs[tab_id].to_dict()
        return states

    async def merge_tab_results(self, tab_ids: List[int]) -> Dict[str, Any]:
        """
        Merge extracted data from multiple tabs.

        Called by Planner at synthesis step.
        Aggregates extracted data from specified tabs.

        Args:
            tab_ids: List of tab IDs to merge data from

        Returns:
            Merged data dictionary
        """
        merged = {
            "sources": [],
            "data": [],
        }

        for tab_id in tab_ids:
            if tab_id not in self.tabs:
                logger.warning(f"Tab {tab_id} not found for merge")
                continue

            tab_state = self.tabs[tab_id]
            merged["sources"].append({
                "tab_id": tab_id,
                "url": tab_state.current_url,
                "title": tab_state.title,
            })

            if tab_state.extracted_data:
                merged["data"].append({
                    "tab_id": tab_id,
                    **tab_state.extracted_data,
                })

        logger.info(f"Merged results from {len(tab_ids)} tabs")
        return merged

    async def wait_for_tab_load(
        self,
        tab_id: int,
        timeout_ms: int = 30000
    ) -> bool:
        """
        Wait for a tab to finish loading.

        Args:
            tab_id: Tab to wait for
            timeout_ms: Maximum wait time

        Returns:
            True if loaded, False if timeout
        """
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")

        tab_state = self.tabs[tab_id]

        try:
            await tab_state.page.wait_for_load_state(
                "networkidle",
                timeout=timeout_ms,
            )
            tab_state.status = TabStatus.READY
            return True
        except Exception as e:
            logger.warning(f"Tab {tab_id}: Wait timeout - {e}")
            return False

    @property
    def active_tab_id(self) -> Optional[int]:
        """Get currently active tab ID"""
        return self._active_tab_id

    @property
    def tab_count(self) -> int:
        """Get number of open tabs"""
        return len(self.tabs)

    def get_page(self, tab_id: int) -> Page:
        """Get Playwright Page object for a tab"""
        if tab_id not in self.tabs:
            raise ValueError(f"Tab {tab_id} not found")
        return self.tabs[tab_id].page
