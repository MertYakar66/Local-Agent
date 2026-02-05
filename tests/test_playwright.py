"""Tests for Playwright browser automation"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, Any

from playwright.async_api import async_playwright, Browser, Page


# Test configuration
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720
MAX_TABS = 10


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def browser():
    """Create browser instance for tests"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser):
    """Create page instance for tests"""
    context = await browser.new_context(
        viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
    )
    page = await context.new_page()
    yield page
    await context.close()


class TestBrowserBasics:
    """Test basic browser functionality"""

    @pytest.mark.asyncio
    async def test_browser_launches(self, browser: Browser):
        """Test that browser launches successfully"""
        assert browser.is_connected()
        print("✓ Browser launched successfully")

    @pytest.mark.asyncio
    async def test_page_creation(self, browser: Browser):
        """Test creating a new page"""
        context = await browser.new_context()
        page = await context.new_page()

        assert page is not None
        print("✓ Page created successfully")

        await context.close()

    @pytest.mark.asyncio
    async def test_viewport_size(self, page: Page):
        """Test viewport is set correctly"""
        viewport = page.viewport_size
        assert viewport["width"] == VIEWPORT_WIDTH
        assert viewport["height"] == VIEWPORT_HEIGHT
        print(f"✓ Viewport: {viewport['width']}x{viewport['height']}")


class TestNavigation:
    """Test page navigation"""

    @pytest.mark.asyncio
    async def test_navigate_to_url(self, page: Page):
        """Test navigating to a URL"""
        await page.goto("https://example.com")
        assert "example.com" in page.url
        print(f"✓ Navigated to: {page.url}")

    @pytest.mark.asyncio
    async def test_page_title(self, page: Page):
        """Test getting page title"""
        await page.goto("https://example.com")
        title = await page.title()
        assert len(title) > 0
        print(f"✓ Page title: {title}")

    @pytest.mark.asyncio
    async def test_wait_for_load(self, page: Page):
        """Test waiting for page load"""
        await page.goto("https://example.com", wait_until="networkidle")
        # Page should be stable
        assert page.url is not None
        print("✓ Page loaded with networkidle")


class TestScreenshots:
    """Test screenshot functionality"""

    @pytest.mark.asyncio
    async def test_screenshot_capture(self, page: Page):
        """Test capturing a screenshot"""
        await page.goto("https://example.com")
        screenshot = await page.screenshot()

        assert len(screenshot) > 0
        assert screenshot[:8] == b'\x89PNG\r\n\x1a\n'  # PNG signature
        print(f"✓ Screenshot captured: {len(screenshot)} bytes")

    @pytest.mark.asyncio
    async def test_screenshot_dimensions(self, page: Page):
        """Test screenshot has correct dimensions"""
        from PIL import Image
        import io

        await page.goto("https://example.com")
        screenshot = await page.screenshot()

        image = Image.open(io.BytesIO(screenshot))
        assert image.width == VIEWPORT_WIDTH
        assert image.height == VIEWPORT_HEIGHT
        print(f"✓ Screenshot dimensions: {image.width}x{image.height}")


class TestInteractions:
    """Test page interactions"""

    @pytest.mark.asyncio
    async def test_click_coordinates(self, page: Page):
        """Test clicking at coordinates"""
        await page.goto("https://example.com")

        # Click at center of page
        center_x = VIEWPORT_WIDTH // 2
        center_y = VIEWPORT_HEIGHT // 2

        await page.mouse.click(center_x, center_y)
        print(f"✓ Clicked at ({center_x}, {center_y})")

    @pytest.mark.asyncio
    async def test_keyboard_input(self, page: Page):
        """Test keyboard input"""
        await page.goto("https://example.com")

        await page.keyboard.press("Tab")
        await page.keyboard.type("test input")
        print("✓ Keyboard input successful")

    @pytest.mark.asyncio
    async def test_scroll(self, page: Page):
        """Test page scrolling"""
        await page.goto("https://example.com")

        # Scroll down
        await page.evaluate("window.scrollBy(0, 300)")
        scroll_y = await page.evaluate("window.scrollY")

        assert scroll_y > 0
        print(f"✓ Scrolled to Y={scroll_y}")


class TestMultiTab:
    """Test multi-tab functionality"""

    @pytest.mark.asyncio
    async def test_create_multiple_tabs(self, browser: Browser):
        """Test creating multiple tabs"""
        context = await browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        )

        pages = []
        for i in range(5):
            page = await context.new_page()
            pages.append(page)

        assert len(pages) == 5
        print(f"✓ Created {len(pages)} tabs")

        await context.close()

    @pytest.mark.asyncio
    async def test_navigate_multiple_tabs(self, browser: Browser):
        """Test navigating multiple tabs to different URLs"""
        context = await browser.new_context()

        urls = [
            "https://example.com",
            "https://example.org",
            "https://example.net",
        ]

        pages = []
        for url in urls:
            page = await context.new_page()
            await page.goto(url)
            pages.append(page)

        # Verify each tab has correct URL
        for i, (page, expected_url) in enumerate(zip(pages, urls)):
            assert expected_url.replace("https://", "") in page.url
            print(f"✓ Tab {i}: {page.url}")

        await context.close()

    @pytest.mark.asyncio
    async def test_tab_isolation(self, browser: Browser):
        """Test that tabs are isolated (cookies don't leak)"""
        # Create two contexts (separate cookie jars)
        context1 = await browser.new_context()
        context2 = await browser.new_context()

        page1 = await context1.new_page()
        page2 = await context2.new_page()

        # Set a cookie in context1
        await context1.add_cookies([{
            "name": "test_cookie",
            "value": "context1",
            "domain": "example.com",
            "path": "/",
        }])

        # Navigate both pages
        await page1.goto("https://example.com")
        await page2.goto("https://example.com")

        # Check cookies
        cookies1 = await context1.cookies()
        cookies2 = await context2.cookies()

        assert any(c["name"] == "test_cookie" for c in cookies1)
        assert not any(c["name"] == "test_cookie" for c in cookies2)

        print("✓ Cookie isolation verified")

        await context1.close()
        await context2.close()

    @pytest.mark.asyncio
    async def test_ten_tabs_stability(self, browser: Browser):
        """Test 10 tabs running simultaneously"""
        context = await browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        )

        pages = []
        for i in range(MAX_TABS):
            page = await context.new_page()
            await page.goto("https://example.com")
            pages.append(page)

        # All should be connected
        assert len(pages) == MAX_TABS
        for i, page in enumerate(pages):
            assert not page.is_closed()

        print(f"✓ {MAX_TABS} tabs running stably")

        # Take screenshot from each tab
        for i, page in enumerate(pages):
            screenshot = await page.screenshot()
            assert len(screenshot) > 0

        print(f"✓ All {MAX_TABS} tabs captured screenshots")

        await context.close()


class TestActionExecution:
    """Test action execution similar to agent"""

    @pytest.mark.asyncio
    async def test_fill_form(self, page: Page):
        """Test filling a form field"""
        # Create a simple HTML with an input
        await page.set_content("""
            <html>
                <body>
                    <input type="text" id="test-input" placeholder="Enter text">
                </body>
            </html>
        """)

        # Fill using coordinates (simulate agent behavior)
        input_box = await page.query_selector("#test-input")
        box = await input_box.bounding_box()

        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        await page.mouse.click(center_x, center_y)
        await page.keyboard.type("Hello World")

        value = await input_box.input_value()
        assert value == "Hello World"
        print("✓ Form fill successful")

    @pytest.mark.asyncio
    async def test_click_button(self, page: Page):
        """Test clicking a button"""
        clicked = False

        await page.set_content("""
            <html>
                <body>
                    <button id="test-button" onclick="this.textContent='Clicked!'">
                        Click Me
                    </button>
                </body>
            </html>
        """)

        button = await page.query_selector("#test-button")
        box = await button.bounding_box()

        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        await page.mouse.click(center_x, center_y)

        # Wait for text change
        await asyncio.sleep(0.1)
        text = await button.text_content()
        assert text == "Clicked!"
        print("✓ Button click successful")


class TestErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_timeout_handling(self, page: Page):
        """Test handling navigation timeout"""
        with pytest.raises(Exception):
            await page.goto("https://httpstat.us/504?sleep=60000", timeout=1000)

        print("✓ Timeout handled correctly")

    @pytest.mark.asyncio
    async def test_invalid_url(self, page: Page):
        """Test handling invalid URL"""
        with pytest.raises(Exception):
            await page.goto("not-a-valid-url")

        print("✓ Invalid URL handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
