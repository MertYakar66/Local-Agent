"""End-to-end integration tests for e-commerce scenarios"""

import asyncio
import pytest
from pathlib import Path
import tempfile

from src.browser.tab_manager import TabManager
from src.browser.action_executor import ActionExecutor


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestEcommerceScenarios:
    """End-to-end tests for e-commerce use cases"""

    @pytest.mark.asyncio
    async def test_multi_tab_comparison_setup(self):
        """Test setting up multiple tabs for price comparison"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            # Create tabs for different "stores" (using example domains)
            stores = [
                ("example.com", 0),
                ("example.org", 1),
                ("example.net", 2),
            ]

            for url, tab_id in stores:
                await tab_manager.create_tab(tab_id)
                await tab_manager.navigate(tab_id, url)

            # Verify all tabs are open
            assert tab_manager.tab_count == 3

            # Get all tab states
            states = await tab_manager.get_all_tab_states()

            for tab_id, state in states.items():
                assert state["status"] == "ready"
                print(f"✓ Tab {tab_id}: {state['current_url']}")

            # Test merging results
            merged = await tab_manager.merge_tab_results([0, 1, 2])
            assert len(merged["sources"]) == 3
            print(f"✓ Merged {len(merged['sources'])} tab results")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_screenshot_capture_all_tabs(self):
        """Test capturing screenshots from all tabs"""
        from PIL import Image
        import io

        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            # Open multiple tabs
            for i in range(3):
                await tab_manager.create_tab(i)
                await tab_manager.navigate(i, f"https://example.{'com' if i == 0 else 'org' if i == 1 else 'net'}")

            # Capture screenshots from each
            screenshots = []
            for i in range(3):
                screenshot = await tab_manager.get_tab_screenshot(i)
                screenshots.append(screenshot)

            # Verify all screenshots
            for i, screenshot in enumerate(screenshots):
                assert len(screenshot) > 0
                image = Image.open(io.BytesIO(screenshot))
                assert image.width == 1280
                assert image.height == 720
                print(f"✓ Tab {i} screenshot: {len(screenshot)} bytes")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_action_executor_coordinates(self):
        """Test action executor coordinate translation"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.com")

            page = tab_manager.get_page(0)
            executor = ActionExecutor(page, 1280, 720)

            # Test coordinate translation
            bbox = [500, 500, 600, 600]  # Normalized 0-1000
            pixel_bbox = executor._bbox_to_pixels(bbox)

            # 500/1000 * 1280 = 640, 500/1000 * 720 = 360
            expected_x1 = int((500 / 1000) * 1280)
            expected_y1 = int((500 / 1000) * 720)

            assert pixel_bbox[0] == expected_x1
            assert pixel_bbox[1] == expected_y1
            print(f"✓ Coordinate translation: {bbox} -> {pixel_bbox}")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_scroll_action(self):
        """Test scroll action execution"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.com")

            page = tab_manager.get_page(0)
            executor = ActionExecutor(page, 1280, 720)

            # Get initial scroll position
            initial_scroll = await page.evaluate("window.scrollY")

            # Execute scroll action
            result = await executor.execute({
                "action_type": "scroll",
                "value": "down",
            })

            assert result.success
            print(f"✓ Scroll action executed")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_extract_action(self):
        """Test data extraction action"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.com")

            page = tab_manager.get_page(0)
            executor = ActionExecutor(page, 1280, 720)

            # Execute extract action
            result = await executor.execute({
                "action_type": "extract",
                "target_element": {
                    "description": "page title",
                },
            })

            assert result.success
            assert result.extracted_data is not None
            print(f"✓ Extract action: {result.extracted_data}")

        finally:
            await tab_manager.close()


class TestParallelTabOperations:
    """Test parallel operations across tabs"""

    @pytest.mark.asyncio
    async def test_concurrent_navigation(self):
        """Test navigating multiple tabs concurrently"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            # Create all tabs first
            for i in range(5):
                await tab_manager.create_tab(i)

            # Navigate all tabs concurrently
            urls = [
                "example.com",
                "example.org",
                "example.net",
                "example.com",
                "example.org",
            ]

            tasks = [
                tab_manager.navigate(i, url)
                for i, url in enumerate(urls)
            ]

            await asyncio.gather(*tasks)

            # Verify all navigated
            states = await tab_manager.get_all_tab_states()
            assert len(states) == 5

            for tab_id, state in states.items():
                assert state["status"] == "ready"
                print(f"✓ Tab {tab_id}: {state['current_url']}")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_screenshots(self):
        """Test capturing screenshots from multiple tabs concurrently"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            # Set up tabs
            for i in range(5):
                await tab_manager.create_tab(i)
                await tab_manager.navigate(i, "https://example.com")

            # Capture screenshots concurrently
            tasks = [
                tab_manager.get_tab_screenshot(i)
                for i in range(5)
            ]

            screenshots = await asyncio.gather(*tasks)

            assert len(screenshots) == 5
            for i, screenshot in enumerate(screenshots):
                assert len(screenshot) > 0
                print(f"✓ Tab {i} screenshot: {len(screenshot)} bytes")

        finally:
            await tab_manager.close()


class TestTabStability:
    """Test tab stability under load"""

    @pytest.mark.asyncio
    async def test_ten_tabs_stability(self):
        """Test 10 tabs running for extended period"""
        tab_manager = TabManager(headless=True, max_tabs=10)
        await tab_manager.initialize()

        try:
            # Create all 10 tabs
            for i in range(10):
                await tab_manager.create_tab(i)
                await tab_manager.navigate(i, "https://example.com")

            assert tab_manager.tab_count == 10

            # Perform operations on each tab
            for i in range(10):
                state = await tab_manager.get_tab_state(i)
                assert state.status.value == "ready"
                screenshot = await tab_manager.get_tab_screenshot(i)
                assert len(screenshot) > 0

            print(f"✓ All 10 tabs stable and functional")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_tab_recovery_after_close(self):
        """Test that tabs can be reopened after closing"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            # Create and close a tab
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.com")
            await tab_manager.close_tab(0)

            assert 0 not in tab_manager.tabs

            # Recreate the tab
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.org")

            state = await tab_manager.get_tab_state(0)
            assert "example.org" in state.current_url

            print("✓ Tab recovery successful")

        finally:
            await tab_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
