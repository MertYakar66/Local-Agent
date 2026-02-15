"""End-to-end integration tests for e-commerce scenarios"""

import asyncio
import pytest
from pathlib import Path
import tempfile

from src.browser.tab_manager import TabManager
from src.agents.dom_actor import DOMActorAgent, DOMAction


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
    async def test_dom_snapshot(self):
        """Test DOM snapshot extraction from a page"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.com")

            page = tab_manager.get_page(0)
            dom_actor = DOMActorAgent()

            # Get DOM snapshot
            dom = await dom_actor.get_dom_snapshot(page)

            assert len(dom) > 0
            assert "title" in dom.lower() or "elements" in dom.lower()
            print(f"✓ DOM snapshot extracted: {len(dom)} chars")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_dom_action_execution(self):
        """Test executing DOM-based actions directly via Playwright"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://example.com")

            page = tab_manager.get_page(0)
            dom_actor = DOMActorAgent()

            # Test scroll action
            scroll_action = DOMAction(
                action_type="scroll",
                value="down",
                reasoning="Scroll down to see more content",
            )
            result = await dom_actor.execute_action(page, scroll_action)
            assert result["success"]
            print("✓ Scroll action executed")

            # Test extract action
            extract_action = DOMAction(
                action_type="extract",
                selector="h1",
                reasoning="Extract page heading",
            )
            result = await dom_actor.execute_action(page, extract_action)
            assert result["success"]
            assert "extracted_data" in result
            print(f"✓ Extract action: {result['extracted_data']}")

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
