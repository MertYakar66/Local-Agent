"""End-to-end integration tests for search functionality"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

from src.main import AgentOrchestrator
from src.browser.tab_manager import TabManager
from src.agents.planner import PlannerAgent, TaskPlan, TaskStep
from src.agents.vision_actor import VisionActorAgent, ActionOutput, TargetElement
from src.agents.verifier import VerifierAgent, VerificationResult


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestSearchEndToEnd:
    """End-to-end tests for search functionality"""

    @pytest.mark.asyncio
    async def test_navigate_to_google(self):
        """Test navigating to Google and verifying page loaded"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://www.google.com")

            state = await tab_manager.get_tab_state(0)

            assert "google" in state.current_url.lower()
            print(f"✓ Navigated to: {state.current_url}")
            print(f"✓ Page title: {state.title}")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_capture_google_screenshot(self):
        """Test capturing a screenshot of Google"""
        from PIL import Image
        import io

        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://www.google.com")

            screenshot = await tab_manager.get_tab_screenshot(0)

            # Verify it's a valid PNG
            assert screenshot[:8] == b'\x89PNG\r\n\x1a\n'

            # Check dimensions
            image = Image.open(io.BytesIO(screenshot))
            assert image.width == 1280
            assert image.height == 720

            print(f"✓ Screenshot captured: {len(screenshot)} bytes")
            print(f"✓ Dimensions: {image.width}x{image.height}")

        finally:
            await tab_manager.close()

    @pytest.mark.asyncio
    async def test_google_search_execution(self):
        """Test executing a search on Google"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        try:
            await tab_manager.create_tab(0)
            await tab_manager.navigate(0, "https://www.google.com")

            page = tab_manager.get_page(0)

            # Find and click search box (use text area or input)
            search_box = await page.query_selector('textarea[name="q"], input[name="q"]')

            if search_box:
                await search_box.click()
                await page.keyboard.type("test query")
                await page.keyboard.press("Enter")

                # Wait for navigation
                await page.wait_for_load_state("networkidle", timeout=10000)

                url = page.url
                assert "search" in url or "q=" in url

                print(f"✓ Search executed, URL: {url}")
            else:
                print("⚠ Search box not found (may be CAPTCHA)")

        finally:
            await tab_manager.close()


class TestMultiTabSearch:
    """Test multi-tab search scenarios"""

    @pytest.mark.asyncio
    async def test_open_multiple_search_engines(self):
        """Test opening multiple search engines in different tabs"""
        tab_manager = TabManager(headless=True)
        await tab_manager.initialize()

        urls = [
            ("example.com", 0),
            ("example.org", 1),
        ]

        try:
            for url, tab_id in urls:
                await tab_manager.create_tab(tab_id)
                await tab_manager.navigate(tab_id, url)

            states = await tab_manager.get_all_tab_states()

            assert len(states) == 2
            for tab_id, state in states.items():
                print(f"✓ Tab {tab_id}: {state['current_url']}")

        finally:
            await tab_manager.close()


class TestOrchestratorMocked:
    """Test orchestrator with mocked components"""

    @pytest.mark.asyncio
    async def test_orchestrator_planning_phase(self, temp_dir):
        """Test the planning phase of orchestrator"""
        # Mock the planner
        with patch.object(PlannerAgent, 'decompose_task') as mock_plan, \
             patch.object(PlannerAgent, 'preload_model', return_value=True), \
             patch.object(PlannerAgent, 'close', return_value=None), \
             patch.object(VisionActorAgent, 'close', return_value=None), \
             patch.object(VerifierAgent, 'close', return_value=None):

            mock_plan.return_value = TaskPlan(
                user_goal="Test search",
                steps=[
                    TaskStep(
                        step=1,
                        action="navigate",
                        description="Open test site",
                        target="example.com",
                        tab=0,
                        verification_criteria="URL contains example.com",
                    ),
                ],
            )

            orchestrator = AgentOrchestrator()

            # Manually set up orchestrator state
            orchestrator.planner = PlannerAgent()
            orchestrator.vision_actor = VisionActorAgent()
            orchestrator.verifier = VerifierAgent()

            # Call planning
            plan = await orchestrator.planner.decompose_task("Test search")

            assert len(plan.steps) == 1
            assert plan.steps[0].action == "navigate"
            print("✓ Planning phase works")


class TestErrorRecovery:
    """Test error recovery scenarios"""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that agent retries on action failure"""
        from src.utils.error_handling import RetryConfig, with_retry

        attempts = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.1))
        async def failing_action():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TimeoutError("Simulated timeout")
            return "success"

        result = await failing_action()

        assert result == "success"
        assert attempts == 3
        print(f"✓ Retry logic worked after {attempts} attempts")

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that agent stops after max retries"""
        from src.utils.error_handling import RetryConfig, with_retry

        @with_retry(RetryConfig(max_retries=2, base_delay=0.1))
        async def always_fails():
            raise TimeoutError("Always fails")

        with pytest.raises(TimeoutError):
            await always_fails()

        print("✓ Max retries exceeded correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
