"""Tests for agent modules"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.planner import PlannerAgent, TaskPlan, TaskStep
from src.agents.dom_actor import DOMActorAgent, DOMAction
from src.agents.verifier import VerifierAgent, VerificationResult


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestTaskStep:
    """Test TaskStep dataclass"""

    def test_create_step(self):
        """Test creating a TaskStep"""
        step = TaskStep(
            step=1,
            action="navigate",
            description="Open Amazon homepage",
            target="amazon.com",
            tab=0,
            verification_criteria="URL contains amazon.com",
        )

        assert step.step == 1
        assert step.action == "navigate"
        assert step.status == "pending"

    def test_step_to_dict(self):
        """Test converting step to dictionary"""
        step = TaskStep(
            step=1,
            action="click",
            description="Click search button",
            target="search button",
            tab=0,
            verification_criteria="Search results appear",
        )

        d = step.to_dict()
        assert d["step"] == 1
        assert d["action"] == "click"
        assert d["status"] == "pending"

    def test_step_from_dict(self):
        """Test creating step from dictionary"""
        data = {
            "step": 2,
            "action": "fill",
            "description": "Type search query",
            "target": "search input",
            "tab": 0,
            "verification_criteria": "Text appears in input",
        }

        step = TaskStep.from_dict(data)
        assert step.step == 2
        assert step.action == "fill"


class TestTaskPlan:
    """Test TaskPlan dataclass"""

    def test_create_plan(self):
        """Test creating a TaskPlan"""
        steps = [
            TaskStep(step=1, action="navigate", description="Open site",
                     target="example.com", tab=0, verification_criteria="Site loads"),
            TaskStep(step=2, action="click", description="Click button",
                     target="submit button", tab=0, verification_criteria="Action completes"),
        ]

        plan = TaskPlan(
            user_goal="Test goal",
            steps=steps,
        )

        assert plan.user_goal == "Test goal"
        assert len(plan.steps) == 2
        assert plan.current_step_index == 0

    def test_plan_current_step(self):
        """Test getting current step"""
        steps = [
            TaskStep(step=1, action="navigate", description="Step 1",
                     target="site", tab=0, verification_criteria=""),
            TaskStep(step=2, action="click", description="Step 2",
                     target="button", tab=0, verification_criteria=""),
        ]

        plan = TaskPlan(user_goal="Test", steps=steps)

        assert plan.current_step.step == 1
        plan.current_step_index = 1
        assert plan.current_step.step == 2

    def test_plan_is_complete(self):
        """Test checking if plan is complete"""
        steps = [
            TaskStep(step=1, action="navigate", description="Step 1",
                     target="site", tab=0, verification_criteria=""),
        ]

        plan = TaskPlan(user_goal="Test", steps=steps)

        assert not plan.is_complete

        plan.steps[0].status = "completed"
        assert plan.is_complete

    def test_plan_tabs_used(self):
        """Test getting tabs used in plan"""
        steps = [
            TaskStep(step=1, action="navigate", description="Tab 0",
                     target="site1", tab=0, verification_criteria=""),
            TaskStep(step=2, action="navigate", description="Tab 1",
                     target="site2", tab=1, verification_criteria=""),
            TaskStep(step=3, action="extract", description="Tab 0 again",
                     target="data", tab=0, verification_criteria=""),
            TaskStep(step=4, action="synthesize", description="Merge",
                     target="", tab=-1, verification_criteria=""),
        ]

        plan = TaskPlan(user_goal="Test", steps=steps)

        assert plan.tabs_used == [0, 1]

    def test_mark_step_completed(self):
        """Test marking step as completed"""
        steps = [
            TaskStep(step=1, action="navigate", description="Step 1",
                     target="site", tab=0, verification_criteria=""),
            TaskStep(step=2, action="click", description="Step 2",
                     target="button", tab=0, verification_criteria=""),
        ]

        plan = TaskPlan(user_goal="Test", steps=steps)

        plan.mark_step_completed(0, {"url": "example.com"})

        assert plan.steps[0].status == "completed"
        assert plan.steps[0].result == {"url": "example.com"}
        assert plan.current_step_index == 1


class TestDOMAction:
    """Test DOMAction dataclass"""

    def test_create_action(self):
        """Test creating a DOMAction"""
        action = DOMAction(
            action_type="click",
            selector="#search-btn",
            reasoning="Found the search button",
            confidence=0.98,
        )

        assert action.action_type == "click"
        assert action.selector == "#search-btn"
        assert action.confidence == 0.98

    def test_action_validity(self):
        """Test action validation"""
        # Valid click action
        action = DOMAction(action_type="click", selector="#btn")
        assert action.is_valid

        # Invalid click - no selector
        action_no_sel = DOMAction(action_type="click")
        assert not action_no_sel.is_valid

        # Valid press_key (no selector needed)
        action_key = DOMAction(action_type="press_key", value="Enter")
        assert action_key.is_valid

        # Valid scroll (no selector needed)
        action_scroll = DOMAction(action_type="scroll", value="down")
        assert action_scroll.is_valid

        # Invalid - no action type
        action_empty = DOMAction(action_type="")
        assert not action_empty.is_valid

    def test_action_from_dict(self):
        """Test creating DOMAction from dictionary"""
        data = {
            "action_type": "fill",
            "selector": "input[name=q]",
            "value": "test query",
            "reasoning": "Fill search input",
        }

        action = DOMAction.from_dict(data)
        assert action.action_type == "fill"
        assert action.selector == "input[name=q]"
        assert action.value == "test query"

    def test_action_to_dict(self):
        """Test converting DOMAction to dictionary"""
        action = DOMAction(
            action_type="click",
            selector="text=Submit",
            reasoning="Click submit button",
        )

        d = action.to_dict()
        assert d["action_type"] == "click"
        assert d["selector"] == "text=Submit"


class TestVerificationResult:
    """Test VerificationResult dataclass"""

    def test_create_result(self):
        """Test creating a VerificationResult"""
        result = VerificationResult(
            success=True,
            confidence=0.95,
            reasoning="URL changed as expected",
            url_changed=True,
        )

        assert result.success
        assert result.confidence == 0.95
        assert result.url_changed

    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = VerificationResult(
            success=False,
            confidence=0.7,
            reasoning="No visual change",
            retry_strategy="Try scrolling down",
        )

        d = result.to_dict()
        assert d["success"] is False
        assert d["retry_strategy"] == "Try scrolling down"


class TestPlannerAgent:
    """Test PlannerAgent with mocked LLM"""

    @pytest.mark.asyncio
    async def test_decompose_task(self):
        """Test task decomposition"""
        with patch.object(PlannerAgent, 'generate_json') as mock_gen:
            mock_gen.return_value = [
                {
                    "step": 1,
                    "action": "navigate",
                    "description": "Open Amazon",
                    "target": "amazon.com",
                    "tab": 0,
                    "verification_criteria": "URL contains amazon.com",
                },
                {
                    "step": 2,
                    "action": "fill",
                    "description": "Search for product",
                    "target": "search input",
                    "tab": 0,
                    "value": "iPhone 15",
                    "verification_criteria": "Results appear",
                },
            ]

            planner = PlannerAgent()
            plan = await planner.decompose_task("Compare iPhone prices")

            assert len(plan.steps) == 2
            assert plan.steps[0].action == "navigate"
            assert plan.steps[1].action == "fill"


class TestVerifierAgent:
    """Test VerifierAgent with mocked LLM"""

    @pytest.mark.asyncio
    async def test_verify_navigation(self):
        """Test navigation verification"""
        verifier = VerifierAgent()

        result = await verifier.verify_navigation(
            url_before="https://google.com",
            url_after="https://google.com/search?q=test",
            expected_url_pattern="search",
        )

        assert result.success
        assert result.url_changed

    @pytest.mark.asyncio
    async def test_verify_failed_navigation(self):
        """Test failed navigation verification"""
        verifier = VerifierAgent()

        result = await verifier.verify_navigation(
            url_before="https://example.com",
            url_after="https://example.com",
            expected_url_pattern="search",
        )

        assert not result.success
        assert not result.url_changed

    def test_should_retry(self):
        """Test retry decision logic"""
        verifier = VerifierAgent()

        # Should retry on failure with strategy
        result_fail = VerificationResult(
            success=False,
            confidence=0.8,
            reasoning="Click didn't work",
            retry_strategy="Try scrolling",
        )
        should_retry, reason = verifier.should_retry(result_fail, 1, 3)
        assert should_retry
        assert reason == "Try scrolling"

        # Should not retry on success
        result_success = VerificationResult(success=True, confidence=0.95, reasoning="OK")
        should_retry, _ = verifier.should_retry(result_success, 1, 3)
        assert not should_retry

        # Should not retry after max attempts
        should_retry, reason = verifier.should_retry(result_fail, 3, 3)
        assert not should_retry


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
