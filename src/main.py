"""Main entry point and orchestration loop for the autonomous browser agent

Enhanced with security hardening:
- Emergency stop mechanism
- Circuit breaker pattern
- Security audit logging
- Rate limiting protection
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.utils.config import config
from src.utils.error_handling import HITLRequired, AgentError
from src.utils.security import (
    get_emergency_stop,
    get_security_log,
    get_rate_limiter,
    SecurityLevel,
    SecurityEvent,
)
from src.browser.tab_manager import TabManager
from src.browser.action_executor import ActionExecutor, is_sensitive_action
from src.browser.screenshot_utils import ScreenshotManager
from src.agents.planner import PlannerAgent, TaskPlan, TaskStep
from src.agents.vision_actor import VisionActorAgent, ActionOutput
from src.agents.verifier import VerifierAgent, VerificationResult
from src.memory.chroma_manager import ChromaManager, PageMemory, ExtractedData
from src.memory.logger import StructuredLogger


# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 10  # Stop after 10 consecutive failures
CIRCUIT_BREAKER_RESET_TIME = 300  # Reset after 5 minutes


class AgentOrchestrator:
    """
    Main orchestrator for the autonomous browser agent.

    Implements the 4-stage ReAct-Plus loop:
    1. PLANNER: Decomposes goal into steps
    2. VISION-ACTION: Executes one step with screenshot analysis
    3. EXECUTION: Performs Playwright interaction
    4. VERIFICATION: Checks success and updates memory

    Security Features:
    - Emergency stop mechanism
    - Circuit breaker pattern for failure protection
    - Security audit logging
    - Rate limiting protection
    """

    def __init__(
        self,
        hitl_callback: Optional[Callable[[Dict[str, Any], bytes], bool]] = None,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            hitl_callback: Callback for human-in-the-loop approval
                          Takes (action, screenshot) -> approved (bool)
            status_callback: Callback for status updates to UI
        """
        self.hitl_callback = hitl_callback
        self.status_callback = status_callback

        # Components (initialized in start())
        self.tab_manager: Optional[TabManager] = None
        self.screenshot_manager: Optional[ScreenshotManager] = None
        self.planner: Optional[PlannerAgent] = None
        self.vision_actor: Optional[VisionActorAgent] = None
        self.verifier: Optional[VerifierAgent] = None
        self.memory: Optional[ChromaManager] = None
        self.structured_logger: Optional[StructuredLogger] = None

        # State
        self._running = False
        self._current_plan: Optional[TaskPlan] = None
        self._stop_requested = False

        # Security components
        self._emergency_stop = get_emergency_stop()
        self._security_log = get_security_log()
        self._rate_limiter = get_rate_limiter()

        # Circuit breaker state
        self._consecutive_failures = 0
        self._last_failure_time: Optional[datetime] = None

    async def start(self) -> None:
        """Initialize all components"""
        logger.info("Starting agent orchestrator...")

        # Initialize browser
        self.tab_manager = TabManager()
        await self.tab_manager.initialize()

        # Initialize screenshot manager
        self.screenshot_manager = ScreenshotManager()

        # Initialize agents
        self.planner = PlannerAgent()
        self.vision_actor = VisionActorAgent()
        self.verifier = VerifierAgent()

        # Initialize memory
        self.memory = ChromaManager()
        await self.memory.initialize()

        # Initialize logger
        self.structured_logger = StructuredLogger()

        # Preload models
        await self.planner.preload_model()

        self._running = True
        logger.info("Agent orchestrator started successfully")

    async def stop(self) -> None:
        """Cleanup and stop all components"""
        logger.info("Stopping agent orchestrator...")

        self._stop_requested = True

        if self.tab_manager:
            await self.tab_manager.close()

        if self.planner:
            await self.planner.close()
        if self.vision_actor:
            await self.vision_actor.close()
        if self.verifier:
            await self.verifier.close()

        self._running = False
        logger.info("Agent orchestrator stopped")

    def _update_status(self, status: Dict[str, Any]) -> None:
        """Send status update to UI"""
        if self.status_callback:
            self.status_callback(status)

    async def _request_hitl_approval(
        self,
        action: Dict[str, Any],
        screenshot: bytes,
        reason: str,
    ) -> bool:
        """Request human approval for sensitive action"""
        if self.hitl_callback:
            return self.hitl_callback(action, screenshot)

        # Default: log and auto-reject
        logger.warning(
            f"HITL approval required but no callback set. "
            f"Reason: {reason}. Auto-rejecting."
        )
        return False

    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop - halts all operations"""
        self._emergency_stop.trigger_stop(reason)
        self._security_log.log_event(SecurityEvent(
            timestamp=datetime.now(),
            event_type="emergency_stop_triggered",
            severity=SecurityLevel.CRITICAL,
            description=f"Emergency stop triggered: {reason}",
        ))
        self._stop_requested = True

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop to allow operations to resume"""
        self._emergency_stop.reset()
        self._consecutive_failures = 0
        self._security_log.log_event(SecurityEvent(
            timestamp=datetime.now(),
            event_type="emergency_stop_reset",
            severity=SecurityLevel.HIGH,
            description="Emergency stop reset - operations can resume",
        ))

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is tripped (too many failures)"""
        # Reset after timeout
        if self._last_failure_time:
            time_since_failure = (datetime.now() - self._last_failure_time).total_seconds()
            if time_since_failure > CIRCUIT_BREAKER_RESET_TIME:
                self._consecutive_failures = 0
                self._last_failure_time = None

        return self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD

    def _record_circuit_breaker_failure(self) -> None:
        """Record a failure for circuit breaker"""
        self._consecutive_failures += 1
        self._last_failure_time = datetime.now()

        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            self.trigger_emergency_stop(
                f"Circuit breaker tripped: {CIRCUIT_BREAKER_THRESHOLD} consecutive failures"
            )

    def _record_circuit_breaker_success(self) -> None:
        """Record a success - reduces circuit breaker count"""
        if self._consecutive_failures > 0:
            self._consecutive_failures -= 1

    async def run_task(self, user_goal: str) -> Dict[str, Any]:
        """
        Run a complete task from goal to completion.

        This is the main entry point for task execution.

        Args:
            user_goal: The high-level task description

        Returns:
            Task result dictionary

        Security:
            - Checks emergency stop before starting
            - Validates rate limits
            - Monitors for circuit breaker trips
        """
        # Security Check 1: Emergency stop
        if self._emergency_stop.is_stopped():
            error_msg = f"Cannot start task - emergency stop active: {self._emergency_stop.get_status()['reason']}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "emergency_stop": True,
            }

        # Security Check 2: Circuit breaker
        if self._check_circuit_breaker():
            error_msg = "Cannot start task - circuit breaker tripped (too many failures)"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "circuit_breaker": True,
            }

        # Security Check 3: Rate limiting
        allowed, rate_error = self._rate_limiter.is_allowed("tasks")
        if not allowed:
            error_msg = f"Cannot start task - {rate_error}"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "rate_limited": True,
            }

        if not self._running:
            await self.start()

        logger.info(f"Starting task: {user_goal}")

        # Log task start for security audit
        self._security_log.log_event(SecurityEvent(
            timestamp=datetime.now(),
            event_type="task_started",
            severity=SecurityLevel.LOW,
            description=f"Task started: {user_goal[:100]}",
            details={"goal": user_goal},
        ))

        # Start logging
        task_id = self.structured_logger.start_task(user_goal)

        self._update_status({
            "stage": "planning",
            "task_id": task_id,
            "user_goal": user_goal,
        })

        try:
            # Stage 1: PLANNER - Decompose task into steps
            logger.info("[Stage 1] Planning task decomposition...")

            # Check for similar past plans
            similar_plans = await self.memory.find_similar_plans(user_goal, n_results=1)
            context = None
            if similar_plans:
                logger.info("Found similar past plan, using as reference")
                context = f"SIMILAR PAST PLAN (for reference):\n{similar_plans[0]}"

            plan = await self.planner.decompose_task(user_goal, context=context)
            self._current_plan = plan

            logger.info(f"Plan created with {len(plan.steps)} steps using tabs {plan.tabs_used}")

            # Create required tabs
            for tab_id in plan.tabs_used:
                if tab_id >= 0 and tab_id not in self.tab_manager.tabs:
                    await self.tab_manager.create_tab(tab_id)

            # Execute plan steps
            plan.status = "in_progress"
            extracted_data = []

            for step_index, step in enumerate(plan.steps):
                if self._stop_requested:
                    logger.warning("Stop requested, aborting task")
                    break

                logger.info(f"[Step {step.step}] {step.description}")
                step.status = "in_progress"

                # Get current screenshot for UI before step execution
                try:
                    tab_id = step.tab if step.tab >= 0 else self.tab_manager.active_tab_id or 0
                    if tab_id in self.tab_manager.tabs:
                        pre_screenshot = await self.tab_manager.get_tab_screenshot(tab_id)
                        tab_states = await self.tab_manager.get_all_tab_states()
                    else:
                        pre_screenshot = None
                        tab_states = {}
                except Exception:
                    pre_screenshot = None
                    tab_states = {}

                self._update_status({
                    "stage": "executing",
                    "current_step": step.to_dict(),
                    "progress": f"{step_index + 1}/{len(plan.steps)}",
                    "screenshot": pre_screenshot,
                    "tabs": tab_states,
                })

                # Execute step with retry logic
                step_result = await self._execute_step(step, task_id)

                if step_result.get("success"):
                    plan.mark_step_completed(step_index, step_result)

                    # Record success for tiered intelligence (may switch back to primary model)
                    self.vision_actor.record_success()

                    # Record success for circuit breaker
                    self._record_circuit_breaker_success()

                    # Collect extracted data
                    if step_result.get("extracted_data"):
                        extracted_data.append({
                            "step": step.step,
                            "tab": step.tab,
                            **step_result["extracted_data"]
                        })
                else:
                    # Handle step failure
                    error = step_result.get("error", "Unknown error")
                    plan.mark_step_failed(step_index, error)

                    # Record failure for tiered intelligence (may trigger fallback to 30b)
                    switched_to_fallback = self.vision_actor.record_failure()
                    if switched_to_fallback:
                        logger.warning(
                            "[Tiered Intelligence] Switched to fallback model for deeper analysis"
                        )

                    # Record failure for circuit breaker
                    self._record_circuit_breaker_failure()

                    # Check if emergency stop was triggered
                    if self._emergency_stop.is_stopped():
                        logger.error("Emergency stop triggered - aborting task")
                        break

                    # Try to refine plan
                    if step_index < len(plan.steps) - 1:
                        try:
                            plan = await self.planner.refine_plan(
                                plan, step, error
                            )
                            logger.info("Plan refined after failure")
                        except Exception as e:
                            logger.error(f"Failed to refine plan: {e}")

            # Stage: SYNTHESIS (if plan completed)
            final_result = None
            if plan.is_complete:
                logger.info("[Final] Synthesizing results...")

                if extracted_data:
                    synthesis_prompt = await self.planner.create_synthesis_prompt(
                        user_goal, extracted_data
                    )
                    final_result = {
                        "synthesis": await self.planner.generate(synthesis_prompt),
                        "extracted_data": extracted_data,
                        "tabs_used": plan.tabs_used,
                    }

                # Store successful plan in memory
                from src.memory.chroma_manager import TaskPlanMemory
                await self.memory.store_task_plan(TaskPlanMemory(
                    user_goal=user_goal,
                    plan_steps=[s.to_dict() for s in plan.steps],
                    success=True,
                    execution_time_seconds=self.structured_logger.get_current_task().duration_seconds,
                ))

            # Complete logging
            task_log = self.structured_logger.complete_task(
                success=plan.is_complete,
                final_result=final_result,
                error_message=None if plan.is_complete else "Task incomplete",
            )

            self._update_status({
                "stage": "completed",
                "success": plan.is_complete,
                "result": final_result,
            })

            # Log successful task completion
            self._security_log.log_event(SecurityEvent(
                timestamp=datetime.now(),
                event_type="task_completed",
                severity=SecurityLevel.LOW,
                description=f"Task completed successfully: {user_goal[:50]}...",
                details={
                    "task_id": task_id,
                    "steps_completed": plan.current_step_index,
                    "total_steps": len(plan.steps),
                },
            ))

            return {
                "task_id": task_id,
                "success": plan.is_complete,
                "steps_completed": plan.current_step_index,
                "total_steps": len(plan.steps),
                "result": final_result,
                "duration_seconds": task_log.duration_seconds,
            }

        except Exception as e:
            logger.error(f"Task failed with error: {e}")

            self.structured_logger.complete_task(
                success=False,
                error_message=str(e),
            )

            self._update_status({
                "stage": "failed",
                "error": str(e),
            })

            # Log task failure for security audit
            self._security_log.log_event(SecurityEvent(
                timestamp=datetime.now(),
                event_type="task_failed",
                severity=SecurityLevel.MEDIUM,
                description=f"Task failed: {str(e)[:100]}",
                details={
                    "task_id": task_id,
                    "error": str(e),
                },
            ))

            # Record failure for circuit breaker
            self._record_circuit_breaker_failure()

            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
            }

    async def _execute_step(
        self,
        step: TaskStep,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Execute a single step with the full pipeline.

        Stages:
        2. VISION-ACTION: Analyze screenshot, get action
        3. EXECUTION: Perform action via Playwright
        4. VERIFICATION: Check success
        """
        max_retries = config.max_retries_per_action
        last_error = None

        for attempt in range(max_retries):
            try:
                # Handle special actions
                if step.action == "synthesize":
                    # Synthesis is handled at the end
                    return {"success": True, "action": "synthesize"}

                # Get current screenshot
                tab_id = step.tab if step.tab >= 0 else self.tab_manager.active_tab_id or 0
                tab_state = await self.tab_manager.get_tab_state(tab_id)
                url_before = tab_state.current_url

                # Navigate if needed
                if step.action == "navigate" and step.target:
                    await self.tab_manager.navigate(tab_id, step.target)
                    tab_state = await self.tab_manager.get_tab_state(tab_id)

                    # Send screenshot and tab update after navigation
                    try:
                        nav_screenshot = await self.tab_manager.get_tab_screenshot(tab_id)
                        tab_states = await self.tab_manager.get_all_tab_states()
                        self._update_status({
                            "stage": "navigated",
                            "screenshot": nav_screenshot,
                            "tabs": tab_states,
                            "current_url": tab_state.current_url,
                        })
                    except Exception:
                        pass

                    # Verify navigation
                    if step.target.lower() in tab_state.current_url.lower():
                        return {
                            "success": True,
                            "action": "navigate",
                            "url": tab_state.current_url,
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Navigation failed, URL: {tab_state.current_url}",
                        }

                # Capture screenshot
                screenshot_before = await self.tab_manager.get_tab_screenshot(tab_id)

                # Save screenshot
                screenshot_path = await self.screenshot_manager.capture_and_save(
                    screenshot_before,
                    task_id,
                    step.step,
                    suffix="before",
                )

                # Stage 2: VISION-ACTION
                action_context = None
                if attempt > 0:
                    action_context = f"Previous attempt {attempt} failed: {last_error}"

                action = await self.vision_actor.get_action(
                    screenshot_bytes=screenshot_before,
                    step_description=step.description,
                    current_url=tab_state.current_url,
                    additional_context=action_context,
                )

                # Validate action
                if not action.is_valid:
                    logger.warning(f"Invalid action: confidence={action.confidence}")
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "success": False,
                        "error": "Could not identify valid action target",
                    }

                # Check for sensitive actions (HITL)
                if config.require_approval_for_sensitive:
                    action_dict = action.to_dict()
                    if is_sensitive_action(action_dict, tab_state.current_url):
                        logger.warning("Sensitive action detected, requesting approval")

                        approved = await self._request_hitl_approval(
                            action_dict,
                            screenshot_before,
                            reason=f"Sensitive action: {action.action_type} on {action.target_element.description if action.target_element else 'unknown'}",
                        )

                        if not approved:
                            return {
                                "success": False,
                                "error": "Action rejected by user",
                                "hitl_rejected": True,
                            }

                # Stage 3: EXECUTION
                executor = ActionExecutor(
                    self.tab_manager.get_page(tab_id),
                    config.viewport_width,
                    config.viewport_height,
                )

                action_result = await executor.execute(action.to_dict())

                # Capture screenshot after action
                screenshot_after = await self.tab_manager.get_tab_screenshot(tab_id)
                url_after = self.tab_manager.get_page(tab_id).url

                await self.screenshot_manager.capture_and_save(
                    screenshot_after,
                    task_id,
                    step.step,
                    suffix="after",
                )

                # Send updated screenshot and tabs to UI
                try:
                    tab_states = await self.tab_manager.get_all_tab_states()
                    self._update_status({
                        "stage": "action_executed",
                        "screenshot": screenshot_after,
                        "tabs": tab_states,
                        "current_url": url_after,
                        "action_type": action.action_type,
                    })
                except Exception:
                    pass

                # Stage 4: VERIFICATION
                verification = await self.verifier.verify_action(
                    before_screenshot=screenshot_before,
                    after_screenshot=screenshot_after,
                    action=action.to_dict(),
                    url_before=url_before,
                    url_after=url_after,
                    verification_criteria=step.verification_criteria,
                )

                # Log step
                self.structured_logger.log_step(
                    step_number=step.step,
                    stage="complete",
                    tab_id=tab_id,
                    action_type=action.action_type,
                    bbox=list(action.target_element.bbox) if action.target_element else None,
                    confidence=action.confidence,
                    reasoning=action.reasoning,
                    playwright_result="success" if action_result.success else "failed",
                    page_url_after=url_after,
                    duration_ms=action_result.duration_ms,
                    verification_status="success" if verification.success else "failed",
                    screenshot_path=str(screenshot_path),
                )

                # Store page visit in memory
                await self.memory.store_page_visit(PageMemory(
                    url=url_after,
                    title=await self.tab_manager.get_page(tab_id).title(),
                    screenshot_path=str(screenshot_path),
                    action_taken=action.action_type,
                ))

                if verification.success:
                    return {
                        "success": True,
                        "action": action.action_type,
                        "extracted_data": action_result.extracted_data,
                        "url": url_after,
                    }
                else:
                    last_error = verification.reasoning
                    logger.warning(f"Verification failed: {last_error}")

                    # Should we retry?
                    should_retry, reason = self.verifier.should_retry(
                        verification, attempt + 1, max_retries
                    )
                    if not should_retry:
                        return {
                            "success": False,
                            "error": last_error,
                        }

            except Exception as e:
                last_error = str(e)
                logger.error(f"Step execution error: {e}")

                if attempt >= max_retries - 1:
                    return {
                        "success": False,
                        "error": last_error,
                    }

        return {
            "success": False,
            "error": last_error or "Max retries exceeded",
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status including security status"""
        return {
            "running": self._running,
            "current_plan": self._current_plan.to_dict() if self._current_plan else None,
            "tabs": await self.tab_manager.get_all_tab_states() if self.tab_manager else {},
            "memory_stats": await self.memory.get_stats() if self.memory else {},
            "log_stats": self.structured_logger.get_stats() if self.structured_logger else {},
            "security": self.get_security_status(),
        }

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "emergency_stop": self._emergency_stop.get_status(),
            "circuit_breaker": {
                "consecutive_failures": self._consecutive_failures,
                "threshold": CIRCUIT_BREAKER_THRESHOLD,
                "tripped": self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            },
            "recent_security_events": len(self._security_log.get_recent_events(hours=1)),
            "blocked_actions_24h": self._security_log.get_blocked_count(hours=24),
        }


async def run_agent(user_goal: str) -> Dict[str, Any]:
    """
    Convenience function to run a task.

    Args:
        user_goal: The task to accomplish

    Returns:
        Task result
    """
    orchestrator = AgentOrchestrator()

    try:
        await orchestrator.start()
        result = await orchestrator.run_task(user_goal)
        return result
    finally:
        await orchestrator.stop()


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vision-driven autonomous browser agent"
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="The task to accomplish",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit dashboard instead of CLI",
    )

    args = parser.parse_args()

    if args.ui:
        # Launch Streamlit UI
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).parent / "ui" / "streamlit_app.py"),
        ])
    elif args.goal:
        # Run task from CLI
        result = asyncio.run(run_agent(args.goal))
        print(f"\nTask Result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Steps: {result.get('steps_completed')}/{result.get('total_steps')}")
        if result.get("result"):
            print(f"  Result: {result['result']}")
        if result.get("error"):
            print(f"  Error: {result['error']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
