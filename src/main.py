"""Main entry point and orchestration loop for the autonomous browser agent.

DOM-based approach: Uses Playwright selectors + Claude/Ollama LLM for planning.
No screenshots or vision models needed.
"""

import asyncio
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
from src.agents.planner import PlannerAgent, TaskPlan, TaskStep
from src.agents.dom_actor import DOMActorAgent, DOMAction
from src.agents.verifier import VerifierAgent, VerificationResult
from src.memory.chroma_manager import ChromaManager, PageMemory
from src.memory.logger import StructuredLogger


# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 10
CIRCUIT_BREAKER_RESET_TIME = 300


class AgentOrchestrator:
    """
    Main orchestrator for the autonomous browser agent.

    Pipeline per step:
    1. PLANNER: Decompose goal into steps (LLM, text-only)
    2. DOM-ACTOR: Read DOM, ask LLM for selector (no screenshots)
    3. EXECUTION: Playwright runs the action directly
    4. VERIFICATION: Check URL/DOM state
    """

    def __init__(
        self,
        hitl_callback: Optional[Callable[[Dict[str, Any], bytes], bool]] = None,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.hitl_callback = hitl_callback
        self.status_callback = status_callback

        # Components (initialized in start())
        self.tab_manager: Optional[TabManager] = None
        self.planner: Optional[PlannerAgent] = None
        self.dom_actor: Optional[DOMActorAgent] = None
        self.verifier: Optional[VerifierAgent] = None
        self.memory: Optional[ChromaManager] = None
        self.structured_logger: Optional[StructuredLogger] = None

        # State
        self._running = False
        self._current_plan: Optional[TaskPlan] = None
        self._stop_requested = False

        # Security
        self._emergency_stop = get_emergency_stop()
        self._security_log = get_security_log()
        self._rate_limiter = get_rate_limiter()

        # Circuit breaker
        self._consecutive_failures = 0
        self._last_failure_time: Optional[datetime] = None

    async def start(self) -> None:
        """Initialize all components"""
        logger.info("Starting agent orchestrator...")

        # Initialize browser
        self.tab_manager = TabManager()
        await self.tab_manager.initialize()

        # Initialize agents
        self.planner = PlannerAgent()
        self.dom_actor = DOMActorAgent()
        self.verifier = VerifierAgent()

        # Initialize memory
        self.memory = ChromaManager()
        await self.memory.initialize()

        # Initialize logger
        self.structured_logger = StructuredLogger()

        # Preload model (only does work for Ollama)
        await self.planner.preload_model()

        self._running = True
        logger.info(
            f"Agent orchestrator started (LLM: {config.llm_provider}, "
            f"model: {self.planner.current_model})"
        )

    async def stop(self) -> None:
        """Cleanup and stop all components"""
        logger.info("Stopping agent orchestrator...")
        self._stop_requested = True

        if self.tab_manager:
            await self.tab_manager.close()
        for agent in [self.planner, self.dom_actor, self.verifier]:
            if agent:
                await agent.close()

        self._running = False
        logger.info("Agent orchestrator stopped")

    def _update_status(self, status: Dict[str, Any]) -> None:
        if self.status_callback:
            self.status_callback(status)

    async def _request_hitl_approval(
        self, action: Dict[str, Any], reason: str,
    ) -> bool:
        if self.hitl_callback:
            return self.hitl_callback(action, b"")
        logger.warning(f"HITL required but no callback. Reason: {reason}. Auto-reject.")
        return False

    def trigger_emergency_stop(self, reason: str) -> None:
        self._emergency_stop.trigger_stop(reason)
        self._security_log.log_event(SecurityEvent(
            timestamp=datetime.now(),
            event_type="emergency_stop_triggered",
            severity=SecurityLevel.CRITICAL,
            description=f"Emergency stop: {reason}",
        ))
        self._stop_requested = True

    def reset_emergency_stop(self) -> None:
        self._emergency_stop.reset()
        self._consecutive_failures = 0

    def _check_circuit_breaker(self) -> bool:
        if self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed > CIRCUIT_BREAKER_RESET_TIME:
                self._consecutive_failures = 0
                self._last_failure_time = None
        return self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_time = datetime.now()
        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            self.trigger_emergency_stop(
                f"Circuit breaker: {CIRCUIT_BREAKER_THRESHOLD} consecutive failures"
            )

    def _record_success(self) -> None:
        if self._consecutive_failures > 0:
            self._consecutive_failures -= 1

    async def run_task(self, user_goal: str) -> Dict[str, Any]:
        """Run a complete task from goal to completion."""
        # Security checks
        if self._emergency_stop.is_stopped():
            return {"success": False, "error": "Emergency stop active"}
        if self._check_circuit_breaker():
            return {"success": False, "error": "Circuit breaker tripped"}
        allowed, rate_error = self._rate_limiter.is_allowed("tasks")
        if not allowed:
            return {"success": False, "error": f"Rate limited: {rate_error}"}

        if not self._running:
            await self.start()

        logger.info(f"Starting task: {user_goal}")

        self._security_log.log_event(SecurityEvent(
            timestamp=datetime.now(),
            event_type="task_started",
            severity=SecurityLevel.LOW,
            description=f"Task started: {user_goal[:100]}",
            details={"goal": user_goal},
        ))

        task_id = self.structured_logger.start_task(user_goal)
        self._update_status({"stage": "planning", "task_id": task_id, "user_goal": user_goal})

        try:
            # Stage 1: PLAN
            logger.info("[Stage 1] Planning...")
            plan = await self.planner.decompose_task(user_goal)
            self._current_plan = plan
            logger.info(f"Plan: {len(plan.steps)} steps, tabs: {plan.tabs_used}")

            # Create tabs
            for tab_id in plan.tabs_used:
                if tab_id >= 0 and tab_id not in self.tab_manager.tabs:
                    await self.tab_manager.create_tab(tab_id)

            plan.status = "in_progress"
            extracted_data = []

            for step_index, step in enumerate(plan.steps):
                if self._stop_requested:
                    logger.warning("Stop requested")
                    break

                logger.info(f"[Step {step.step}] {step.description}")
                step.status = "in_progress"

                self._update_status({
                    "stage": "executing",
                    "current_step": step.to_dict(),
                    "progress": f"{step_index + 1}/{len(plan.steps)}",
                })

                step_result = await self._execute_step(step, task_id)

                if step_result.get("success"):
                    plan.mark_step_completed(step_index, step_result)
                    self.dom_actor.record_success()
                    self._record_success()

                    if step_result.get("extracted_data"):
                        extracted_data.append({
                            "step": step.step,
                            "tab": step.tab,
                            **step_result["extracted_data"]
                        })
                else:
                    error = step_result.get("error", "Unknown error")
                    plan.mark_step_failed(step_index, error)
                    self.dom_actor.record_failure()
                    self._record_failure()

                    if self._emergency_stop.is_stopped():
                        break

                    # Try plan refinement
                    if step_index < len(plan.steps) - 1:
                        try:
                            plan = await self.planner.refine_plan(plan, step, error)
                        except Exception as e:
                            logger.error(f"Plan refinement failed: {e}")

            # Synthesis
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
                    }

                from src.memory.chroma_manager import TaskPlanMemory
                await self.memory.store_task_plan(TaskPlanMemory(
                    user_goal=user_goal,
                    plan_steps=[s.to_dict() for s in plan.steps],
                    success=True,
                    execution_time_seconds=self.structured_logger.get_current_task().duration_seconds,
                ))

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

            self._security_log.log_event(SecurityEvent(
                timestamp=datetime.now(),
                event_type="task_completed",
                severity=SecurityLevel.LOW,
                description=f"Task completed: {user_goal[:50]}...",
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
            logger.error(f"Task failed: {e}")
            self.structured_logger.complete_task(success=False, error_message=str(e))
            self._update_status({"stage": "failed", "error": str(e)})
            self._record_failure()
            return {"task_id": task_id, "success": False, "error": str(e)}

    async def _execute_step(
        self, step: TaskStep, task_id: str,
    ) -> Dict[str, Any]:
        """Execute a single step: DOM analysis -> action -> verification."""
        max_retries = config.max_retries_per_action
        last_error = None

        for attempt in range(max_retries):
            try:
                if step.action == "synthesize":
                    return {"success": True, "action": "synthesize"}

                tab_id = step.tab if step.tab >= 0 else self.tab_manager.active_tab_id or 0
                page = self.tab_manager.get_page(tab_id)
                url_before = page.url

                # Navigate: handled directly
                if step.action == "navigate" and step.target:
                    await self.tab_manager.navigate(tab_id, step.target)
                    url_after = page.url

                    self._update_status({
                        "stage": "navigated",
                        "current_url": url_after,
                    })

                    # Log the step
                    self.structured_logger.log_step(
                        step_number=step.step, stage="complete", tab_id=tab_id,
                        action_type="navigate", page_url_after=url_after,
                        verification_status="success",
                    )

                    if step.target.lower() in url_after.lower():
                        return {"success": True, "action": "navigate", "url": url_after}
                    else:
                        return {"success": True, "action": "navigate", "url": url_after}

                # Press key: execute directly
                if step.action == "press_key":
                    key = step.value or "Enter"
                    await page.keyboard.press(key)
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                    url_after = page.url

                    verification = await self.verifier.verify_action(
                        page=page,
                        action={"action_type": "press_key", "value": key},
                        url_before=url_before,
                        url_after=url_after,
                        verification_criteria=step.verification_criteria,
                    )

                    self.structured_logger.log_step(
                        step_number=step.step, stage="complete", tab_id=tab_id,
                        action_type="press_key", page_url_after=url_after,
                        verification_status="success" if verification.success else "failed",
                    )

                    if verification.success:
                        return {"success": True, "action": "press_key", "url": url_after}
                    else:
                        last_error = verification.reasoning
                        continue

                # For all other actions: ask DOM actor for selector, then execute
                fill_value = step.value if step.action == "fill" else None
                context = f"Attempt {attempt + 1} failed: {last_error}" if attempt > 0 else None

                action = await self.dom_actor.get_action(
                    page=page,
                    step_description=step.description,
                    current_url=url_before,
                    fill_value=fill_value,
                    additional_context=context,
                )

                if not action.is_valid:
                    logger.warning(f"Invalid action from DOM actor")
                    if attempt < max_retries - 1:
                        continue
                    return {"success": False, "error": "Invalid action generated"}

                # Execute
                action_result = await self.dom_actor.execute_action(page, action)

                url_after = page.url

                self._update_status({
                    "stage": "action_executed",
                    "current_url": url_after,
                    "action_type": action.action_type,
                })

                if not action_result.get("success"):
                    last_error = action_result.get("error", "Action failed")
                    logger.warning(f"Action failed: {last_error}")
                    if attempt < max_retries - 1:
                        continue
                    return {"success": False, "error": last_error}

                # Verify
                verification = await self.verifier.verify_action(
                    page=page,
                    action=action.to_dict(),
                    url_before=url_before,
                    url_after=url_after,
                    verification_criteria=step.verification_criteria,
                )

                self.structured_logger.log_step(
                    step_number=step.step, stage="complete", tab_id=tab_id,
                    action_type=action.action_type,
                    reasoning=action.reasoning,
                    page_url_after=url_after,
                    verification_status="success" if verification.success else "failed",
                )

                # Store page visit
                await self.memory.store_page_visit(PageMemory(
                    url=url_after,
                    title=await page.title(),
                    action_taken=action.action_type,
                ))

                if verification.success:
                    return {
                        "success": True,
                        "action": action.action_type,
                        "extracted_data": action_result.get("extracted_data"),
                        "url": url_after,
                    }
                else:
                    last_error = verification.reasoning
                    logger.warning(f"Verification failed: {last_error}")
                    should_retry, reason = self.verifier.should_retry(
                        verification, attempt + 1, max_retries
                    )
                    if not should_retry:
                        return {"success": False, "error": last_error}

            except Exception as e:
                last_error = str(e)
                logger.error(f"Step error: {e}")
                if attempt >= max_retries - 1:
                    return {"success": False, "error": last_error}

        return {"success": False, "error": last_error or "Max retries exceeded"}

    async def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "current_plan": self._current_plan.to_dict() if self._current_plan else None,
            "tabs": await self.tab_manager.get_all_tab_states() if self.tab_manager else {},
            "security": self.get_security_status(),
        }

    def get_security_status(self) -> Dict[str, Any]:
        return {
            "emergency_stop": self._emergency_stop.get_status(),
            "circuit_breaker": {
                "consecutive_failures": self._consecutive_failures,
                "threshold": CIRCUIT_BREAKER_THRESHOLD,
                "tripped": self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD,
            },
        }


async def run_agent(user_goal: str) -> Dict[str, Any]:
    """Convenience function to run a task."""
    orchestrator = AgentOrchestrator()
    try:
        await orchestrator.start()
        return await orchestrator.run_task(user_goal)
    finally:
        await orchestrator.stop()


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous browser agent (Claude + Playwright)"
    )
    parser.add_argument("goal", nargs="?", help="The task to accomplish")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.ui:
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).parent / "ui" / "streamlit_app.py"),
        ])
    elif args.goal:
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
