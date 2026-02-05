"""Error handling and retry logic for the autonomous browser agent"""

import asyncio
import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar, ParamSpec
from loguru import logger


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"  # Can be retried automatically
    MEDIUM = "medium"  # May require alternative approach
    HIGH = "high"  # Requires user intervention
    CRITICAL = "critical"  # System should stop


class AgentError(Exception):
    """Base exception for all agent errors"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recoverable = recoverable
        self.context = context or {}


class ActionError(AgentError):
    """Error during action execution (click, fill, scroll, etc.)"""

    def __init__(
        self,
        message: str,
        action_type: Optional[str] = None,
        target_element: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.action_type = action_type
        self.target_element = target_element


class VerificationError(AgentError):
    """Error during action verification"""

    def __init__(
        self,
        message: str,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.expected = expected
        self.actual = actual


class BrowserError(AgentError):
    """Error related to browser/Playwright operations"""

    def __init__(
        self,
        message: str,
        tab_id: Optional[int] = None,
        url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.tab_id = tab_id
        self.url = url


class ModelError(AgentError):
    """Error related to LLM inference"""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        prompt_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.prompt_length = prompt_length


class MemoryError(AgentError):
    """Error related to ChromaDB or logging"""

    def __init__(
        self,
        message: str,
        collection: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.collection = collection


class HITLRequired(AgentError):
    """Exception raised when human approval is required"""

    def __init__(
        self,
        message: str,
        action: dict[str, Any],
        screenshot_path: Optional[str] = None,
        reason: Optional[str] = None
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            recoverable=True
        )
        self.action = action
        self.screenshot_path = screenshot_path
        self.reason = reason


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 2.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_backoff: bool = True
    retry_on: tuple = (ActionError, BrowserError, TimeoutError)
    on_retry: Optional[Callable[[Exception, int], None]] = None


P = ParamSpec("P")
T = TypeVar("T")


def with_retry(
    config: Optional[RetryConfig] = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for adding retry logic to async functions.

    Error Recovery Strategy (from specification):
    1. Wait 2 seconds (maybe page still loading)
    2. Scroll element into view
    3. Retry click
    4. If still fails after 3 attempts:
       - Log detailed error (screenshot + stack trace)
       - Ask Planner to generate alternative approach
       - If no alternative: Escalate to user

    Args:
        config: RetryConfig with retry parameters

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except config.retry_on as e:
                    last_exception = e

                    if attempt >= config.max_retries:
                        logger.error(
                            f"Failed after {config.max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    if config.exponential_backoff:
                        delay = min(
                            config.base_delay * (2 ** attempt),
                            config.max_delay
                        )
                    else:
                        delay = config.base_delay

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Call optional retry callback
                    if config.on_retry:
                        config.on_retry(e, attempt)

                    await asyncio.sleep(delay)

                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error: {e}")
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop completed without result")

        return wrapper
    return decorator


class ErrorRecoveryStrategy:
    """Strategies for recovering from different error types"""

    @staticmethod
    async def handle_element_not_found(
        error: ActionError,
        page: Any,  # Playwright Page
        original_action: dict
    ) -> dict:
        """
        Handle element not found errors.

        Strategy:
        1. Wait 2 seconds for page load
        2. Scroll down to check if element is below viewport
        3. Re-capture screenshot
        4. Return action for Vision-Actor to re-analyze
        """
        from src.browser.tab_manager import TabManager

        logger.info("Attempting recovery: element not found")

        # Wait for potential lazy loading
        await asyncio.sleep(2.0)

        # Try scrolling down
        await page.evaluate("window.scrollBy(0, 300)")
        await asyncio.sleep(0.5)

        return {
            "recovery_type": "rescroll_and_retry",
            "message": "Element not found. Scrolled down and waiting for retry.",
            "should_recapture_screenshot": True,
            "original_action": original_action
        }

    @staticmethod
    async def handle_timeout(
        error: BrowserError,
        page: Any,  # Playwright Page
    ) -> dict:
        """
        Handle page timeout errors.

        Strategy:
        1. Check if page is still responding
        2. Try refreshing if stuck
        3. Mark as needing manual intervention if refresh fails
        """
        logger.info("Attempting recovery: page timeout")

        try:
            # Check if page is responsive
            await page.evaluate("1 + 1", timeout=5000)

            # Page is responsive, just slow
            return {
                "recovery_type": "page_slow",
                "message": "Page is slow but responsive. Continuing...",
                "should_wait": True,
                "wait_time": 5.0
            }
        except Exception:
            # Page is unresponsive
            try:
                await page.reload(timeout=30000)
                return {
                    "recovery_type": "page_reloaded",
                    "message": "Page was unresponsive. Reloaded successfully.",
                    "should_recapture_screenshot": True
                }
            except Exception as reload_error:
                return {
                    "recovery_type": "manual_intervention",
                    "message": f"Page unresponsive and reload failed: {reload_error}",
                    "requires_user_action": True
                }

    @staticmethod
    async def handle_captcha_detected(
        screenshot_path: str,
        page: Any
    ) -> dict:
        """
        Handle CAPTCHA detection.

        Strategy:
        1. Pause agent
        2. Display screenshot in dashboard
        3. Request human to solve
        4. Resume after human completes
        """
        logger.warning("CAPTCHA detected - requesting human assistance")

        return {
            "recovery_type": "captcha_detected",
            "message": "CAPTCHA detected. Please solve it manually.",
            "screenshot_path": screenshot_path,
            "requires_user_action": True,
            "pause_agent": True
        }

    @staticmethod
    def build_error_report(
        error: AgentError,
        step_info: Optional[dict] = None,
        screenshot_path: Optional[str] = None
    ) -> dict:
        """
        Build a detailed error report for logging and debugging.
        """
        return {
            "error_type": type(error).__name__,
            "message": error.message,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "context": error.context,
            "step_info": step_info,
            "screenshot_path": screenshot_path
        }


# Convenience function for handling VRAM issues
async def handle_vram_overflow(
    current_usage_gb: float,
    max_vram_gb: float
) -> dict:
    """
    Handle out of VRAM situations.

    Strategy:
    1. Prune KV cache
    2. Close unused tabs
    3. If still fails, restart Ollama
    """
    import subprocess

    logger.warning(
        f"VRAM usage ({current_usage_gb:.1f}GB) approaching limit ({max_vram_gb:.1f}GB)"
    )

    actions_taken = []

    # Try to clear CUDA cache if using PyTorch directly
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            actions_taken.append("cleared_cuda_cache")
    except ImportError:
        pass

    # Suggest KV cache pruning (Ollama handles this internally)
    actions_taken.append("recommend_kv_prune")

    return {
        "recovery_type": "vram_management",
        "actions_taken": actions_taken,
        "message": "Attempted to free VRAM. May need to close tabs or restart Ollama.",
        "current_usage_gb": current_usage_gb,
        "max_vram_gb": max_vram_gb
    }
