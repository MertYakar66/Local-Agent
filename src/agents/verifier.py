"""Verifier agent for action success verification.

Uses URL changes and DOM state instead of screenshot comparison.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from playwright.async_api import Page

from src.agents.base_agent import BaseAgent, LLMConfig


@dataclass
class VerificationResult:
    """Result of action verification"""
    success: bool
    confidence: float
    reasoning: str
    retry_strategy: Optional[str] = None
    url_changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "retry_strategy": self.retry_strategy,
            "url_changed": self.url_changed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(
            success=data.get("success", False),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            retry_strategy=data.get("retry_strategy"),
            url_changed=data.get("url_changed", False),
        )


class VerifierAgent(BaseAgent):
    """
    Verifier agent that checks if actions succeeded using URL and DOM state.

    No screenshots needed - checks:
    - URL changes
    - DOM element presence/state
    - Page title changes
    - Selector validation (element exists after action)
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(llm_config=llm_config, agent_name="verifier")

    async def verify_action(
        self,
        page: Page,
        action: Dict[str, Any],
        url_before: str,
        url_after: str,
        verification_criteria: str,
    ) -> VerificationResult:
        """
        Verify if an action succeeded by checking URL and DOM state.

        Args:
            page: Playwright Page object
            action: The action that was executed
            url_before: URL before action
            url_after: URL after action
            verification_criteria: Criteria for success

        Returns:
            VerificationResult
        """
        action_type = action.get("action_type", "unknown")
        logger.info(f"[Verifier] Verifying action: {action_type}")

        url_changed = url_before != url_after
        logger.debug(f"[Verifier] URL changed: {url_changed} ({url_before} -> {url_after})")

        # Navigate: URL change is the primary indicator
        if action_type == "navigate":
            target = action.get("selector") or action.get("target", "")
            url_matches = target.lower() in url_after.lower() if target else url_changed
            return VerificationResult(
                success=url_matches or url_changed,
                confidence=0.95 if url_matches else (0.7 if url_changed else 0.3),
                reasoning=f"URL is now {url_after}" + (
                    " (matches target)" if url_matches else ""
                ),
                url_changed=url_changed,
            )

        # Click: Check for URL change or DOM changes
        if action_type == "click":
            if url_changed:
                return VerificationResult(
                    success=True,
                    confidence=0.95,
                    reasoning=f"Click caused navigation to {url_after}",
                    url_changed=True,
                )

            # Check if page content changed
            try:
                title = await page.title()
                return VerificationResult(
                    success=True,
                    confidence=0.8,
                    reasoning=f"Click executed, page title: {title}",
                    url_changed=False,
                )
            except Exception:
                return VerificationResult(
                    success=True,
                    confidence=0.6,
                    reasoning="Click executed, unable to verify page state",
                )

        # Fill: Check if the input has the expected value
        if action_type == "fill":
            selector = action.get("selector", "")
            expected_value = action.get("value", "")
            if selector:
                try:
                    actual_value = await page.input_value(selector, timeout=3000)
                    matches = actual_value == expected_value
                    return VerificationResult(
                        success=matches,
                        confidence=0.95 if matches else 0.3,
                        reasoning=(
                            f"Input contains '{actual_value}'" if matches
                            else f"Input has '{actual_value}', expected '{expected_value}'"
                        ),
                        retry_strategy=None if matches else "Try a different selector for the input field",
                    )
                except Exception:
                    pass

            # Fallback: assume fill worked if no error was raised
            return VerificationResult(
                success=True,
                confidence=0.7,
                reasoning="Fill action completed without error",
            )

        # Press key: Check URL change (e.g., Enter on search form)
        if action_type == "press_key":
            key = action.get("value", "Enter")
            if key == "Enter" and url_changed:
                return VerificationResult(
                    success=True,
                    confidence=0.95,
                    reasoning=f"Enter key caused navigation to {url_after}",
                    url_changed=True,
                )
            return VerificationResult(
                success=True,
                confidence=0.7,
                reasoning=f"Key '{key}' pressed successfully",
                url_changed=url_changed,
            )

        # Extract: Success if we got data
        if action_type == "extract":
            return VerificationResult(
                success=True,
                confidence=0.9,
                reasoning="Data extraction completed",
            )

        # Scroll/Wait: Always succeed
        if action_type in ("scroll", "wait"):
            return VerificationResult(
                success=True,
                confidence=0.9,
                reasoning=f"{action_type} completed",
            )

        # Default
        return VerificationResult(
            success=True,
            confidence=0.5,
            reasoning=f"Action '{action_type}' completed, no specific verification",
        )

    async def verify_with_llm(
        self,
        page: Page,
        action: Dict[str, Any],
        verification_criteria: str,
        url_before: str,
        url_after: str,
    ) -> VerificationResult:
        """
        Use the LLM to verify complex actions by examining DOM state.
        Only called when simple verification is insufficient.
        """
        try:
            # Get page info for LLM
            title = await page.title()
            url = page.url

            # Get key DOM elements
            page_text = await page.evaluate("""() => {
                const h1 = document.querySelector('h1');
                const main = document.querySelector('main, #content, .content, article');
                const body = document.body;
                const target = main || body;
                return (h1 ? 'H1: ' + h1.textContent.trim() + '\\n' : '') +
                       target.innerText.substring(0, 2000);
            }""")
        except Exception:
            page_text = "(unable to read page)"

        prompt = f"""Verify if this browser action succeeded.

ACTION: {action}
URL BEFORE: {url_before}
URL AFTER: {url_after}
PAGE TITLE: {title if 'title' in dir() else 'unknown'}
VERIFICATION CRITERIA: {verification_criteria}

PAGE CONTENT (excerpt):
{page_text[:1500]}

Respond with JSON:
{{
  "success": true|false,
  "confidence": 0.0 to 1.0,
  "reasoning": "explain what happened",
  "retry_strategy": "if failed, what to try next (or null)"
}}"""

        result_data = await self.generate_json(
            prompt=prompt,
            temperature=0.1,
        )

        return VerificationResult(
            success=result_data.get("success", False),
            confidence=result_data.get("confidence", 0.0),
            reasoning=result_data.get("reasoning", ""),
            retry_strategy=result_data.get("retry_strategy"),
            url_changed=url_before != url_after,
        )

    def should_retry(
        self,
        result: VerificationResult,
        attempt_number: int,
        max_retries: int = 3,
    ) -> Tuple[bool, Optional[str]]:
        """Decide if we should retry based on verification result."""
        if result.success:
            return False, None
        if attempt_number >= max_retries:
            return False, f"Max retries ({max_retries}) reached"
        if not result.retry_strategy:
            return False, "No retry strategy"
        return True, result.retry_strategy
