"""Verifier agent for action success verification"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from PIL import Image
import io

from src.agents.base_agent import BaseAgent, LLMConfig
from src.utils.vision_utils import image_to_base64, calculate_image_diff


@dataclass
class VerificationResult:
    """Result of action verification"""
    success: bool
    confidence: float
    reasoning: str
    retry_strategy: Optional[str] = None
    url_changed: bool = False
    screenshot_diff_percent: float = 0.0
    expected_element_found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "retry_strategy": self.retry_strategy,
            "url_changed": self.url_changed,
            "screenshot_diff_percent": self.screenshot_diff_percent,
            "expected_element_found": self.expected_element_found,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        """Create from dictionary"""
        return cls(
            success=data.get("success", False),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            retry_strategy=data.get("retry_strategy"),
            url_changed=data.get("url_changed", False),
            screenshot_diff_percent=data.get("screenshot_diff_percent", 0.0),
            expected_element_found=data.get("expected_element_found", False),
        )


# Verifier prompt template from specification
VERIFIER_PROMPT_TEMPLATE = """You are a Verification agent. You must determine if a browser action succeeded.

INTENDED ACTION: {action_json}
BEFORE SCREENSHOT: [base64 image 1]
AFTER SCREENSHOT: [base64 image 2]
URL BEFORE: {url_before}
URL AFTER: {url_after}
VERIFICATION CRITERIA: {verification_criteria}

Respond ONLY with JSON:
{{
  "success": true|false,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<explain what changed (or didn't change)>",
  "retry_strategy": "<if failed, suggest what to try next>"
}}

EXAMPLES:

1. SUCCESS CASE:
INTENDED: Click "Add to Cart" button
BEFORE: Product page with "Add to Cart" button visible
AFTER: Page shows "Item added to your cart" message
URL BEFORE: amazon.com/product/12345
URL AFTER: amazon.com/cart
OUTPUT:
{{
  "success": true,
  "confidence": 0.95,
  "reasoning": "URL changed to cart page and I see confirmation message",
  "retry_strategy": null
}}

2. FAILURE CASE:
INTENDED: Click "Sign In" button
BEFORE: Homepage with "Sign In" button in top-right
AFTER: Same homepage, no visible change
URL BEFORE: example.com
URL AFTER: example.com
OUTPUT:
{{
  "success": false,
  "confidence": 0.9,
  "reasoning": "No visual change detected. Button may be non-functional or behind a modal.",
  "retry_strategy": "Scroll down to check if a login modal appeared below viewport, or try clicking a different Sign In link."
}}

NOW VERIFY THE ACTION:"""


class VerifierAgent(BaseAgent):
    """
    Verifier agent for checking if actions succeeded.

    Uses both visual comparison and LLM analysis to verify:
    - Before/after screenshot comparison (pixel diff)
    - URL change detection
    - Expected element appearance
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(llm_config=llm_config, agent_name="verifier")

    async def verify_action(
        self,
        before_screenshot: bytes,
        after_screenshot: bytes,
        action: Dict[str, Any],
        url_before: str,
        url_after: str,
        verification_criteria: str,
    ) -> VerificationResult:
        """
        Verify if an action succeeded by comparing before/after states.

        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            action: The action that was executed
            url_before: URL before action
            url_after: URL after action
            verification_criteria: Criteria for success

        Returns:
            VerificationResult with success status and details
        """
        logger.info(f"[Verifier] Verifying action: {action.get('action_type', 'unknown')}")

        # Quick checks first (no LLM needed)
        url_changed = url_before != url_after

        # Calculate visual difference
        before_img = Image.open(io.BytesIO(before_screenshot))
        after_img = Image.open(io.BytesIO(after_screenshot))
        diff_percent = calculate_image_diff(before_img, after_img)

        logger.debug(
            f"[Verifier] URL changed: {url_changed}, "
            f"Visual diff: {diff_percent:.1f}%"
        )

        # For navigation actions, URL change is often sufficient
        if action.get("action_type") == "navigate" and url_changed:
            expected_url = action.get("target_element", {}).get("description", "")
            if not expected_url:
                expected_url = action.get("target", "")

            # Check if URL matches expected
            url_matches = expected_url.lower() in url_after.lower()

            return VerificationResult(
                success=url_matches,
                confidence=0.95 if url_matches else 0.5,
                reasoning=f"URL changed to {url_after}" + (
                    ", matches expected" if url_matches else ", but doesn't match expected"
                ),
                url_changed=True,
                screenshot_diff_percent=diff_percent,
            )

        # For actions with minimal visual change, use quick heuristics
        if diff_percent < 1.0 and not url_changed:
            # Very little changed - likely failed
            return VerificationResult(
                success=False,
                confidence=0.8,
                reasoning="Minimal visual change and URL unchanged",
                retry_strategy="Wait longer for page to update, or try clicking a different element",
                url_changed=False,
                screenshot_diff_percent=diff_percent,
            )

        # Use LLM for complex verification
        return await self._verify_with_llm(
            before_screenshot=before_screenshot,
            after_screenshot=after_screenshot,
            action=action,
            url_before=url_before,
            url_after=url_after,
            verification_criteria=verification_criteria,
            url_changed=url_changed,
            diff_percent=diff_percent,
        )

    async def _verify_with_llm(
        self,
        before_screenshot: bytes,
        after_screenshot: bytes,
        action: Dict[str, Any],
        url_before: str,
        url_after: str,
        verification_criteria: str,
        url_changed: bool,
        diff_percent: float,
    ) -> VerificationResult:
        """Use LLM for complex verification"""
        logger.debug("[Verifier] Using LLM for verification")

        # Convert screenshots to base64
        before_b64 = image_to_base64(before_screenshot)
        after_b64 = image_to_base64(after_screenshot)

        # Build prompt
        import json
        action_json = json.dumps(action, indent=2)

        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            action_json=action_json,
            url_before=url_before,
            url_after=url_after,
            verification_criteria=verification_criteria,
        )

        # Generate verification with both screenshots
        result_data = await self.generate_json(
            prompt=prompt,
            images=[before_b64, after_b64],
            temperature=0.1,
        )

        result = VerificationResult(
            success=result_data.get("success", False),
            confidence=result_data.get("confidence", 0.0),
            reasoning=result_data.get("reasoning", ""),
            retry_strategy=result_data.get("retry_strategy"),
            url_changed=url_changed,
            screenshot_diff_percent=diff_percent,
        )

        logger.info(
            f"[Verifier] Result: {'SUCCESS' if result.success else 'FAILED'} "
            f"(confidence: {result.confidence:.2f})"
        )

        return result

    async def verify_navigation(
        self,
        url_before: str,
        url_after: str,
        expected_url_pattern: str,
    ) -> VerificationResult:
        """
        Quick verification for navigation actions.

        Args:
            url_before: URL before navigation
            url_after: URL after navigation
            expected_url_pattern: Pattern or substring expected in new URL

        Returns:
            VerificationResult
        """
        import re

        url_changed = url_before != url_after

        # Check if URL matches expected pattern
        try:
            pattern_match = bool(re.search(expected_url_pattern, url_after, re.IGNORECASE))
        except re.error:
            # Treat as substring match if not valid regex
            pattern_match = expected_url_pattern.lower() in url_after.lower()

        success = url_changed and pattern_match

        return VerificationResult(
            success=success,
            confidence=0.95 if success else 0.7,
            reasoning=f"URL {'changed' if url_changed else 'unchanged'}, "
                      f"pattern {'matched' if pattern_match else 'not matched'}",
            retry_strategy="Check for redirects or try different URL" if not success else None,
            url_changed=url_changed,
        )

    async def verify_text_appears(
        self,
        after_screenshot: bytes,
        expected_text: str,
    ) -> VerificationResult:
        """
        Verify that specific text appears in the screenshot.

        Args:
            after_screenshot: Screenshot after action
            expected_text: Text that should appear

        Returns:
            VerificationResult
        """
        logger.debug(f"[Verifier] Checking for text: {expected_text}")

        screenshot_b64 = image_to_base64(after_screenshot)

        prompt = f"""Look at this screenshot and determine if the following text appears:
"{expected_text}"

Respond with JSON:
{{
  "found": true|false,
  "confidence": 0.0 to 1.0,
  "location": "where on the page you found it (or null)",
  "reasoning": "explain your finding"
}}"""

        result = await self.generate_json(
            prompt=prompt,
            images=[screenshot_b64],
            temperature=0.1,
        )

        found = result.get("found", False)

        return VerificationResult(
            success=found,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            expected_element_found=found,
        )

    async def verify_element_exists(
        self,
        after_screenshot: bytes,
        element_description: str,
    ) -> VerificationResult:
        """
        Verify that a specific element exists in the screenshot.

        Args:
            after_screenshot: Screenshot after action
            element_description: Description of element to find

        Returns:
            VerificationResult
        """
        logger.debug(f"[Verifier] Checking for element: {element_description}")

        screenshot_b64 = image_to_base64(after_screenshot)

        prompt = f"""Look at this screenshot and determine if this element exists:
"{element_description}"

Respond with JSON:
{{
  "found": true|false,
  "confidence": 0.0 to 1.0,
  "bbox": [x1, y1, x2, y2] or null,
  "reasoning": "explain your finding"
}}"""

        result = await self.generate_json(
            prompt=prompt,
            images=[screenshot_b64],
            temperature=0.1,
        )

        found = result.get("found", False)

        return VerificationResult(
            success=found,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            expected_element_found=found,
        )

    def should_retry(
        self,
        result: VerificationResult,
        attempt_number: int,
        max_retries: int = 3,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if we should retry based on verification result.

        Args:
            result: The verification result
            attempt_number: Current attempt (1-indexed)
            max_retries: Maximum allowed retries

        Returns:
            Tuple of (should_retry, reason)
        """
        if result.success:
            return False, None

        if attempt_number >= max_retries:
            return False, f"Max retries ({max_retries}) reached"

        if not result.retry_strategy:
            return False, "No retry strategy suggested"

        return True, result.retry_strategy
