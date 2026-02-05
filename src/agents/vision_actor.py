"""Vision-Action agent for screenshot analysis and action generation"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.agents.base_agent import BaseAgent, LLMConfig
from src.utils.config import config
from src.utils.vision_utils import image_to_base64


@dataclass
class TargetElement:
    """Represents a UI element identified by the vision model"""
    description: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) normalized 0-1000
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "description": self.description,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetElement":
        """Create from dictionary"""
        bbox = data.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, list):
            bbox = tuple(bbox)
        return cls(
            description=data.get("description", ""),
            bbox=bbox,
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class ActionOutput:
    """Output from the Vision-Action agent"""
    action_type: str  # click, fill, scroll, extract, wait
    target_element: Optional[TargetElement] = None
    value: Optional[str] = None  # Text to type for fill actions
    reasoning: str = ""
    confidence: float = 0.0
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_type": self.action_type,
            "target_element": self.target_element.to_dict() if self.target_element else None,
            "value": self.value,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionOutput":
        """Create from dictionary"""
        target_data = data.get("target_element")
        target = TargetElement.from_dict(target_data) if target_data else None

        return cls(
            action_type=data.get("action_type", ""),
            target_element=target,
            value=data.get("value"),
            reasoning=data.get("reasoning", ""),
            confidence=target.confidence if target else 0.0,
        )

    @property
    def is_valid(self) -> bool:
        """Check if action is valid and actionable"""
        if not self.action_type:
            return False

        # Extract doesn't need coordinates
        if self.action_type == "extract":
            return True

        # Wait and scroll may not need target
        if self.action_type in ("wait", "scroll"):
            return True

        # Click and fill need target with bbox
        if self.action_type in ("click", "fill", "hover"):
            if not self.target_element:
                return False
            if not self.target_element.bbox:
                return False
            if self.target_element.confidence < config.action_confidence_threshold:
                return False

        return True


# Vision-Action prompt template from specification
VISION_ACTION_PROMPT_TEMPLATE = """You are a Vision-Action agent. You see a screenshot and must output ONE precise browser action.

CURRENT TASK: {step_description}
WEBPAGE URL: {current_url}
SCREENSHOT: [base64 image attached]

Analyze the screenshot and respond ONLY with JSON in this EXACT format:
{{
  "action_type": "click|fill|scroll|extract|wait",
  "target_element": {{
    "description": "<what element you see>",
    "bbox": [x1, y1, x2, y2],
    "confidence": <0.0 to 1.0>
  }},
  "value": "<text to type (only for 'fill' action)>",
  "reasoning": "<why you chose this element>"
}}

IMPORTANT RULES:
1. Bounding box coordinates are normalized 0-1000 (not pixels!)
2. x1,y1 = top-left corner, x2,y2 = bottom-right corner
3. Only respond with ONE action (not a list)
4. If you cannot find the element, set confidence to 0.0 and explain in reasoning
5. For "extract" actions, describe what data to extract (e.g., "price text in red")

EXAMPLE:
TASK: "Click the search button"
SCREENSHOT: [shows Google homepage with search button at center]
OUTPUT:
{{
  "action_type": "click",
  "target_element": {{
    "description": "Blue 'Google Search' button with white text",
    "bbox": [425, 380, 575, 420],
    "confidence": 0.98
  }},
  "value": null,
  "reasoning": "I see the primary search button centered below the search bar. It is the most prominent call-to-action on the page."
}}

NOW ANALYZE THE SCREENSHOT AND RESPOND:"""


class VisionActorAgent(BaseAgent):
    """
    Vision-Action agent for analyzing screenshots and generating actions.

    Runs PER STEP (image + text, slower) → 2-4 seconds
    Uses Qwen2.5-VL in vision mode with screenshot input
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(llm_config=llm_config, agent_name="vision_actor")

    async def get_action(
        self,
        screenshot_bytes: bytes,
        step_description: str,
        current_url: str,
        additional_context: Optional[str] = None,
    ) -> ActionOutput:
        """
        Analyze screenshot and generate action for the current step.

        Args:
            screenshot_bytes: PNG screenshot bytes
            step_description: Description of what to do
            current_url: Current page URL
            additional_context: Optional context (e.g., previous failed attempts)

        Returns:
            ActionOutput with action type and coordinates
        """
        logger.info(f"[VisionActor] Analyzing screenshot for: {step_description}")

        # Convert screenshot to base64
        screenshot_b64 = image_to_base64(screenshot_bytes)

        # Build prompt
        prompt = VISION_ACTION_PROMPT_TEMPLATE.format(
            step_description=step_description,
            current_url=current_url,
        )

        if additional_context:
            prompt = f"{additional_context}\n\n{prompt}"

        # Generate action with vision
        action_data = await self.generate_json(
            prompt=prompt,
            images=[screenshot_b64],
            temperature=0.2,  # Very low for precise coordinates
        )

        # Parse action
        action = ActionOutput.from_dict(action_data)

        # Validate confidence
        if action.target_element:
            confidence = action.target_element.confidence
            if confidence < config.action_confidence_threshold:
                logger.warning(
                    f"[VisionActor] Low confidence action: {confidence:.2f} < "
                    f"{config.action_confidence_threshold}"
                )

        logger.info(
            f"[VisionActor] Generated action: {action.action_type} "
            f"(confidence: {action.confidence:.2f})"
        )
        logger.debug(f"[VisionActor] Reasoning: {action.reasoning}")

        return action

    async def reanalyze_after_failure(
        self,
        screenshot_bytes: bytes,
        step_description: str,
        current_url: str,
        previous_action: ActionOutput,
        failure_reason: str,
    ) -> ActionOutput:
        """
        Re-analyze screenshot after a failed action.

        Asks the model to try a different approach.

        Args:
            screenshot_bytes: Current screenshot
            step_description: Original step description
            current_url: Current page URL
            previous_action: The action that failed
            failure_reason: Why it failed

        Returns:
            New ActionOutput with alternative approach
        """
        logger.info(f"[VisionActor] Re-analyzing after failure: {failure_reason}")

        context = f"""PREVIOUS ATTEMPT FAILED:
Action: {previous_action.action_type}
Target: {previous_action.target_element.description if previous_action.target_element else 'unknown'}
Failure reason: {failure_reason}

Please try a DIFFERENT element or approach. Look for alternative buttons, links, or input fields."""

        return await self.get_action(
            screenshot_bytes=screenshot_bytes,
            step_description=step_description,
            current_url=current_url,
            additional_context=context,
        )

    async def detect_page_elements(
        self,
        screenshot_bytes: bytes,
        element_type: str = "all",
    ) -> List[TargetElement]:
        """
        Detect all elements of a specific type on the page.

        Useful for debugging and visualization.

        Args:
            screenshot_bytes: PNG screenshot
            element_type: Type of elements to find (buttons, links, inputs, all)

        Returns:
            List of detected elements
        """
        logger.info(f"[VisionActor] Detecting {element_type} elements")

        screenshot_b64 = image_to_base64(screenshot_bytes)

        prompt = f"""Analyze this screenshot and identify all {element_type} elements.

For each element, provide:
- Description of the element
- Bounding box coordinates (normalized 0-1000)
- Confidence score

Respond with a JSON array:
[
  {{"description": "...", "bbox": [x1, y1, x2, y2], "confidence": 0.95}},
  ...
]

Only include clearly visible, interactive elements."""

        elements_data = await self.generate_json(
            prompt=prompt,
            images=[screenshot_b64],
            temperature=0.3,
        )

        if isinstance(elements_data, list):
            elements = [TargetElement.from_dict(e) for e in elements_data]
        else:
            elements = []

        logger.info(f"[VisionActor] Detected {len(elements)} elements")
        return elements

    async def detect_captcha(
        self,
        screenshot_bytes: bytes,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if the page contains a CAPTCHA.

        Args:
            screenshot_bytes: PNG screenshot

        Returns:
            Tuple of (is_captcha, captcha_type)
        """
        logger.debug("[VisionActor] Checking for CAPTCHA")

        screenshot_b64 = image_to_base64(screenshot_bytes)

        prompt = """Analyze this screenshot for CAPTCHA or bot verification.

Look for:
- reCAPTCHA checkboxes or challenges
- hCaptcha puzzles
- Text-based CAPTCHAs
- "Verify you are human" messages
- Cloudflare verification pages

Respond with JSON:
{
  "has_captcha": true/false,
  "captcha_type": "recaptcha|hcaptcha|text|cloudflare|other|none",
  "description": "brief description of what you see"
}"""

        result = await self.generate_json(
            prompt=prompt,
            images=[screenshot_b64],
            temperature=0.1,
        )

        has_captcha = result.get("has_captcha", False)
        captcha_type = result.get("captcha_type", "none")

        if has_captcha:
            logger.warning(f"[VisionActor] CAPTCHA detected: {captcha_type}")

        return has_captcha, captcha_type if has_captcha else None

    async def extract_text_content(
        self,
        screenshot_bytes: bytes,
        target_description: str,
    ) -> Dict[str, Any]:
        """
        Extract specific text content from screenshot.

        Args:
            screenshot_bytes: PNG screenshot
            target_description: What text to extract

        Returns:
            Extracted text and metadata
        """
        logger.info(f"[VisionActor] Extracting text: {target_description}")

        screenshot_b64 = image_to_base64(screenshot_bytes)

        prompt = f"""Extract the following from this screenshot:
{target_description}

Respond with JSON containing the extracted information:
{{
  "found": true/false,
  "content": "the extracted text or data",
  "location": "where on the page this was found",
  "confidence": 0.0 to 1.0
}}"""

        result = await self.generate_json(
            prompt=prompt,
            images=[screenshot_b64],
            temperature=0.1,
        )

        return result
