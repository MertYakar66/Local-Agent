"""DOM-based action agent - uses Playwright selectors instead of vision/screenshots.

The LLM generates Playwright selectors and the browser executes them directly.
No screenshots or coordinate mapping needed.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from playwright.async_api import Page

from src.agents.base_agent import BaseAgent, LLMConfig
from src.utils.config import config


@dataclass
class DOMAction:
    """A browser action using Playwright selectors"""
    action_type: str  # click, fill, press_key, scroll, select, wait, extract
    selector: Optional[str] = None  # Playwright selector
    value: Optional[str] = None  # Text to type or key to press
    reasoning: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "selector": self.selector,
            "value": self.value,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DOMAction":
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got: {type(data)}")
        return cls(
            action_type=data.get("action_type", ""),
            selector=data.get("selector"),
            value=data.get("value"),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 1.0),
        )

    @property
    def is_valid(self) -> bool:
        if not self.action_type:
            return False
        if self.action_type in ("wait", "scroll", "press_key"):
            return True
        if self.action_type in ("click", "fill", "select") and not self.selector:
            return False
        return True


DOM_ACTION_PROMPT = """You are a browser automation agent. Given the page's DOM content and current URL, generate a Playwright action to accomplish the task.

CURRENT TASK: {step_description}
CURRENT URL: {current_url}
{fill_value_instruction}

PAGE DOM (simplified):
{dom_content}

Respond with JSON only:
{{
  "action_type": "click|fill|press_key|scroll|select|wait|extract",
  "selector": "playwright selector string (e.g. 'input[name=search]', 'text=Submit', '#login-btn', 'role=button[name=Search]')",
  "value": "text to type for fill, key for press_key (Enter/Tab/Escape), or null",
  "reasoning": "why you chose this action and selector"
}}

SELECTOR TIPS:
- Use 'text=...' for visible text links/buttons
- Use 'role=button[name=...]' for buttons
- Use 'input[type=search]', 'input[name=...]', '#id', '.class' for inputs
- Use 'placeholder=...' for input fields with placeholder text
- Prefer specific selectors (id > name > role > text > css)

RESPOND WITH JSON ONLY:"""


class DOMActorAgent(BaseAgent):
    """
    DOM-based action agent.

    Instead of analyzing screenshots with a vision model, this agent:
    1. Reads the page's DOM structure
    2. Asks the LLM for the right Playwright selector
    3. Executes the action directly via Playwright

    Much faster and more reliable than vision-based approach.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(llm_config=llm_config, agent_name="dom_actor")

    async def get_dom_snapshot(self, page: Page, max_length: int = 8000) -> str:
        """Extract a simplified DOM snapshot from the page for LLM context."""
        try:
            # Get interactive elements and key page structure
            dom = await page.evaluate("""() => {
                function getSelector(el) {
                    if (el.id) return '#' + el.id;
                    if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                    if (el.type && el.tagName === 'INPUT') return 'input[type="' + el.type + '"]';
                    if (el.className && typeof el.className === 'string') {
                        const cls = el.className.trim().split(/\\s+/)[0];
                        if (cls) return el.tagName.toLowerCase() + '.' + cls;
                    }
                    return el.tagName.toLowerCase();
                }

                function getElements() {
                    const results = [];
                    // Interactive elements
                    const interactiveSelectors = 'a, button, input, select, textarea, [role="button"], [role="link"], [role="textbox"], [role="searchbox"], [onclick], [tabindex]';
                    const elements = document.querySelectorAll(interactiveSelectors);

                    for (const el of elements) {
                        if (!el.offsetParent && el.tagName !== 'BODY') continue; // hidden

                        const info = {
                            tag: el.tagName.toLowerCase(),
                            selector: getSelector(el),
                        };

                        // Text content (trimmed)
                        const text = (el.textContent || '').trim().substring(0, 80);
                        if (text) info.text = text;

                        // Attributes
                        if (el.type) info.type = el.type;
                        if (el.placeholder) info.placeholder = el.placeholder;
                        if (el.name) info.name = el.name;
                        if (el.href) info.href = el.href.substring(0, 100);
                        if (el.value) info.value = el.value.substring(0, 50);
                        if (el.getAttribute('role')) info.role = el.getAttribute('role');
                        if (el.getAttribute('aria-label')) info.ariaLabel = el.getAttribute('aria-label');

                        results.push(info);
                    }

                    // Also get page title and headings for context
                    const headings = [];
                    document.querySelectorAll('h1, h2, h3').forEach(h => {
                        const t = (h.textContent || '').trim().substring(0, 100);
                        if (t) headings.push({tag: h.tagName.toLowerCase(), text: t});
                    });

                    return {title: document.title, headings, elements: results};
                }

                return JSON.stringify(getElements());
            }""")

            # Truncate if too long
            if len(dom) > max_length:
                dom = dom[:max_length] + '...(truncated)'

            return dom

        except Exception as e:
            logger.warning(f"[DOMActorAgent] Failed to get DOM snapshot: {e}")
            return '{"title": "unknown", "headings": [], "elements": []}'

    async def get_action(
        self,
        page: Page,
        step_description: str,
        current_url: str,
        fill_value: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> DOMAction:
        """
        Analyze the page DOM and generate a Playwright action.

        Args:
            page: Playwright Page object
            step_description: What to do
            current_url: Current page URL
            fill_value: Value for fill actions
            additional_context: Extra context (e.g., retry info)

        Returns:
            DOMAction with selector and action details
        """
        logger.info(f"[DOMActorAgent] Getting action for: {step_description}")

        # Get DOM snapshot
        dom_content = await self.get_dom_snapshot(page)

        # Build fill instruction
        fill_instruction = ""
        if fill_value:
            fill_instruction = f'\nFILL WITH VALUE: "{fill_value}"'

        # Build prompt
        prompt = DOM_ACTION_PROMPT.format(
            step_description=step_description,
            current_url=current_url,
            fill_value_instruction=fill_instruction,
            dom_content=dom_content,
        )

        if additional_context:
            prompt = f"CONTEXT: {additional_context}\n\n{prompt}"

        # Ask LLM for action
        action_data = await self.generate_json(
            prompt=prompt,
            temperature=0.1,
        )

        action = DOMAction.from_dict(action_data)

        # Override value with fill_value if provided
        if fill_value and action.action_type == "fill":
            action.value = fill_value

        logger.info(
            f"[DOMActorAgent] Action: {action.action_type} "
            f"selector='{action.selector}' reasoning='{action.reasoning}'"
        )

        return action

    async def execute_action(
        self,
        page: Page,
        action: DOMAction,
    ) -> Dict[str, Any]:
        """
        Execute a DOMAction directly via Playwright.

        Returns:
            Result dict with success/error info
        """
        try:
            action_type = action.action_type
            selector = action.selector
            value = action.value

            if action_type == "click":
                await page.click(selector, timeout=10000)
                # Wait for potential navigation
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                return {"success": True, "action": "click", "selector": selector}

            elif action_type == "fill":
                if not value:
                    return {"success": False, "error": "No value for fill action"}
                await page.fill(selector, value, timeout=10000)
                return {"success": True, "action": "fill", "selector": selector}

            elif action_type == "press_key":
                key = value or "Enter"
                await page.keyboard.press(key)
                # Wait for potential navigation after Enter
                if key == "Enter":
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass  # Page may not navigate
                return {"success": True, "action": "press_key", "key": key}

            elif action_type == "select":
                if not value:
                    return {"success": False, "error": "No value for select action"}
                await page.select_option(selector, value, timeout=10000)
                return {"success": True, "action": "select", "selector": selector}

            elif action_type == "scroll":
                direction = value or "down"
                amount = 500
                if direction == "down":
                    await page.evaluate(f"window.scrollBy(0, {amount})")
                elif direction == "up":
                    await page.evaluate(f"window.scrollBy(0, -{amount})")
                return {"success": True, "action": "scroll", "direction": direction}

            elif action_type == "wait":
                import asyncio
                await asyncio.sleep(2)
                return {"success": True, "action": "wait"}

            elif action_type == "extract":
                if selector:
                    text = await page.inner_text(selector, timeout=5000)
                else:
                    text = await page.inner_text("body")
                return {
                    "success": True,
                    "action": "extract",
                    "extracted_data": {"text": text[:2000]},
                }

            else:
                return {"success": False, "error": f"Unknown action: {action_type}"}

        except Exception as e:
            logger.error(f"[DOMActorAgent] Action failed: {e}")
            return {"success": False, "error": str(e)}
