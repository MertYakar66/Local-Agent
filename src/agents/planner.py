"""Planner agent for task decomposition"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent, LLMConfig
from src.utils.config import config


class ActionKind(Enum):
    """Types of actions the planner can generate"""
    NAVIGATE = "navigate"
    SEARCH = "search"
    CLICK = "click"
    FILL = "fill"
    EXTRACT = "extract"
    SYNTHESIZE = "synthesize"
    SCROLL = "scroll"
    WAIT = "wait"


@dataclass
class TaskStep:
    """A single step in the task plan"""
    step: int
    action: str
    description: str
    target: str
    tab: int
    verification_criteria: str
    value: Optional[str] = None  # Text to fill for fill actions
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step": self.step,
            "action": self.action,
            "description": self.description,
            "target": self.target,
            "tab": self.tab,
            "verification_criteria": self.verification_criteria,
            "value": self.value,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStep":
        """Create from dictionary"""
        return cls(
            step=data.get("step", 0),
            action=data.get("action", ""),
            description=data.get("description", ""),
            target=data.get("target", ""),
            tab=data.get("tab", 0),
            verification_criteria=data.get("verification_criteria", ""),
            value=data.get("value"),
            status=data.get("status", "pending"),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class TaskPlan:
    """A complete task plan with multiple steps"""
    user_goal: str
    steps: List[TaskStep] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed

    @property
    def current_step(self) -> Optional[TaskStep]:
        """Get current step"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed"""
        return all(s.status == "completed" for s in self.steps)

    @property
    def tabs_used(self) -> List[int]:
        """Get list of unique tab IDs used in this plan"""
        return sorted(set(s.tab for s in self.steps if s.tab >= 0))

    def mark_step_completed(self, step_index: int, result: Optional[Dict[str, Any]] = None):
        """Mark a step as completed"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].status = "completed"
            self.steps[step_index].result = result
            self.current_step_index = step_index + 1

    def mark_step_failed(self, step_index: int, error: str):
        """Mark a step as failed"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].status = "failed"
            self.steps[step_index].error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_goal": self.user_goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status,
        }


# Planner prompt template - concise to encourage direct JSON output
PLANNER_PROMPT_TEMPLATE = """Output a JSON array for this task. No explanation, just JSON.

Task: {user_goal}

Format: [{{"step":N,"action":"TYPE","target":"TARGET","tab":0,"description":"DESC","value":"TEXT_FOR_FILL_OR_NULL","verification_criteria":"CHECK"}}]
Actions: navigate (target=URL), fill (value=text to type), click (target=button/link)

Example for "Search cats on Wikipedia":
[{{"step":1,"action":"navigate","target":"https://www.wikipedia.org","tab":0,"description":"Open Wikipedia","value":null,"verification_criteria":"wikipedia in URL"}},{{"step":2,"action":"fill","target":"search box","tab":0,"description":"Type search","value":"cats","verification_criteria":"text entered"}},{{"step":3,"action":"click","target":"search button","tab":0,"description":"Search","value":null,"verification_criteria":"results load"}}]

JSON for "{user_goal}":
["""


class PlannerAgent(BaseAgent):
    """
    Planner agent for decomposing tasks into steps.

    Runs ONCE per task (text-only, fast) → 1-2 seconds
    Uses Qwen2.5-VL in text-only mode for efficiency
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(llm_config=llm_config, agent_name="planner")

    def _parse_plan_json(self, text: str) -> List[Dict[str, Any]]:
        """Parse JSON plan from model output, handling common issues"""
        import json
        import re

        # Log raw input for debugging
        logger.debug(f"[Planner] Parsing JSON from: {text[:300]}...")

        # Clean up the text
        text = text.strip()

        # Remove common prefixes that models add
        prefixes_to_remove = [
            "```json", "```", "Here is", "Here's", "The JSON", "JSON:",
            "Output:", "Result:", "Answer:", "Plan:"
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Find the JSON array in the text
        # Look for [ ... ] pattern
        start_idx = text.find('[')
        if start_idx == -1:
            # No array found, maybe it's wrapped differently
            logger.warning(f"[Planner] No JSON array found in: {text[:200]}")
            raise ValueError("No JSON array found in model output")

        # Find the matching closing bracket
        bracket_count = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break

        if end_idx == -1:
            # No matching bracket, try to complete the JSON
            text = text[start_idx:] + ']'
        else:
            text = text[start_idx:end_idx]

        # Fix double bracket issue: [[ -> [
        while text.startswith('[['):
            text = text[1:]

        # Remove ... or other continuation markers
        text = text.replace('...', '')

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                logger.debug(f"[Planner] Successfully parsed {len(data)} steps")
                return data
        except json.JSONDecodeError:
            pass

        # Try to fix common JSON issues
        # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r',\s*}', '}', text)

        # Fix unquoted keys (some models do this)
        text = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1"\2":', text)

        try:
            data = json.loads(text)
            if isinstance(data, list):
                logger.debug(f"[Planner] Successfully parsed {len(data)} steps after fixes")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"[Planner] JSON parse error: {e}")
            logger.error(f"[Planner] Failed text: {text[:500]}")

        raise ValueError(f"Could not parse plan JSON from model output")

    async def decompose_task(
        self,
        user_goal: str,
        context: Optional[str] = None,
    ) -> TaskPlan:
        """
        Decompose a user goal into a step-by-step plan.

        Args:
            user_goal: The high-level task description
            context: Optional additional context (e.g., past successful plans)

        Returns:
            TaskPlan with list of TaskStep objects
        """
        logger.info(f"[Planner] Decomposing task: {user_goal}")

        # Build prompt
        prompt = PLANNER_PROMPT_TEMPLATE.format(user_goal=user_goal)

        # Don't add context - it makes the model too verbose
        # The prompt is already optimized for direct JSON output

        # Generate plan (no images needed for planning)
        # Use json_mode to force JSON output and bypass thinking mode
        response = await self.generate(
            prompt=prompt,
            temperature=0.1,  # Very low temperature for consistent JSON
            max_tokens=1024,  # Allow enough tokens for complete JSON
            json_mode=True,  # Force JSON output format
        )

        # Log raw response for debugging
        logger.debug(f"[Planner] Raw response: {response[:500] if response else 'EMPTY'}...")

        # Handle empty response
        if not response or not response.strip():
            raise ValueError("Model returned empty response")

        # The prompt ends with "[" so prepend it to complete the array
        json_text = "[" + response.strip()

        # Try to parse JSON
        steps_data = self._parse_plan_json(json_text)

        # Parse steps
        if isinstance(steps_data, list):
            steps = [TaskStep.from_dict(s) for s in steps_data]
        elif isinstance(steps_data, dict) and "steps" in steps_data:
            steps = [TaskStep.from_dict(s) for s in steps_data["steps"]]
        else:
            raise ValueError(f"Unexpected plan format: {type(steps_data)}")

        # Create task plan
        plan = TaskPlan(
            user_goal=user_goal,
            steps=steps,
            status="pending",
        )

        logger.info(
            f"[Planner] Created plan with {len(steps)} steps "
            f"using tabs: {plan.tabs_used}"
        )

        return plan

    async def refine_plan(
        self,
        original_plan: TaskPlan,
        failed_step: TaskStep,
        error_context: str,
    ) -> TaskPlan:
        """
        Refine a plan after a step failure.

        Generates an alternative approach for the failed step.

        Args:
            original_plan: The original task plan
            failed_step: The step that failed
            error_context: Description of why it failed

        Returns:
            Updated TaskPlan with alternative approach
        """
        logger.info(f"[Planner] Refining plan after failure at step {failed_step.step}")

        prompt = f"""You are a Task Planner. A previous plan failed and needs adjustment.

ORIGINAL GOAL: {original_plan.user_goal}

ORIGINAL PLAN:
{[s.to_dict() for s in original_plan.steps]}

FAILED STEP: Step {failed_step.step} - "{failed_step.description}"
ERROR: {error_context}

Generate an ALTERNATIVE approach for this step. The new approach should:
1. Achieve the same goal but differently
2. Work around the error if possible
3. Keep the same step number

Respond ONLY with the replacement step as JSON:
{{
  "step": {failed_step.step},
  "action": "...",
  "description": "...",
  "target": "...",
  "tab": ...,
  "verification_criteria": "..."
}}"""

        alternative_step_data = await self.generate_json(
            prompt=prompt,
            temperature=0.5,  # Slightly higher for creative alternatives
        )

        # Create new step
        alternative_step = TaskStep.from_dict(alternative_step_data)

        # Replace failed step in plan
        new_steps = original_plan.steps.copy()
        step_index = failed_step.step - 1
        if 0 <= step_index < len(new_steps):
            new_steps[step_index] = alternative_step
            new_steps[step_index].status = "pending"

        new_plan = TaskPlan(
            user_goal=original_plan.user_goal,
            steps=new_steps,
            current_step_index=step_index,
            status="in_progress",
        )

        logger.info(
            f"[Planner] Generated alternative: {alternative_step.description}"
        )

        return new_plan

    async def create_synthesis_prompt(
        self,
        user_goal: str,
        extracted_data: List[Dict[str, Any]],
    ) -> str:
        """
        Create a synthesis prompt for combining extracted data.

        Args:
            user_goal: Original user goal
            extracted_data: Data extracted from various tabs

        Returns:
            Synthesis prompt for final output generation
        """
        prompt = f"""Synthesize the following extracted data to answer the user's goal.

USER GOAL: {user_goal}

EXTRACTED DATA:
{extracted_data}

Create a clear, structured summary that directly addresses the user's goal.
If comparing items, use a table format.
If summarizing, use bullet points.
Include all relevant details from the extracted data.

Respond with a well-formatted summary."""

        return prompt
