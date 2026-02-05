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


# Planner prompt template from specification
PLANNER_PROMPT_TEMPLATE = """You are a Task Planner for a browser automation agent.

USER GOAL: {user_goal}

Break this goal into a sequence of browser actions. Respond ONLY with a JSON array of steps.

Each step MUST have this format:
{{
  "step": <number>,
  "action": "navigate|search|click|fill|extract|synthesize",
  "description": "<what to do>",
  "target": "<URL or query or element description>",
  "tab": <tab_id (0-9)>,
  "verification_criteria": "<how to confirm success>"
}}

RULES:
1. Use multiple tabs for parallel tasks (e.g., compare 3 sites → use tabs 0, 1, 2)
2. Add a "synthesize" step at the end to merge results
3. Keep steps atomic (one clear action per step)
4. Add verification criteria for every step

EXAMPLE:
USER GOAL: "Compare iPhone 15 Pro prices on Amazon and B&H"
OUTPUT:
[
  {{"step": 1, "action": "navigate", "target": "amazon.com", "tab": 0, "description": "Open Amazon homepage", "verification_criteria": "URL contains amazon.com"}},
  {{"step": 2, "action": "search", "target": "iPhone 15 Pro", "tab": 0, "description": "Search for product", "verification_criteria": "Search results page loads"}},
  {{"step": 3, "action": "extract", "target": "first result price", "tab": 0, "description": "Get price from top result", "verification_criteria": "Price stored in memory"}},
  {{"step": 4, "action": "navigate", "target": "bhphotovideo.com", "tab": 1, "description": "Open B&H homepage", "verification_criteria": "URL contains bhphotovideo.com"}},
  {{"step": 5, "action": "search", "target": "iPhone 15 Pro", "tab": 1, "description": "Search on B&H", "verification_criteria": "Search results page loads"}},
  {{"step": 6, "action": "extract", "target": "first result price", "tab": 1, "description": "Get price from B&H", "verification_criteria": "Price stored in memory"}},
  {{"step": 7, "action": "synthesize", "target": "compare prices from tabs 0 and 1", "tab": -1, "description": "Build comparison table", "verification_criteria": "Table generated"}}
]

NOW CREATE A PLAN FOR: {user_goal}"""


class PlannerAgent(BaseAgent):
    """
    Planner agent for decomposing tasks into steps.

    Runs ONCE per task (text-only, fast) → 1-2 seconds
    Uses Qwen2.5-VL in text-only mode for efficiency
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(llm_config=llm_config, agent_name="planner")

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

        if context:
            prompt = f"CONTEXT:\n{context}\n\n{prompt}"

        # Generate plan (no images needed for planning)
        steps_data = await self.generate_json(
            prompt=prompt,
            temperature=0.3,  # Low temperature for consistent planning
        )

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
