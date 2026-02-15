"""Reusable UI components for the Streamlit dashboard"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

# Note: Streamlit imports are done inside functions to avoid import errors
# when this module is imported from non-Streamlit contexts


def render_screenshot(
    screenshot_bytes: Optional[bytes] = None,
    screenshot_path: Optional[Path] = None,
    caption: str = "Live Screenshot",
    width: Optional[int] = None,
) -> None:
    """
    Render a screenshot in the Streamlit app.

    Args:
        screenshot_bytes: Raw image bytes
        screenshot_path: Path to image file
        caption: Image caption
        width: Display width
    """
    import streamlit as st

    if screenshot_bytes:
        image = Image.open(io.BytesIO(screenshot_bytes))
    elif screenshot_path and screenshot_path.exists():
        image = Image.open(screenshot_path)
    else:
        st.info("No screenshot available")
        return

    st.image(image, caption=caption, width="stretch" if width is None else "content")


def render_step_history(
    steps: List[Dict[str, Any]],
    current_step_index: int = -1,
) -> None:
    """
    Render the step execution history.

    Args:
        steps: List of step dictionaries
        current_step_index: Index of currently executing step
    """
    import streamlit as st

    st.subheader("Step History")

    for i, step in enumerate(steps):
        status = step.get("status", "pending")

        # Status icon
        if status == "completed":
            icon = "✅"
        elif status == "failed":
            icon = "❌"
        elif i == current_step_index:
            icon = "🔄"
        else:
            icon = "⏳"

        # Step display
        step_num = step.get("step", i + 1)
        description = step.get("description", "Unknown step")

        with st.container():
            col1, col2 = st.columns([1, 10])

            with col1:
                st.write(icon)

            with col2:
                step_text = f"**Step {step_num}**: {description}"

                if status == "in_progress":
                    step_text += " *[IN PROGRESS]*"
                elif status == "failed":
                    error = step.get("error", "Unknown error")
                    step_text += f"\n  ⚠️ Error: {error}"

                st.markdown(step_text)


def render_hitl_approval(
    action: Dict[str, Any],
    screenshot_bytes: Optional[bytes] = None,
    reason: str = "Sensitive action detected",
) -> Optional[str]:
    """
    Render the human-in-the-loop approval modal.

    Args:
        action: Action dictionary to approve
        screenshot_bytes: Screenshot showing the target
        reason: Why approval is needed

    Returns:
        "approve", "reject", "modify", or None if no action taken
    """
    import streamlit as st

    st.warning(f"⚠️ APPROVAL REQUIRED: {reason}")

    # Show screenshot with target highlighted
    if screenshot_bytes:
        st.image(
            Image.open(io.BytesIO(screenshot_bytes)),
            caption="Screenshot of target element",
            width="stretch",
        )

    # Show action details
    st.subheader("Proposed Action")

    action_type = action.get("action_type", "unknown")
    target = action.get("target_element", {})
    description = target.get("description", "Unknown element")
    confidence = target.get("confidence", 0.0)

    st.write(f"**Action**: {action_type}")
    st.write(f"**Target**: {description}")
    st.write(f"**Confidence**: {confidence:.0%}")

    if action.get("value"):
        st.write(f"**Value**: {action['value']}")

    if action.get("reasoning"):
        st.write(f"**Reasoning**: {action['reasoning']}")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve", key="hitl_approve", type="primary"):
            return "approve"

    with col2:
        if st.button("❌ Reject", key="hitl_reject"):
            return "reject"

    with col3:
        if st.button("✏️ Modify", key="hitl_modify"):
            return "modify"

    return None


def render_metrics(
    vram_usage_gb: float = 0.0,
    tokens_per_sec: float = 0.0,
    step_latency_ms: float = 0.0,
    active_tabs: int = 0,
) -> None:
    """
    Render performance metrics.

    Args:
        vram_usage_gb: Current VRAM usage
        tokens_per_sec: Inference speed
        step_latency_ms: Average step latency
        active_tabs: Number of open browser tabs
    """
    import streamlit as st

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        vram_color = "normal" if vram_usage_gb < 14.0 else "inverse"
        st.metric(
            "VRAM Usage",
            f"{vram_usage_gb:.1f} GB",
            delta=None,
        )

    with col2:
        st.metric(
            "Tokens/sec",
            f"{tokens_per_sec:.1f}",
            delta=None,
        )

    with col3:
        st.metric(
            "Step Latency",
            f"{step_latency_ms:.0f} ms",
            delta=None,
        )

    with col4:
        st.metric(
            "Active Tabs",
            f"{active_tabs}",
            delta=None,
        )


def render_thought_process(
    thoughts: List[Dict[str, Any]],
    max_display: int = 10,
) -> None:
    """
    Render the agent's thought process log.

    Args:
        thoughts: List of thought dictionaries with stage, content, timestamp
        max_display: Maximum number of thoughts to show
    """
    import streamlit as st

    st.subheader("Agent Thoughts")

    # Show most recent thoughts
    recent = thoughts[-max_display:] if len(thoughts) > max_display else thoughts

    for thought in reversed(recent):
        stage = thought.get("stage", "")
        content = thought.get("content", "")
        timestamp = thought.get("timestamp", "")

        stage_emoji = {
            "planner": "📋",
            "dom_actor": "🎯",
            "execution": "🎯",
            "verification": "✓",
        }.get(stage, "💭")

        st.markdown(f"{stage_emoji} **{stage.upper()}**: {content}")


def render_task_input() -> Optional[str]:
    """
    Render the task input form.

    Returns:
        User's goal string if submitted, None otherwise
    """
    import streamlit as st

    st.subheader("New Task")

    with st.form("task_form"):
        goal = st.text_area(
            "What would you like the agent to do?",
            placeholder="e.g., Compare iPhone 15 prices on Amazon and B&H",
            height=100,
        )

        submitted = st.form_submit_button("🚀 Start Task", type="primary")

        if submitted and goal.strip():
            return goal.strip()

    return None


def render_controls(is_running: bool = False) -> Optional[str]:
    """
    Render agent control buttons.

    Args:
        is_running: Whether agent is currently running

    Returns:
        Control action ("stop", "retry", "reset") or None
    """
    import streamlit as st

    col1, col2, col3 = st.columns(3)

    with col1:
        if is_running:
            if st.button("⏹️ Stop Agent", type="secondary"):
                return "stop"
        else:
            st.button("⏹️ Stop Agent", disabled=True)

    with col2:
        if st.button("🔄 Retry Last Step", disabled=is_running):
            return "retry"

    with col3:
        if st.button("🗑️ Reset", disabled=is_running):
            return "reset"

    return None


def render_log_viewer(logs: List[Dict[str, Any]]) -> None:
    """
    Render the log viewer.

    Args:
        logs: List of log entries
    """
    import streamlit as st

    st.subheader("Logs")

    if not logs:
        st.info("No logs yet")
        return

    for log in reversed(logs[-20:]):
        level = log.get("level", "INFO")
        message = log.get("message", "")
        timestamp = log.get("timestamp", "")

        level_color = {
            "DEBUG": "gray",
            "INFO": "blue",
            "WARNING": "orange",
            "ERROR": "red",
        }.get(level, "white")

        st.markdown(
            f"<span style='color:{level_color}'>[{level}]</span> {message}",
            unsafe_allow_html=True,
        )


def render_tab_overview(tabs: Dict[int, Dict[str, Any]]) -> None:
    """
    Render overview of all browser tabs.

    Args:
        tabs: Dictionary of tab states
    """
    import streamlit as st

    if not tabs:
        st.info("No tabs open")
        return

    for tab_id, state in tabs.items():
        status = state.get("status", "unknown")
        url = state.get("current_url", "about:blank")
        title = state.get("title", "Untitled")

        status_icon = {
            "ready": "🟢",
            "loading": "🟡",
            "error": "🔴",
            "closed": "⚫",
        }.get(status, "⚪")

        with st.expander(f"Tab {tab_id} {status_icon} - {title[:30]}..."):
            st.write(f"**URL**: {url}")
            st.write(f"**Status**: {status}")
            if state.get("last_action"):
                st.write(f"**Last Action**: {state['last_action']}")
