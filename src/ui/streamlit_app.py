"""Streamlit dashboard for the autonomous browser agent"""

import asyncio
import io
import sys
import platform
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

# Fix Windows asyncio subprocess issue
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import config
from src.ui.components import (
    render_screenshot,
    render_step_history,
    render_hitl_approval,
    render_metrics,
    render_thought_process,
    render_task_input,
    render_controls,
    render_tab_overview,
)


# Page configuration
st.set_page_config(
    page_title="Autonomous Browser Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .status-running {
        color: #00aa00;
        font-weight: bold;
    }
    .status-stopped {
        color: #aa0000;
    }
    .step-success {
        color: #00aa00;
    }
    .step-failed {
        color: #aa0000;
    }
    .step-pending {
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.agent_running = False
        st.session_state.current_task = None
        st.session_state.current_screenshot = None
        st.session_state.steps = []
        st.session_state.thoughts = deque(maxlen=50)
        st.session_state.logs = deque(maxlen=100)
        st.session_state.hitl_pending = None
        st.session_state.hitl_response = None
        st.session_state.tabs = {}
        st.session_state.metrics = {
            "vram_usage_gb": 0.0,
            "tokens_per_sec": 0.0,
            "step_latency_ms": 0.0,
            "active_tabs": 0,
        }
        st.session_state.task_result = None
        st.session_state.orchestrator_holder = {"orchestrator": None}
        st.session_state.agent_thread = None


# Thread-safe queues for cross-thread communication
import queue
_log_queue = queue.Queue()
_thought_queue = queue.Queue()
_result_queue = queue.Queue()


def add_thought(stage: str, content: str):
    """Add a thought to the thought log (thread-safe)"""
    _thought_queue.put({
        "stage": stage,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })


def add_log(level: str, message: str):
    """Add a log entry (thread-safe)"""
    _log_queue.put({
        "level": level,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    })


def process_queues():
    """Process queued updates from background threads"""
    # Process logs
    while not _log_queue.empty():
        try:
            log = _log_queue.get_nowait()
            if "logs" in st.session_state:
                st.session_state.logs.append(log)
        except queue.Empty:
            break

    # Process thoughts
    while not _thought_queue.empty():
        try:
            thought = _thought_queue.get_nowait()
            if "thoughts" in st.session_state:
                st.session_state.thoughts.append(thought)
        except queue.Empty:
            break

    # Process results
    while not _result_queue.empty():
        try:
            result = _result_queue.get_nowait()
            st.session_state.task_result = result
            st.session_state.agent_running = False
        except queue.Empty:
            break


def status_callback(status: Dict[str, Any]):
    """Callback for agent status updates"""
    stage = status.get("stage", "")
    add_thought(stage, str(status))

    if "current_step" in status:
        step = status["current_step"]
        # Update steps list
        step_num = step.get("step", 0)
        while len(st.session_state.steps) < step_num:
            st.session_state.steps.append({})
        if step_num > 0:
            st.session_state.steps[step_num - 1] = step


def hitl_callback(action: Dict[str, Any], screenshot: bytes) -> bool:
    """Callback for HITL approval requests"""
    st.session_state.hitl_pending = {
        "action": action,
        "screenshot": screenshot,
    }

    # Wait for user response
    timeout = 300  # 5 minute timeout
    start = time.time()
    while st.session_state.hitl_response is None:
        time.sleep(0.5)
        if time.time() - start > timeout:
            st.session_state.hitl_pending = None
            return False

    response = st.session_state.hitl_response
    st.session_state.hitl_pending = None
    st.session_state.hitl_response = None

    return response == "approve"


def run_agent_async(goal: str, orchestrator_holder: dict):
    """Run agent in background thread"""
    loop = None
    try:
        from src.main import AgentOrchestrator

        # Create event loop for this thread with Windows fix
        if platform.system() == "Windows":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create orchestrator
        orchestrator = AgentOrchestrator(
            hitl_callback=hitl_callback,
            status_callback=status_callback,
        )
        orchestrator_holder["orchestrator"] = orchestrator

        add_log("INFO", "Agent orchestrator initialized")

        # Run task
        result = loop.run_until_complete(orchestrator.run_task(goal))

        _result_queue.put(result)
        add_log("INFO", f"Task completed: {'SUCCESS' if result.get('success') else 'FAILED'}")

    except Exception as e:
        import traceback
        error_msg = f"Agent error: {str(e)}"
        add_log("ERROR", error_msg)
        add_log("DEBUG", traceback.format_exc())
        _result_queue.put({"success": False, "error": str(e)})

    finally:
        orchestrator = orchestrator_holder.get("orchestrator")
        if orchestrator and loop:
            try:
                loop.run_until_complete(orchestrator.stop())
            except Exception:
                pass
        if loop:
            loop.close()


def start_agent(goal: str):
    """Start the agent in a background thread"""
    st.session_state.agent_running = True
    st.session_state.current_task = goal
    st.session_state.steps = []
    st.session_state.task_result = None
    st.session_state.orchestrator_holder = {"orchestrator": None}

    add_log("INFO", f"Starting task: {goal}")

    thread = threading.Thread(
        target=run_agent_async,
        args=(goal, st.session_state.orchestrator_holder),
        daemon=True
    )
    thread.start()
    st.session_state.agent_thread = thread


def stop_agent():
    """Request agent stop"""
    holder = getattr(st.session_state, 'orchestrator_holder', None)
    if holder and holder.get("orchestrator"):
        holder["orchestrator"]._stop_requested = True
    add_log("WARNING", "Stop requested")


def main():
    """Main Streamlit app"""
    init_session_state()

    # Process any queued updates from background threads
    process_queues()

    # Header
    st.title("🤖 Autonomous Browser Agent")

    if st.session_state.agent_running:
        st.markdown(
            "<span class='status-running'>● Agent Running</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='status-stopped'>○ Agent Stopped</span>",
            unsafe_allow_html=True,
        )

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        # Task input
        if not st.session_state.agent_running:
            goal = render_task_input()
            if goal:
                start_agent(goal)
                st.rerun()

        # Control buttons
        control = render_controls(st.session_state.agent_running)
        if control == "stop":
            stop_agent()
        elif control == "reset":
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        # Metrics
        st.header("Performance")
        render_metrics(**st.session_state.metrics)

        st.divider()

        # Tab overview
        render_tab_overview(st.session_state.tabs)

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        # Screenshot panel
        st.subheader("Live View")

        screenshot_container = st.empty()

        if st.session_state.current_screenshot:
            with screenshot_container:
                st.image(
                    Image.open(io.BytesIO(st.session_state.current_screenshot)),
                    caption="Current Page",
                    use_container_width=True,
                )
        else:
            with screenshot_container:
                st.info("No screenshot available. Start a task to begin.")

        # Current task
        if st.session_state.current_task:
            st.info(f"**Current Task**: {st.session_state.current_task}")

    with col2:
        # Thought process
        render_thought_process(list(st.session_state.thoughts))

        st.divider()

        # Step history
        render_step_history(
            st.session_state.steps,
            current_step_index=len(st.session_state.steps) - 1 if st.session_state.steps else -1,
        )

    # HITL Approval Modal
    if st.session_state.hitl_pending:
        st.divider()
        pending = st.session_state.hitl_pending

        response = render_hitl_approval(
            action=pending["action"],
            screenshot_bytes=pending["screenshot"],
            reason="This action requires your approval before proceeding.",
        )

        if response:
            st.session_state.hitl_response = response
            st.rerun()

    # Task result
    if st.session_state.task_result:
        st.divider()
        result = st.session_state.task_result

        if result.get("success"):
            st.success("✅ Task Completed Successfully!")

            if result.get("result"):
                with st.expander("View Result"):
                    st.json(result["result"])
        else:
            st.error(f"❌ Task Failed: {result.get('error', 'Unknown error')}")

        st.write(f"**Steps completed**: {result.get('steps_completed', 0)}/{result.get('total_steps', 0)}")

        if result.get("duration_seconds"):
            st.write(f"**Duration**: {result['duration_seconds']:.1f} seconds")

    # Auto-refresh while running
    if st.session_state.agent_running:
        time.sleep(0.5)
        st.rerun()


def run():
    """Entry point for streamlit"""
    main()


if __name__ == "__main__":
    main()
