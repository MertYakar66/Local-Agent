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
# Using Streamlit's cache_resource to ensure persistence across reruns
import queue
import threading

class _QueueManager:
    """Queue manager with all queues for cross-thread communication"""
    def __init__(self):
        self.log_queue = queue.Queue()
        self.thought_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.step_queue = queue.Queue()
        self.screenshot_queue = queue.Queue()
        self.tabs_queue = queue.Queue()
        self.hitl_request_queue = queue.Queue()
        self.hitl_response_event = threading.Event()
        self.hitl_response_value = None
        self.stop_flag = threading.Event()
        print("[DEBUG] QueueManager created!")

@st.cache_resource
def get_queue_manager() -> _QueueManager:
    """Get the singleton queue manager - cached across all Streamlit reruns"""
    return _QueueManager()

# Get singleton instance using Streamlit's caching
_queues = get_queue_manager()
_log_queue = _queues.log_queue
_thought_queue = _queues.thought_queue
_result_queue = _queues.result_queue
_step_queue = _queues.step_queue
_screenshot_queue = _queues.screenshot_queue
_tabs_queue = _queues.tabs_queue
_hitl_request_queue = _queues.hitl_request_queue
_hitl_response_event = _queues.hitl_response_event
_stop_flag = _queues.stop_flag


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
    processed_any = False

    # Process logs
    while not _log_queue.empty():
        try:
            log = _log_queue.get_nowait()
            if "logs" in st.session_state:
                st.session_state.logs.append(log)
            processed_any = True
        except queue.Empty:
            break

    # Process thoughts
    thought_count = 0
    while not _thought_queue.empty():
        try:
            thought = _thought_queue.get_nowait()
            if "thoughts" in st.session_state:
                st.session_state.thoughts.append(thought)
            thought_count += 1
        except queue.Empty:
            break
    if thought_count:
        print(f"[DEBUG] Processed {thought_count} thoughts")
        processed_any = True

    # Process steps
    step_count = 0
    while not _step_queue.empty():
        try:
            step_update = _step_queue.get_nowait()
            if "steps" in st.session_state:
                step_num = step_update.get("step", 0)
                while len(st.session_state.steps) < step_num:
                    st.session_state.steps.append({})
                if step_num > 0:
                    st.session_state.steps[step_num - 1] = step_update
                step_count += 1
        except queue.Empty:
            break
    if step_count:
        print(f"[DEBUG] Processed {step_count} steps, total steps now: {len(st.session_state.steps)}")
        processed_any = True

    # Process screenshots
    screenshot_count = 0
    while not _screenshot_queue.empty():
        try:
            screenshot = _screenshot_queue.get_nowait()
            st.session_state.current_screenshot = screenshot
            screenshot_count += 1
        except queue.Empty:
            break
    if screenshot_count:
        screenshot_size = len(st.session_state.current_screenshot) if st.session_state.current_screenshot else 0
        print(f"[DEBUG] Processed {screenshot_count} screenshots, final size: {screenshot_size} bytes")
        processed_any = True

    # Process tabs
    tabs_count = 0
    while not _tabs_queue.empty():
        try:
            tabs = _tabs_queue.get_nowait()
            if "tabs" in st.session_state:
                st.session_state.tabs = tabs
            tabs_count += 1
        except queue.Empty:
            break
    if tabs_count:
        tab_keys = list(st.session_state.tabs.keys()) if st.session_state.tabs else []
        print(f"[DEBUG] Processed {tabs_count} tab updates, tabs now: {tab_keys}")
        processed_any = True

    # Process HITL requests
    while not _hitl_request_queue.empty():
        try:
            hitl_request = _hitl_request_queue.get_nowait()
            st.session_state.hitl_pending = hitl_request
            processed_any = True
        except queue.Empty:
            break

    # Process results
    while not _result_queue.empty():
        try:
            result = _result_queue.get_nowait()
            st.session_state.task_result = result
            st.session_state.agent_running = False
            print(f"[DEBUG] Task result received: success={result.get('success')}")
            processed_any = True
        except queue.Empty:
            break

    return processed_any


def status_callback(status: Dict[str, Any]):
    """Callback for agent status updates (thread-safe)"""
    stage = status.get("stage", "")

    # Debug: Log callback invocations
    print(f"[DEBUG] status_callback called: stage={stage}, has_screenshot={bool(status.get('screenshot'))}, has_tabs={bool(status.get('tabs'))}")

    # Don't log screenshot bytes in thought
    status_for_thought = {k: v for k, v in status.items() if k != "screenshot"}
    add_thought(stage, str(status_for_thought))

    if "current_step" in status:
        step = status["current_step"]
        # Queue step update instead of direct access
        _step_queue.put(step)
        print(f"[DEBUG] Queued step: {step.get('step', 'N/A')} - {step.get('description', 'N/A')[:50]}")

    if "screenshot" in status and status["screenshot"]:
        # Queue screenshot update
        screenshot_data = status["screenshot"]
        _screenshot_queue.put(screenshot_data)
        print(f"[DEBUG] Queued screenshot: {len(screenshot_data)} bytes")

    if "tabs" in status and status["tabs"]:
        # Queue tabs update
        tabs_data = status["tabs"]
        _tabs_queue.put(tabs_data)
        print(f"[DEBUG] Queued tabs: {list(tabs_data.keys()) if isinstance(tabs_data, dict) else 'invalid'}")


def hitl_callback(action: Dict[str, Any], screenshot: bytes) -> bool:
    """Callback for HITL approval requests (thread-safe)"""
    # Queue the HITL request
    _hitl_request_queue.put({
        "action": action,
        "screenshot": screenshot,
    })

    # Clear the event and wait for response
    _hitl_response_event.clear()
    _queues.hitl_response_value = None

    # Wait for user response with timeout
    timeout = 300  # 5 minute timeout
    if _hitl_response_event.wait(timeout=timeout):
        return _queues.hitl_response_value == "approve"

    return False


def set_hitl_response(response: str):
    """Set HITL response from main thread"""
    _queues.hitl_response_value = response
    _hitl_response_event.set()


def run_agent_async(goal: str, orchestrator_holder: dict):
    """Run agent in background thread"""
    global _stop_flag
    _stop_flag.clear()  # Clear stop flag at start

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

        # Check stop flag before running
        if _stop_flag.is_set():
            add_log("WARNING", "Stop requested before task started")
            _result_queue.put({"success": False, "error": "Stopped by user"})
            return

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
    global _stop_flag
    _stop_flag.clear()  # Clear stop flag before starting

    st.session_state.agent_running = True
    st.session_state.current_task = goal
    st.session_state.steps = []
    st.session_state.task_result = None
    st.session_state.tabs = {}
    st.session_state.current_screenshot = None
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
    global _stop_flag
    _stop_flag.set()  # Set global stop flag immediately

    holder = getattr(st.session_state, 'orchestrator_holder', None)
    if holder and holder.get("orchestrator"):
        holder["orchestrator"]._stop_requested = True
    add_log("WARNING", "Stop requested")
    st.session_state.agent_running = False


def main():
    """Main Streamlit app"""
    init_session_state()

    # Process any queued updates from background threads
    # Call it twice to ensure all data is processed
    processed = process_queues()
    if processed:
        print(f"[DEBUG] main(): Processed queue updates")

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
            st.rerun()  # Force UI refresh after stop
        elif control == "reset":
            _stop_flag.set()  # Ensure any running agent stops
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        # Metrics - update active_tabs from actual tabs
        metrics = st.session_state.metrics.copy()
        metrics["active_tabs"] = len(st.session_state.tabs) if st.session_state.tabs else 0

        st.header("Performance")
        render_metrics(**metrics)

        st.divider()

        # Tab overview
        st.header("Browser Tabs")
        if st.session_state.tabs:
            render_tab_overview(st.session_state.tabs)
        else:
            st.info("No browser tabs open yet")

        st.divider()

        # Debug panel
        with st.expander("🐛 Debug Info", expanded=False):
            st.write(f"**Agent Running**: {st.session_state.agent_running}")
            st.write(f"**Current Task**: {st.session_state.current_task}")
            st.write(f"**Steps Count**: {len(st.session_state.steps)}")
            st.write(f"**Thoughts Count**: {len(st.session_state.thoughts)}")
            st.write(f"**Tabs**: {list(st.session_state.tabs.keys()) if st.session_state.tabs else 'None'}")
            has_screenshot = st.session_state.current_screenshot is not None
            screenshot_size = len(st.session_state.current_screenshot) if has_screenshot else 0
            st.write(f"**Screenshot**: {'Yes' if has_screenshot else 'No'} ({screenshot_size} bytes)")
            st.write(f"**Task Result**: {st.session_state.task_result}")

            # Queue status
            st.write("---")
            st.write(f"**Log Queue Size**: {_log_queue.qsize()}")
            st.write(f"**Thought Queue Size**: {_thought_queue.qsize()}")
            st.write(f"**Step Queue Size**: {_step_queue.qsize()}")
            st.write(f"**Screenshot Queue Size**: {_screenshot_queue.qsize()}")
            st.write(f"**Tabs Queue Size**: {_tabs_queue.qsize()}")
            st.write(f"**Result Queue Size**: {_result_queue.qsize()}")

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        # Screenshot panel
        st.subheader("Live View")

        screenshot_container = st.empty()

        # Try to get screenshot from session state first
        screenshot_bytes = st.session_state.current_screenshot

        # Fallback: try to load latest screenshot from disk
        if not screenshot_bytes and st.session_state.agent_running:
            try:
                # Use correct path from config (logs/screenshots or config.screenshots_dir)
                screenshot_dir = config.screenshots_dir
                if screenshot_dir.exists():
                    screenshots = sorted(screenshot_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if screenshots:
                        screenshot_bytes = screenshots[0].read_bytes()
                        print(f"[DEBUG] Loaded fallback screenshot from disk: {screenshots[0]}")
            except Exception as e:
                print(f"[DEBUG] Failed to load fallback screenshot: {e}")

        if screenshot_bytes:
            with screenshot_container:
                st.image(
                    Image.open(io.BytesIO(screenshot_bytes)),
                    caption=f"Current Page ({len(screenshot_bytes):,} bytes)",
                    use_container_width=True,
                )
        else:
            with screenshot_container:
                if st.session_state.agent_running:
                    st.warning("⏳ Waiting for screenshot from agent...")
                else:
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
            set_hitl_response(response)
            st.session_state.hitl_pending = None
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
        # Check if agent thread is still alive
        agent_thread = getattr(st.session_state, 'agent_thread', None)
        if agent_thread and not agent_thread.is_alive():
            # Thread finished, process remaining queues
            print("[DEBUG] Agent thread finished, processing remaining queues...")
            process_queues()
            st.session_state.agent_running = False
            st.rerun()  # Final rerun to show results

        time.sleep(0.2)  # Shorter interval for more responsive updates
        st.rerun()


def run():
    """Entry point for streamlit"""
    main()


if __name__ == "__main__":
    main()
