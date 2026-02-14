"""Streamlit dashboard for the autonomous browser agent"""

import asyncio
import sys
import platform
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Fix Windows asyncio subprocess issue
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import config


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
    .stApp { max-width: 1400px; margin: 0 auto; }
    .status-running { color: #00aa00; font-weight: bold; }
    .status-stopped { color: #aa0000; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.agent_running = False
        st.session_state.current_task = None
        st.session_state.steps = []
        st.session_state.thoughts = deque(maxlen=50)
        st.session_state.logs = deque(maxlen=100)
        st.session_state.hitl_pending = None
        st.session_state.hitl_response = None
        st.session_state.tabs = {}
        st.session_state.task_result = None
        st.session_state.orchestrator_holder = {"orchestrator": None}
        st.session_state.agent_thread = None


# Thread-safe queues
import queue

class _QueueManager:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.thought_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.step_queue = queue.Queue()
        self.tabs_queue = queue.Queue()
        self.hitl_request_queue = queue.Queue()
        self.hitl_response_event = threading.Event()
        self.hitl_response_value = None
        self.stop_flag = threading.Event()

@st.cache_resource
def get_queue_manager() -> _QueueManager:
    return _QueueManager()

_queues = get_queue_manager()
_log_queue = _queues.log_queue
_thought_queue = _queues.thought_queue
_result_queue = _queues.result_queue
_step_queue = _queues.step_queue
_tabs_queue = _queues.tabs_queue
_hitl_request_queue = _queues.hitl_request_queue
_hitl_response_event = _queues.hitl_response_event
_stop_flag = _queues.stop_flag


def add_thought(stage: str, content: str):
    _thought_queue.put({
        "stage": stage,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })

def add_log(level: str, message: str):
    _log_queue.put({
        "level": level,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    })


def process_queues():
    processed = False

    while not _log_queue.empty():
        try:
            log = _log_queue.get_nowait()
            if "logs" in st.session_state:
                st.session_state.logs.append(log)
            processed = True
        except queue.Empty:
            break

    while not _thought_queue.empty():
        try:
            thought = _thought_queue.get_nowait()
            if "thoughts" in st.session_state:
                st.session_state.thoughts.append(thought)
            processed = True
        except queue.Empty:
            break

    while not _step_queue.empty():
        try:
            step = _step_queue.get_nowait()
            if "steps" in st.session_state:
                step_num = step.get("step", 0)
                while len(st.session_state.steps) < step_num:
                    st.session_state.steps.append({})
                if step_num > 0:
                    st.session_state.steps[step_num - 1] = step
            processed = True
        except queue.Empty:
            break

    while not _tabs_queue.empty():
        try:
            tabs = _tabs_queue.get_nowait()
            if "tabs" in st.session_state:
                st.session_state.tabs = tabs
            processed = True
        except queue.Empty:
            break

    while not _hitl_request_queue.empty():
        try:
            hitl = _hitl_request_queue.get_nowait()
            st.session_state.hitl_pending = hitl
            processed = True
        except queue.Empty:
            break

    while not _result_queue.empty():
        try:
            result = _result_queue.get_nowait()
            st.session_state.task_result = result
            st.session_state.agent_running = False
            processed = True
        except queue.Empty:
            break

    return processed


def status_callback(status: Dict[str, Any]):
    stage = status.get("stage", "")
    add_thought(stage, str(status))

    if "current_step" in status:
        _step_queue.put(status["current_step"])

    if "tabs" in status and status["tabs"]:
        _tabs_queue.put(status["tabs"])


def hitl_callback(action: Dict[str, Any], screenshot: bytes) -> bool:
    _hitl_request_queue.put({"action": action})
    _hitl_response_event.clear()
    _queues.hitl_response_value = None
    if _hitl_response_event.wait(timeout=300):
        return _queues.hitl_response_value == "approve"
    return False


def set_hitl_response(response: str):
    _queues.hitl_response_value = response
    _hitl_response_event.set()


def run_agent_async(goal: str, orchestrator_holder: dict):
    _stop_flag.clear()
    loop = None
    try:
        from src.main import AgentOrchestrator

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        orchestrator = AgentOrchestrator(
            hitl_callback=hitl_callback,
            status_callback=status_callback,
        )
        orchestrator_holder["orchestrator"] = orchestrator

        add_log("INFO", f"Agent initialized (provider: {config.llm_provider})")

        if _stop_flag.is_set():
            _result_queue.put({"success": False, "error": "Stopped by user"})
            return

        result = loop.run_until_complete(orchestrator.run_task(goal))
        _result_queue.put(result)
        add_log("INFO", f"Task: {'SUCCESS' if result.get('success') else 'FAILED'}")

    except Exception as e:
        import traceback
        add_log("ERROR", f"Agent error: {e}")
        add_log("DEBUG", traceback.format_exc())
        _result_queue.put({"success": False, "error": str(e)})
    finally:
        orch = orchestrator_holder.get("orchestrator")
        if orch and loop:
            try:
                loop.run_until_complete(orch.stop())
            except Exception:
                pass
        if loop:
            loop.close()


def start_agent(goal: str):
    _stop_flag.clear()
    st.session_state.agent_running = True
    st.session_state.current_task = goal
    st.session_state.steps = []
    st.session_state.task_result = None
    st.session_state.tabs = {}
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
    _stop_flag.set()
    holder = getattr(st.session_state, 'orchestrator_holder', None)
    if holder and holder.get("orchestrator"):
        holder["orchestrator"]._stop_requested = True
    st.session_state.agent_running = False


def main():
    init_session_state()
    process_queues()

    # Header
    st.title("🤖 Autonomous Browser Agent")
    provider_info = f"**LLM**: {config.llm_provider.upper()}"
    if config.is_claude:
        provider_info += f" ({config.claude_model})"
    else:
        provider_info += f" ({config.model_name})"

    if st.session_state.agent_running:
        st.markdown(f"<span class='status-running'>● Running</span> | {provider_info}", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-stopped'>○ Stopped</span> | {provider_info}", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        if not st.session_state.agent_running:
            st.subheader("New Task")
            goal = st.text_area("What should the agent do?", height=100,
                              placeholder="Search for artificial intelligence on Wikipedia")
            if st.button("🚀 Start Task", type="primary"):
                if goal.strip():
                    start_agent(goal.strip())
                    st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏹ Stop", disabled=not st.session_state.agent_running):
                stop_agent()
                st.rerun()
        with col2:
            if st.button("🔄 Reset"):
                _stop_flag.set()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.divider()

        # Info
        st.header("Agent Info")
        st.write(f"**Provider**: {config.llm_provider}")
        st.write(f"**Model**: {config.claude_model if config.is_claude else config.model_name}")
        st.write(f"**Browser**: {'Headless' if config.headless else 'Visible'}")
        st.write(f"**Tabs**: {len(st.session_state.tabs)}")

        st.divider()

        # Debug
        with st.expander("Debug Info", expanded=False):
            st.write(f"Running: {st.session_state.agent_running}")
            st.write(f"Task: {st.session_state.current_task}")
            st.write(f"Steps: {len(st.session_state.steps)}")
            st.write(f"Thoughts: {len(st.session_state.thoughts)}")
            st.write(f"Result: {st.session_state.task_result}")

    # Main content
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Execution Steps")

        if st.session_state.current_task:
            st.info(f"**Task**: {st.session_state.current_task}")

        if st.session_state.steps:
            for i, step in enumerate(st.session_state.steps):
                if not step:
                    continue
                status = step.get("status", "pending")
                icon = {"completed": "✅", "failed": "❌", "in_progress": "⏳", "pending": "⬜"}.get(status, "⬜")
                desc = step.get("description", f"Step {i+1}")
                action = step.get("action", "")
                st.markdown(f"{icon} **Step {i+1}** [{action}]: {desc}")
        else:
            if st.session_state.agent_running:
                st.info("Planning task...")
            else:
                st.info("No steps yet. Start a task.")

    with col2:
        st.subheader("Agent Thoughts")

        if st.session_state.thoughts:
            for thought in reversed(list(st.session_state.thoughts)[-10:]):
                stage = thought.get("stage", "").upper()
                content = thought.get("content", "")[:200]
                st.markdown(f"**{stage}**: {content}")
        else:
            st.info("No thoughts yet.")

    # HITL
    if st.session_state.hitl_pending:
        st.divider()
        st.warning("⚠️ Action requires approval!")
        st.json(st.session_state.hitl_pending.get("action", {}))
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve"):
                set_hitl_response("approve")
                st.session_state.hitl_pending = None
                st.rerun()
        with col_b:
            if st.button("❌ Reject"):
                set_hitl_response("reject")
                st.session_state.hitl_pending = None
                st.rerun()

    # Result
    if st.session_state.task_result:
        st.divider()
        result = st.session_state.task_result
        if result.get("success"):
            st.success(f"✅ Task Completed! Steps: {result.get('steps_completed')}/{result.get('total_steps')}")
            if result.get("result"):
                with st.expander("View Result"):
                    st.json(result["result"])
        else:
            st.error(f"❌ Failed: {result.get('error', 'Unknown')}")
            st.write(f"Steps: {result.get('steps_completed', 0)}/{result.get('total_steps', 0)}")

    # Auto-refresh
    if st.session_state.agent_running:
        thread = getattr(st.session_state, 'agent_thread', None)
        if thread and not thread.is_alive():
            process_queues()
            st.session_state.agent_running = False
            st.rerun()
        time.sleep(0.3)
        st.rerun()


def run():
    main()

if __name__ == "__main__":
    main()
