"""Structured logging for the autonomous browser agent"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.utils.config import config


@dataclass
class StepLog:
    """Log entry for a single step execution"""
    task_id: str
    step_number: int
    stage: str  # planner, vision_action, execution, verification
    tab_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    # Input
    screenshot_before_path: Optional[str] = None
    prompt: Optional[str] = None
    viewport_size: Optional[Dict[str, int]] = None

    # Output
    action_type: Optional[str] = None
    bbox: Optional[List[int]] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    # Execution
    playwright_result: str = "pending"
    page_url_after: Optional[str] = None
    duration_ms: float = 0.0

    # Verification
    url_changed: bool = False
    expected_element_found: bool = False
    screenshot_diff_percent: float = 0.0
    verification_status: str = "pending"

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "step_number": self.step_number,
            "stage": self.stage,
            "tab_id": self.tab_id,
            "input": {
                "screenshot_path": self.screenshot_before_path,
                "prompt": self.prompt,
                "viewport_size": self.viewport_size,
            },
            "output": {
                "action_type": self.action_type,
                "bbox": self.bbox,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
            },
            "execution": {
                "playwright_result": self.playwright_result,
                "page_url_after": self.page_url_after,
                "duration_ms": self.duration_ms,
            },
            "verification": {
                "url_changed": self.url_changed,
                "expected_element_found": self.expected_element_found,
                "screenshot_diff_percent": self.screenshot_diff_percent,
                "status": self.verification_status,
            },
            "errors": self.errors,
        }


@dataclass
class TaskLog:
    """Log entry for a complete task"""
    task_id: str
    user_goal: str
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    steps: List[StepLog] = field(default_factory=list)
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    final_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "final_result": self.final_result,
            "error_message": self.error_message,
            "steps": [s.to_dict() for s in self.steps],
        }

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


class StructuredLogger:
    """
    Structured logging system with SQLite and JSON backup.

    Logging Specification:
    - Every action logged to SQLite + JSON files
    - Includes screenshot paths, action details, verification results
    - Human-readable and machine-parseable
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        db_name: str = "actions.db",
    ):
        self.log_dir = log_dir or Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tasks_dir = self.log_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.log_dir / db_name

        # Current task context
        self._current_task: Optional[TaskLog] = None

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    user_goal TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    total_steps INTEGER,
                    successful_steps INTEGER,
                    failed_steps INTEGER,
                    error_message TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    step_number INTEGER,
                    stage TEXT,
                    tab_id INTEGER,
                    timestamp TEXT,
                    action_type TEXT,
                    confidence REAL,
                    playwright_result TEXT,
                    duration_ms REAL,
                    verification_status TEXT,
                    errors TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id)
            """)

            conn.commit()

    def start_task(self, user_goal: str) -> str:
        """
        Start logging a new task.

        Args:
            user_goal: The user's goal description

        Returns:
            Generated task ID
        """
        task_id = str(uuid.uuid4())[:8]

        self._current_task = TaskLog(
            task_id=task_id,
            user_goal=user_goal,
            status="in_progress",
        )

        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks (task_id, user_goal, status, start_time, total_steps, successful_steps, failed_steps)
                VALUES (?, ?, ?, ?, 0, 0, 0)
            """, (task_id, user_goal, "in_progress", self._current_task.start_time.isoformat()))
            conn.commit()

        logger.info(f"Started task {task_id}: {user_goal}")
        return task_id

    def log_step(
        self,
        step_number: int,
        stage: str,
        tab_id: int,
        action_type: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        playwright_result: str = "pending",
        page_url_after: Optional[str] = None,
        duration_ms: float = 0.0,
        verification_status: str = "pending",
        screenshot_path: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> StepLog:
        """
        Log a step execution.

        Args:
            step_number: Step number in the plan
            stage: Current stage (planner, vision_action, etc.)
            tab_id: Browser tab ID
            ...other parameters

        Returns:
            Created StepLog
        """
        if not self._current_task:
            raise RuntimeError("No active task. Call start_task() first.")

        step_log = StepLog(
            task_id=self._current_task.task_id,
            step_number=step_number,
            stage=stage,
            tab_id=tab_id,
            action_type=action_type,
            bbox=bbox,
            confidence=confidence,
            reasoning=reasoning,
            playwright_result=playwright_result,
            page_url_after=page_url_after,
            duration_ms=duration_ms,
            verification_status=verification_status,
            screenshot_before_path=screenshot_path,
            errors=errors or [],
        )

        self._current_task.steps.append(step_log)
        self._current_task.total_steps += 1

        if verification_status == "success":
            self._current_task.successful_steps += 1
        elif verification_status == "failed":
            self._current_task.failed_steps += 1

        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO steps (task_id, step_number, stage, tab_id, timestamp, action_type, confidence, playwright_result, duration_ms, verification_status, errors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._current_task.task_id,
                step_number,
                stage,
                tab_id,
                step_log.timestamp.isoformat(),
                action_type,
                confidence,
                playwright_result,
                duration_ms,
                verification_status,
                json.dumps(errors or []),
            ))
            conn.commit()

        logger.debug(
            f"Logged step {step_number} ({stage}): {action_type or 'no action'} - {verification_status}"
        )

        return step_log

    def complete_task(
        self,
        success: bool,
        final_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> TaskLog:
        """
        Mark task as completed and save to disk.

        Args:
            success: Whether the task succeeded
            final_result: Optional final output data
            error_message: Optional error message if failed

        Returns:
            Completed TaskLog
        """
        if not self._current_task:
            raise RuntimeError("No active task to complete")

        self._current_task.end_time = datetime.now()
        self._current_task.status = "completed" if success else "failed"
        self._current_task.final_result = final_result
        self._current_task.error_message = error_message

        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET
                    status = ?,
                    end_time = ?,
                    total_steps = ?,
                    successful_steps = ?,
                    failed_steps = ?,
                    error_message = ?
                WHERE task_id = ?
            """, (
                self._current_task.status,
                self._current_task.end_time.isoformat(),
                self._current_task.total_steps,
                self._current_task.successful_steps,
                self._current_task.failed_steps,
                error_message,
                self._current_task.task_id,
            ))
            conn.commit()

        # Save JSON file
        json_path = self.tasks_dir / f"{self._current_task.task_id}.json"
        with open(json_path, "w") as f:
            json.dump(self._current_task.to_dict(), f, indent=2)

        logger.info(
            f"Completed task {self._current_task.task_id}: "
            f"{'SUCCESS' if success else 'FAILED'} "
            f"({self._current_task.successful_steps}/{self._current_task.total_steps} steps)"
        )

        completed_task = self._current_task
        self._current_task = None

        return completed_task

    def get_current_task(self) -> Optional[TaskLog]:
        """Get the current active task"""
        return self._current_task

    def get_task_by_id(self, task_id: str) -> Optional[TaskLog]:
        """Load a task from database/JSON"""
        json_path = self.tasks_dir / f"{task_id}.json"

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)

            # Reconstruct TaskLog
            task = TaskLog(
                task_id=data["task_id"],
                user_goal=data["user_goal"],
                status=data["status"],
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
                total_steps=data["total_steps"],
                successful_steps=data["successful_steps"],
                failed_steps=data["failed_steps"],
                final_result=data["final_result"],
                error_message=data["error_message"],
            )
            return task

        return None

    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tasks from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM tasks ORDER BY start_time DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all steps for a task"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM steps WHERE task_id = ? ORDER BY step_number
            """, (task_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            completed_tasks = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'completed'"
            ).fetchone()[0]
            failed_tasks = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'failed'"
            ).fetchone()[0]
            total_steps = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "total_steps": total_steps,
            "log_dir": str(self.log_dir),
            "db_path": str(self.db_path),
        }

    def cleanup_old_logs(self, max_age_days: int = 30) -> int:
        """Delete logs older than max_age_days"""
        cutoff = datetime.now().isoformat()[:10]  # Date only

        deleted = 0

        # Delete from database (simplified - real implementation would use date comparison)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM tasks WHERE date(start_time) < date('now', ?)
            """, (f"-{max_age_days} days",))
            deleted += cursor.rowcount
            conn.commit()

        # Delete JSON files
        for json_file in self.tasks_dir.glob("*.json"):
            try:
                stat = json_file.stat()
                age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                if age_days > max_age_days:
                    json_file.unlink()
                    deleted += 1
            except Exception:
                pass

        logger.info(f"Cleaned up {deleted} old log entries")
        return deleted
