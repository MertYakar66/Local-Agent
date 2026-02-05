"""Tests for memory and logging modules"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.memory.chroma_manager import (
    ChromaManager,
    PageMemory,
    ExtractedData,
    TaskPlanMemory,
)
from src.memory.logger import StructuredLogger, StepLog, TaskLog


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPageMemory:
    """Test PageMemory dataclass"""

    def test_create_page_memory(self):
        """Test creating a PageMemory object"""
        memory = PageMemory(
            url="https://example.com",
            title="Example Domain",
            visible_text="This is an example page",
            action_taken="click",
        )

        assert memory.url == "https://example.com"
        assert memory.title == "Example Domain"
        assert memory.action_taken == "click"

    def test_page_memory_to_dict(self):
        """Test converting to dictionary"""
        memory = PageMemory(
            url="https://test.com",
            title="Test Page",
        )

        d = memory.to_dict()
        assert d["url"] == "https://test.com"
        assert "timestamp" in d

    def test_page_memory_embedding_text(self):
        """Test generating embedding text"""
        memory = PageMemory(
            url="https://example.com",
            title="Example",
            visible_text="Some text content",
            action_taken="navigate",
        )

        text = memory.to_embedding_text()
        assert "example.com" in text
        assert "Example" in text
        assert "navigate" in text


class TestExtractedData:
    """Test ExtractedData dataclass"""

    def test_create_extracted_data(self):
        """Test creating ExtractedData"""
        data = ExtractedData(
            data={"price": 999.99, "product": "iPhone 15"},
            source_url="https://amazon.com/product",
            description="Product price",
            task_id="abc123",
        )

        assert data.data["price"] == 999.99
        assert data.source_url == "https://amazon.com/product"

    def test_extracted_data_to_dict(self):
        """Test converting to dictionary"""
        data = ExtractedData(
            data={"value": 100},
            source_url="https://test.com",
            description="Test data",
            task_id="test",
        )

        d = data.to_dict()
        assert d["data"]["value"] == 100
        assert "timestamp" in d


class TestChromaManager:
    """Test ChromaDB manager"""

    @pytest.mark.asyncio
    async def test_initialize(self, temp_dir):
        """Test ChromaDB initialization"""
        manager = ChromaManager(persist_directory=temp_dir / "chroma")
        await manager.initialize()

        stats = await manager.get_stats()
        assert stats["pages_count"] == 0
        assert stats["data_count"] == 0
        assert stats["plans_count"] == 0

    @pytest.mark.asyncio
    async def test_store_page_visit(self, temp_dir):
        """Test storing a page visit"""
        manager = ChromaManager(persist_directory=temp_dir / "chroma")
        await manager.initialize()

        memory = PageMemory(
            url="https://example.com",
            title="Example",
            visible_text="Test content",
        )

        doc_id = await manager.store_page_visit(memory)
        assert doc_id is not None

        stats = await manager.get_stats()
        assert stats["pages_count"] == 1

    @pytest.mark.asyncio
    async def test_store_extracted_data(self, temp_dir):
        """Test storing extracted data"""
        manager = ChromaManager(persist_directory=temp_dir / "chroma")
        await manager.initialize()

        data = ExtractedData(
            data={"price": 100},
            source_url="https://test.com",
            description="Price data",
            task_id="test123",
        )

        doc_id = await manager.store_extracted_data(data)
        assert doc_id is not None

        stats = await manager.get_stats()
        assert stats["data_count"] == 1

    @pytest.mark.asyncio
    async def test_find_similar_pages(self, temp_dir):
        """Test finding similar pages"""
        manager = ChromaManager(persist_directory=temp_dir / "chroma")
        await manager.initialize()

        # Store some pages
        await manager.store_page_visit(PageMemory(
            url="https://amazon.com",
            title="Amazon",
            visible_text="Shopping site",
        ))
        await manager.store_page_visit(PageMemory(
            url="https://ebay.com",
            title="eBay",
            visible_text="Auction site",
        ))

        # Search for similar
        results = await manager.find_similar_pages("online shopping")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_store_and_find_plan(self, temp_dir):
        """Test storing and finding task plans"""
        manager = ChromaManager(persist_directory=temp_dir / "chroma")
        await manager.initialize()

        plan = TaskPlanMemory(
            user_goal="Compare prices on Amazon and eBay",
            plan_steps=[
                {"step": 1, "action": "navigate", "target": "amazon.com"},
                {"step": 2, "action": "search", "target": "iPhone"},
            ],
            success=True,
            execution_time_seconds=30.0,
        )

        doc_id = await manager.store_task_plan(plan)
        assert doc_id is not None

        # Find similar plans
        results = await manager.find_similar_plans("compare product prices")
        assert len(results) > 0


class TestStepLog:
    """Test StepLog dataclass"""

    def test_create_step_log(self):
        """Test creating a StepLog"""
        log = StepLog(
            task_id="task123",
            step_number=1,
            stage="vision_action",
            tab_id=0,
            action_type="click",
            confidence=0.95,
        )

        assert log.task_id == "task123"
        assert log.step_number == 1
        assert log.action_type == "click"

    def test_step_log_to_dict(self):
        """Test converting step log to dict"""
        log = StepLog(
            task_id="task123",
            step_number=2,
            stage="execution",
            tab_id=0,
            playwright_result="success",
        )

        d = log.to_dict()
        assert d["step_number"] == 2
        assert d["execution"]["playwright_result"] == "success"


class TestTaskLog:
    """Test TaskLog dataclass"""

    def test_create_task_log(self):
        """Test creating a TaskLog"""
        log = TaskLog(
            task_id="task123",
            user_goal="Test goal",
        )

        assert log.task_id == "task123"
        assert log.status == "pending"
        assert len(log.steps) == 0

    def test_task_log_duration(self):
        """Test task duration calculation"""
        log = TaskLog(
            task_id="task123",
            user_goal="Test goal",
        )

        # Duration should be > 0 even without end time
        assert log.duration_seconds >= 0


class TestStructuredLogger:
    """Test StructuredLogger"""

    def test_init_logger(self, temp_dir):
        """Test logger initialization"""
        logger = StructuredLogger(log_dir=temp_dir)
        stats = logger.get_stats()

        assert stats["total_tasks"] == 0
        assert Path(stats["db_path"]).exists()

    def test_start_task(self, temp_dir):
        """Test starting a task"""
        logger = StructuredLogger(log_dir=temp_dir)
        task_id = logger.start_task("Test task")

        assert task_id is not None
        assert len(task_id) == 8  # UUID first 8 chars

        current = logger.get_current_task()
        assert current.user_goal == "Test task"

    def test_log_step(self, temp_dir):
        """Test logging a step"""
        logger = StructuredLogger(log_dir=temp_dir)
        logger.start_task("Test task")

        step = logger.log_step(
            step_number=1,
            stage="execution",
            tab_id=0,
            action_type="click",
            confidence=0.95,
            playwright_result="success",
        )

        assert step.step_number == 1
        assert step.action_type == "click"

    def test_complete_task(self, temp_dir):
        """Test completing a task"""
        logger = StructuredLogger(log_dir=temp_dir)
        task_id = logger.start_task("Test task")

        logger.log_step(
            step_number=1,
            stage="execution",
            tab_id=0,
            action_type="click",
            verification_status="success",
        )

        task_log = logger.complete_task(
            success=True,
            final_result={"data": "test"},
        )

        assert task_log.status == "completed"
        assert task_log.successful_steps == 1

        # Check JSON file was created
        json_path = temp_dir / "tasks" / f"{task_id}.json"
        assert json_path.exists()

    def test_get_recent_tasks(self, temp_dir):
        """Test getting recent tasks"""
        logger = StructuredLogger(log_dir=temp_dir)

        # Create a few tasks
        for i in range(3):
            logger.start_task(f"Task {i}")
            logger.complete_task(success=True)

        tasks = logger.get_recent_tasks(limit=2)
        assert len(tasks) == 2

    def test_get_task_steps(self, temp_dir):
        """Test getting steps for a task"""
        logger = StructuredLogger(log_dir=temp_dir)
        task_id = logger.start_task("Test task")

        for i in range(3):
            logger.log_step(
                step_number=i + 1,
                stage="execution",
                tab_id=0,
                action_type="click",
            )

        logger.complete_task(success=True)

        steps = logger.get_task_steps(task_id)
        assert len(steps) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
