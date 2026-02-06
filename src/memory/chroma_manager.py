"""ChromaDB vector store management for the autonomous browser agent"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.utils.config import config


@dataclass
class PageMemory:
    """Memory of a visited page"""
    url: str
    title: str
    screenshot_path: Optional[str] = None
    visible_text: str = ""
    action_taken: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "screenshot_path": self.screenshot_path,
            "visible_text": self.visible_text,
            "action_taken": self.action_taken,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_embedding_text(self) -> str:
        """Generate text for embedding"""
        return f"URL: {self.url}\nTitle: {self.title}\nAction: {self.action_taken or 'viewed'}\nText: {self.visible_text[:500]}"


@dataclass
class ExtractedData:
    """Extracted data with provenance"""
    data: Dict[str, Any]
    source_url: str
    description: str
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "source_url": self.source_url,
            "description": self.description,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_embedding_text(self) -> str:
        """Generate text for embedding"""
        data_str = json.dumps(self.data, indent=2)[:500]
        return f"Source: {self.source_url}\nDescription: {self.description}\nData: {data_str}"


@dataclass
class TaskPlanMemory:
    """Memory of a successful task plan"""
    user_goal: str
    plan_steps: List[Dict[str, Any]]
    success: bool
    execution_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_goal": self.user_goal,
            "plan_steps": self.plan_steps,
            "success": self.success,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_embedding_text(self) -> str:
        """Generate text for embedding"""
        steps_summary = "\n".join(
            f"Step {s.get('step', '?')}: {s.get('description', 'unknown')}"
            for s in self.plan_steps[:10]
        )
        return f"Goal: {self.user_goal}\nSteps:\n{steps_summary}"


class ChromaManager:
    """
    Manages ChromaDB vector store for agent memory.

    Collections:
    1. pages_visited: Screenshots and DOM snapshots
    2. extracted_data: Structured data from tasks
    3. task_plans: Successful task decompositions

    Memory Budget (from specification):
    - ChromaDB with 10k embeddings = ~2GB RAM
    - Uses nomic-embed-text (275M params, runs on CPU)
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = config.embedding_model,
    ):
        self.persist_directory = persist_directory or config.chromadb_dir
        self.embedding_model = embedding_model

        # Initialize ChromaDB client with persistence
        self._client: Optional[chromadb.Client] = None
        self._embedding_function: Optional[Any] = None

        # Collection references
        self._pages_collection: Optional[chromadb.Collection] = None
        self._data_collection: Optional[chromadb.Collection] = None
        self._plans_collection: Optional[chromadb.Collection] = None

    async def initialize(self) -> None:
        """Initialize ChromaDB and collections"""
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        # Initialize embedding function (using default for now)
        # In production, use sentence-transformers with nomic-embed-text
        try:
            from chromadb.utils import embedding_functions
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            logger.warning(f"Using default embeddings: {e}")
            self._embedding_function = None

        # Create or get collections
        self._pages_collection = self._client.get_or_create_collection(
            name="pages_visited",
            metadata={"description": "Screenshots and DOM snapshots of visited pages"},
            embedding_function=self._embedding_function,
        )

        self._data_collection = self._client.get_or_create_collection(
            name="extracted_data",
            metadata={"description": "Structured data extracted during tasks"},
            embedding_function=self._embedding_function,
        )

        self._plans_collection = self._client.get_or_create_collection(
            name="task_plans",
            metadata={"description": "Successful task decompositions for reuse"},
            embedding_function=self._embedding_function,
        )

        logger.info(
            f"ChromaDB initialized with collections: "
            f"pages={self._pages_collection.count()}, "
            f"data={self._data_collection.count()}, "
            f"plans={self._plans_collection.count()}"
        )

    async def store_page_visit(self, page_memory: PageMemory) -> str:
        """
        Store a page visit in memory.

        Args:
            page_memory: PageMemory object

        Returns:
            Generated document ID
        """
        if not self._pages_collection:
            await self.initialize()

        doc_id = str(uuid.uuid4())

        self._pages_collection.add(
            documents=[page_memory.to_embedding_text()],
            metadatas=[page_memory.to_dict()],
            ids=[doc_id],
        )

        logger.debug(f"Stored page visit: {page_memory.url}")
        return doc_id

    async def store_extracted_data(self, extracted_data: ExtractedData) -> str:
        """
        Store extracted data in memory.

        Args:
            extracted_data: ExtractedData object

        Returns:
            Generated document ID
        """
        if not self._data_collection:
            await self.initialize()

        doc_id = str(uuid.uuid4())

        self._data_collection.add(
            documents=[extracted_data.to_embedding_text()],
            metadatas=[{
                **extracted_data.to_dict(),
                "data_json": json.dumps(extracted_data.data),  # Store as string
            }],
            ids=[doc_id],
        )

        logger.debug(f"Stored extracted data from: {extracted_data.source_url}")
        return doc_id

    async def store_task_plan(self, plan_memory: TaskPlanMemory) -> str:
        """
        Store a successful task plan for future reuse.

        Args:
            plan_memory: TaskPlanMemory object

        Returns:
            Generated document ID
        """
        if not self._plans_collection:
            await self.initialize()

        doc_id = str(uuid.uuid4())

        self._plans_collection.add(
            documents=[plan_memory.to_embedding_text()],
            metadatas=[{
                "user_goal": plan_memory.user_goal,
                "success": plan_memory.success,
                "execution_time_seconds": plan_memory.execution_time_seconds,
                "timestamp": plan_memory.timestamp.isoformat(),
                "plan_json": json.dumps(plan_memory.plan_steps),
            }],
            ids=[doc_id],
        )

        logger.debug(f"Stored task plan: {plan_memory.user_goal}")
        return doc_id

    async def find_similar_pages(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find pages similar to the query.

        Args:
            query: Search query
            n_results: Maximum number of results

        Returns:
            List of matching page metadata
        """
        if not self._pages_collection:
            await self.initialize()

        results = self._pages_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        return results.get("metadatas", [[]])[0]

    async def find_relevant_data(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find extracted data relevant to the query.

        Args:
            query: Search query
            n_results: Maximum number of results

        Returns:
            List of matching data entries
        """
        if not self._data_collection:
            await self.initialize()

        results = self._data_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # Parse JSON data back
        entries = []
        for metadata in results.get("metadatas", [[]])[0]:
            if "data_json" in metadata:
                metadata["data"] = json.loads(metadata["data_json"])
                del metadata["data_json"]
            entries.append(metadata)

        return entries

    async def find_similar_plans(
        self,
        user_goal: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find task plans similar to the given goal.

        Useful for bootstrapping planning from past successes.

        Args:
            user_goal: The user's goal description
            n_results: Maximum number of plans to return

        Returns:
            List of similar task plans
        """
        if not self._plans_collection:
            await self.initialize()

        results = self._plans_collection.query(
            query_texts=[user_goal],
            n_results=n_results,
        )

        # Parse plan JSON back
        plans = []
        for metadata in results.get("metadatas", [[]])[0]:
            if "plan_json" in metadata:
                metadata["plan_steps"] = json.loads(metadata["plan_json"])
                del metadata["plan_json"]
            plans.append(metadata)

        return plans

    async def has_visited_page(self, url: str) -> bool:
        """Check if we've visited a specific URL before"""
        if not self._pages_collection:
            await self.initialize()

        results = self._pages_collection.get(
            where={"url": url},
            limit=1,
        )

        return len(results.get("ids", [])) > 0

    async def get_page_history(
        self,
        url: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get history of visits to a URL"""
        if not self._pages_collection:
            await self.initialize()

        results = self._pages_collection.get(
            where={"url": url},
            limit=limit,
        )

        return results.get("metadatas", [])

    async def cleanup_old_entries(
        self,
        max_age_days: int = 30,
    ) -> Dict[str, int]:
        """
        Delete entries older than max_age_days.

        Returns count of deleted entries per collection.
        """
        if not self._client:
            await self.initialize()

        cutoff = datetime.now() - timedelta(days=max_age_days)
        cutoff_str = cutoff.isoformat()

        deleted = {"pages": 0, "data": 0, "plans": 0}

        # Clean pages collection
        try:
            old_pages = self._pages_collection.get(
                where={"timestamp": {"$lt": cutoff_str}},
            )
            if old_pages["ids"]:
                self._pages_collection.delete(ids=old_pages["ids"])
                deleted["pages"] = len(old_pages["ids"])
        except Exception as e:
            logger.warning(f"Error cleaning pages: {e}")

        # Clean data collection
        try:
            old_data = self._data_collection.get(
                where={"timestamp": {"$lt": cutoff_str}},
            )
            if old_data["ids"]:
                self._data_collection.delete(ids=old_data["ids"])
                deleted["data"] = len(old_data["ids"])
        except Exception as e:
            logger.warning(f"Error cleaning data: {e}")

        logger.info(f"Cleaned up old entries: {deleted}")
        return deleted

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self._client:
            await self.initialize()

        return {
            "pages_count": self._pages_collection.count(),
            "data_count": self._data_collection.count(),
            "plans_count": self._plans_collection.count(),
            "persist_directory": str(self.persist_directory),
        }

    async def clear_all(self) -> None:
        """Clear all collections (use with caution)"""
        if not self._client:
            await self.initialize()

        # Delete and recreate collections
        self._client.delete_collection("pages_visited")
        self._client.delete_collection("extracted_data")
        self._client.delete_collection("task_plans")

        # Reinitialize
        await self.initialize()

        logger.warning("All memory collections cleared")
