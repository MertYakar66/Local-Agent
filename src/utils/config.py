"""Central configuration for the autonomous browser agent"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class AgentConfig(BaseModel):
    """Central configuration for the autonomous agent

    Supports both local (Ollama) and cloud (Claude) LLM providers.
    """

    # Project Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # LLM Provider: "claude" or "ollama"
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "claude")
    )

    # Claude API Settings
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    claude_model: str = Field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    )

    # Ollama Settings (fallback / local mode)
    ollama_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "qwen3-vl:8b")
    )
    fallback_model_name: str = Field(
        default_factory=lambda: os.getenv("FALLBACK_MODEL_NAME", "qwen3-vl:30b")
    )
    fallback_after_failures: int = Field(
        default_factory=lambda: int(os.getenv("FALLBACK_AFTER_FAILURES", "3"))
    )
    context_length: int = 32768  # 32k tokens
    temperature: float = 0.3  # Low temp for deterministic actions

    # Browser Settings
    viewport_width: int = Field(
        default_factory=lambda: int(os.getenv("VIEWPORT_WIDTH", "1280"))
    )
    viewport_height: int = Field(
        default_factory=lambda: int(os.getenv("VIEWPORT_HEIGHT", "720"))
    )
    max_tabs: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TABS", "10"))
    )
    page_timeout_ms: int = 30000  # 30 seconds
    headless: bool = Field(
        default_factory=lambda: os.getenv("HEADLESS", "true").lower() == "true"
    )

    # Memory Settings
    chromadb_path: str = Field(
        default_factory=lambda: os.getenv("CHROMADB_PATH", "./data/chromadb")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    log_retention_days: int = 30

    # Safety Settings
    require_approval_for_sensitive: bool = Field(
        default_factory=lambda: os.getenv("REQUIRE_APPROVAL_FOR_SENSITIVE", "true").lower() == "true"
    )
    max_retries_per_action: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES_PER_ACTION", "3"))
    )
    action_confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("ACTION_CONFIDENCE_THRESHOLD", "0.8"))
    )

    # Performance Settings
    max_vram_gb: float = Field(
        default_factory=lambda: float(os.getenv("MAX_VRAM_GB", "15.0"))
    )
    # Logging Settings
    log_dir: str = Field(
        default_factory=lambda: os.getenv("LOG_DIR", "./logs")
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    # Sensitive action detection patterns
    sensitive_action_patterns: list[str] = [
        r"buy|purchase|pay|checkout|confirm.*order",
        r"delete|remove|cancel",
        r"submit.*payment|enter.*credit.*card",
        r"apply.*now|sign.*contract",
    ]

    sensitive_url_patterns: list[str] = [
        r".*/checkout",
        r".*/payment",
        r".*/cart",
        r".*/delete",
        r".*/admin",
    ]

    @property
    def is_claude(self) -> bool:
        """Check if using Claude as the LLM provider"""
        return self.llm_provider.lower() == "claude"

    @property
    def tasks_dir(self) -> Path:
        """Directory for storing task logs"""
        path = Path(self.log_dir) / "tasks"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chromadb_dir(self) -> Path:
        """Directory for ChromaDB storage"""
        path = Path(self.chromadb_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_ollama_api_url(self) -> str:
        """Get full Ollama API URL"""
        return f"{self.ollama_url}/api"

    def get_viewport_size(self) -> dict[str, int]:
        """Get viewport size as dict"""
        return {"width": self.viewport_width, "height": self.viewport_height}

    class Config:
        arbitrary_types_allowed = True


# Global config instance - load config from environment or use defaults
config = AgentConfig()


def get_config() -> AgentConfig:
    """Get the global config instance"""
    return config


def reload_config() -> AgentConfig:
    """Reload config from environment"""
    global config
    load_dotenv(override=True)
    config = AgentConfig()
    return config
