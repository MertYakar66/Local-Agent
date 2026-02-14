"""Base agent class with multi-provider LLM support (Claude API + Ollama)"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx
from loguru import logger

from src.utils.config import config


@dataclass
class LLMConfig:
    """Configuration for LLM inference"""
    # Provider: "claude" or "ollama"
    provider: str = config.llm_provider

    # Claude settings
    anthropic_api_key: str = config.anthropic_api_key
    claude_model: str = config.claude_model

    # Ollama settings
    model_name: str = config.model_name
    fallback_model_name: str = config.fallback_model_name
    fallback_after_failures: int = config.fallback_after_failures
    temperature: float = config.temperature
    context_length: int = config.context_length
    ollama_url: str = config.ollama_url


class BaseAgent:
    """
    Base class for all agents with multi-provider LLM support.

    Supports:
    - Claude API (cloud): Fast, reliable, no local GPU needed
    - Ollama (local): Qwen3-VL with tiered intelligence (8B primary, 30B fallback)
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        agent_name: str = "base",
    ):
        self.llm_config = llm_config or LLMConfig()
        self.agent_name = agent_name

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Tiered intelligence state (Ollama only)
        self._using_fallback: bool = False
        self._consecutive_failures: int = 0
        self._active_model: str = self.llm_config.model_name

    async def initialize(self) -> None:
        """Initialize the HTTP client"""
        if self._client is None:
            if self.llm_config.provider == "claude":
                self._client = httpx.AsyncClient(
                    base_url="https://api.anthropic.com",
                    timeout=httpx.Timeout(120.0),
                    headers={
                        "x-api-key": self.llm_config.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
            else:
                self._client = httpx.AsyncClient(
                    base_url=self.llm_config.ollama_url,
                    timeout=httpx.Timeout(120.0),
                )
            logger.info(
                f"Agent '{self.agent_name}' initialized "
                f"(provider: {self.llm_config.provider})"
            )

    async def close(self) -> None:
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized"""
        if self._client is None:
            await self.initialize()
        return self._client

    async def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        Generate response from the configured LLM provider.

        Args:
            prompt: Text prompt
            images: Optional list of base64-encoded images (Ollama vision only)
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            json_mode: If True, force JSON output format

        Returns:
            Generated text response
        """
        if self.llm_config.provider == "claude":
            return await self._generate_claude(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        else:
            return await self._generate_ollama(
                prompt=prompt,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

    async def _generate_claude(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Generate response using Claude API"""
        client = await self._ensure_client()

        # Build system message for JSON mode
        system_msg = None
        if json_mode:
            system_msg = "You must respond with valid JSON only. No explanations, no markdown, just the JSON object."

        # Build the messages payload
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.llm_config.claude_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system_msg:
            payload["system"] = system_msg

        if temperature is not None:
            payload["temperature"] = temperature
        else:
            payload["temperature"] = self.llm_config.temperature

        logger.debug(f"[{self.agent_name}] Generating via Claude API...")

        try:
            response = await client.post("/v1/messages", json=payload)
            response.raise_for_status()

            result = response.json()
            # Claude response format: {"content": [{"type": "text", "text": "..."}]}
            content_blocks = result.get("content", [])
            generated_text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    generated_text += block.get("text", "")

            # Log usage
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            logger.debug(
                f"[{self.agent_name}] Claude: {input_tokens} input, "
                f"{output_tokens} output tokens"
            )

            return generated_text

        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.agent_name}] Claude API error: {e}")
            logger.error(f"[{self.agent_name}] Response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[{self.agent_name}] Claude generation failed: {e}")
            raise

    async def _generate_ollama(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Generate response using Ollama API (local)"""
        client = await self._ensure_client()

        payload = {
            "model": self._active_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.llm_config.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.llm_config.context_length,
            },
        }

        if json_mode:
            payload["format"] = "json"

        if images:
            payload["images"] = images

        logger.debug(f"[{self.agent_name}] Generating via Ollama...")

        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            # Handle qwen3 thinking mode workaround
            thinking_text = result.get("thinking", "")
            eval_count = result.get("eval_count", 0)

            if eval_count > 0 and not generated_text and thinking_text:
                logger.warning(
                    f"[{self.agent_name}] Response empty but thinking has "
                    f"{len(thinking_text)} chars. Extracting..."
                )
                if '[' in thinking_text or '{' in thinking_text:
                    generated_text = thinking_text

            # Log performance
            total_duration = result.get("total_duration", 0) / 1e9
            if total_duration > 0 and eval_count > 0:
                tokens_per_sec = eval_count / total_duration
                logger.debug(
                    f"[{self.agent_name}] {eval_count} tokens "
                    f"in {total_duration:.2f}s ({tokens_per_sec:.1f} t/s)"
                )

            return generated_text

        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.agent_name}] Ollama HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self.agent_name}] Ollama generation failed: {e}")
            raise

    async def generate_json(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.

        Includes retry logic for malformed JSON.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                response_text = await self.generate(
                    prompt=prompt,
                    images=images,
                    temperature=temperature,
                    json_mode=(self.llm_config.provider == "ollama"),
                )

                json_data = self._extract_json(response_text)

                if json_data is not None:
                    return json_data

                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{self.agent_name}] Failed to parse JSON "
                        f"(attempt {attempt + 1}), retrying..."
                    )
                    prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Respond ONLY with valid JSON, no other text.\n\n"
                        f"Original request:\n{prompt}"
                    )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{self.agent_name}] Generation error "
                    f"(attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise ValueError(
            f"Failed to generate valid JSON after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from model response."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        json_patterns = [
            r'\{[^{}]*\}',
            r'\[[^\[\]]*\]',
            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',
            r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    async def check_model_loaded(self) -> bool:
        """Check if the model is available"""
        if self.llm_config.provider == "claude":
            # Claude is always available if API key is set
            return bool(self.llm_config.anthropic_api_key)

        client = await self._ensure_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(
                self.llm_config.model_name in name for name in model_names
            )
        except Exception as e:
            logger.warning(f"Failed to check model status: {e}")
            return False

    async def preload_model(self) -> bool:
        """Preload the model (Ollama only; Claude needs no preloading)"""
        if self.llm_config.provider == "claude":
            logger.info("Using Claude API - no model preloading needed")
            return True

        logger.info(f"Preloading model: {self.llm_config.model_name}")
        try:
            await self.generate(prompt="Hello", max_tokens=1)
            logger.info("Model preloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
            return False

    # ============================================================
    # Tiered Intelligence: Model Switching Methods (Ollama only)
    # ============================================================

    def record_success(self) -> None:
        """Record a successful action"""
        self._consecutive_failures = 0
        if self._using_fallback:
            self._switch_to_primary()

    def record_failure(self) -> bool:
        """Record a failed action. Returns True if switched to fallback."""
        self._consecutive_failures += 1
        logger.warning(
            f"[{self.agent_name}] Consecutive failures: "
            f"{self._consecutive_failures}"
        )

        if (
            self.llm_config.provider == "ollama"
            and not self._using_fallback
            and self._consecutive_failures >= self.llm_config.fallback_after_failures
        ):
            self._switch_to_fallback()
            return True

        return False

    def _switch_to_fallback(self) -> None:
        """Switch to the fallback model"""
        logger.warning(
            f"[{self.agent_name}] Switching to fallback: "
            f"{self.llm_config.fallback_model_name}"
        )
        self._active_model = self.llm_config.fallback_model_name
        self._using_fallback = True
        self._consecutive_failures = 0

    def _switch_to_primary(self) -> None:
        """Switch back to primary model"""
        logger.info(
            f"[{self.agent_name}] Switching back to primary: "
            f"{self.llm_config.model_name}"
        )
        self._active_model = self.llm_config.model_name
        self._using_fallback = False
        self._consecutive_failures = 0

    def force_fallback(self) -> None:
        if not self._using_fallback:
            self._switch_to_fallback()

    def force_primary(self) -> None:
        if self._using_fallback:
            self._switch_to_primary()

    @property
    def is_using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def current_model(self) -> str:
        if self.llm_config.provider == "claude":
            return self.llm_config.claude_model
        return self._active_model


async def test_ollama_connection(ollama_url: str = config.ollama_url) -> Dict[str, Any]:
    """Test connection to Ollama and verify GPU availability."""
    result = {
        "connected": False,
        "models": [],
        "gpu_available": False,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(base_url=ollama_url, timeout=10.0) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            result["connected"] = True
            result["models"] = [
                m.get("name", "") for m in response.json().get("models", [])
            ]

            test_response = await client.post(
                "/api/generate",
                json={
                    "model": config.model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=60.0,
            )
            if test_response.status_code == 200:
                result["gpu_available"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
