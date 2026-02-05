"""Base agent class with shared LLM connection logic"""

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
    model_name: str = config.model_name
    temperature: float = config.temperature
    context_length: int = config.context_length
    ollama_url: str = config.ollama_url


class BaseAgent:
    """
    Base class for all agents with shared LLM connection logic.

    Optimized for RTX 4080 Super (16GB VRAM):
    - Qwen2.5-VL-7B at 4-bit quantization uses ~10.5GB VRAM
    - KV cache for 32k tokens uses ~4GB VRAM during long sessions
    - Target inference speed: 40-60 tokens/second
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        agent_name: str = "base",
    ):
        self.llm_config = llm_config or LLMConfig()
        self.agent_name = agent_name

        # HTTP client for Ollama API
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize the HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.llm_config.ollama_url,
                timeout=httpx.Timeout(120.0),  # Long timeout for model loading
            )
            logger.info(f"Agent '{self.agent_name}' initialized")

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
    ) -> str:
        """
        Generate response from Ollama.

        Args:
            prompt: Text prompt
            images: Optional list of base64-encoded images
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        client = await self._ensure_client()

        # Build request payload
        payload = {
            "model": self.llm_config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.llm_config.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.llm_config.context_length,
            },
        }

        # Add images for vision model
        if images:
            payload["images"] = images

        logger.debug(f"[{self.agent_name}] Generating response...")

        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            # Log performance metrics
            total_duration = result.get("total_duration", 0) / 1e9  # nanoseconds to seconds
            eval_count = result.get("eval_count", 0)

            if total_duration > 0 and eval_count > 0:
                tokens_per_sec = eval_count / total_duration
                logger.debug(
                    f"[{self.agent_name}] Generated {eval_count} tokens "
                    f"in {total_duration:.2f}s ({tokens_per_sec:.1f} t/s)"
                )

            return generated_text

        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.agent_name}] HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self.agent_name}] Generation failed: {e}")
            raise

    async def generate_json(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response from Ollama.

        Includes retry logic for malformed JSON.

        Args:
            prompt: Text prompt (should ask for JSON output)
            images: Optional list of base64-encoded images
            temperature: Override default temperature
            max_retries: Number of retries for parsing errors

        Returns:
            Parsed JSON as dictionary
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Generate response
                response_text = await self.generate(
                    prompt=prompt,
                    images=images,
                    temperature=temperature,
                )

                # Try to extract JSON from response
                json_data = self._extract_json(response_text)

                if json_data is not None:
                    return json_data

                # If no valid JSON found, retry with clearer instruction
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{self.agent_name}] Failed to parse JSON (attempt {attempt + 1}), "
                        "retrying with clearer instruction..."
                    )
                    prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Respond ONLY with valid JSON, no other text.\n\n"
                        f"Original request:\n{prompt}"
                    )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{self.agent_name}] Generation error (attempt {attempt + 1}): {e}"
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        raise ValueError(
            f"Failed to generate valid JSON after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from model response.

        Handles cases where JSON is embedded in text or wrapped in code blocks.
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # Nested object
            r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]',  # Nested array
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
        """Check if the model is loaded in Ollama"""
        client = await self._ensure_client()

        try:
            response = await client.get("/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check if our model is in the list
            is_loaded = any(
                self.llm_config.model_name in name
                for name in model_names
            )

            return is_loaded

        except Exception as e:
            logger.warning(f"Failed to check model status: {e}")
            return False

    async def preload_model(self) -> bool:
        """
        Preload the model by sending a simple prompt.

        This ensures the model is in GPU memory before actual tasks.
        """
        logger.info(f"Preloading model: {self.llm_config.model_name}")

        try:
            await self.generate(
                prompt="Hello",
                max_tokens=1,
            )
            logger.info("Model preloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        client = await self._ensure_client()

        try:
            response = await client.post(
                "/api/show",
                json={"name": self.llm_config.model_name},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {}


async def test_ollama_connection(ollama_url: str = config.ollama_url) -> Dict[str, Any]:
    """
    Test connection to Ollama and verify GPU availability.

    Returns diagnostic information.
    """
    result = {
        "connected": False,
        "models": [],
        "gpu_available": False,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(base_url=ollama_url, timeout=10.0) as client:
            # Test connection
            response = await client.get("/api/tags")
            response.raise_for_status()

            result["connected"] = True
            result["models"] = [
                m.get("name", "") for m in response.json().get("models", [])
            ]

            # Test GPU by running inference
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
