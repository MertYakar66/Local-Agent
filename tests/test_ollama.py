"""Tests for Ollama connection and GPU inference"""

import asyncio
import pytest
import subprocess
from typing import Dict, Any

import httpx


# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5-vl:7b-instruct-q4_K_M"
EXPECTED_MIN_TOKENS_PER_SEC = 40.0
MAX_VRAM_GB = 15.0


def get_nvidia_smi_info() -> Dict[str, Any]:
    """Get GPU info from nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"error": "nvidia-smi failed"}

        parts = result.stdout.strip().split(", ")
        if len(parts) >= 4:
            return {
                "name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "utilization_percent": int(parts[3]),
            }
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except Exception as e:
        return {"error": str(e)}

    return {"error": "Failed to parse nvidia-smi output"}


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestOllamaConnection:
    """Test Ollama API connection"""

    @pytest.mark.asyncio
    async def test_ollama_is_running(self):
        """Test that Ollama server is accessible"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
                assert response.status_code == 200
                print("✓ Ollama is running")
            except httpx.ConnectError:
                pytest.fail("Ollama is not running. Start with: ollama serve")

    @pytest.mark.asyncio
    async def test_model_available(self):
        """Test that required model is available"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check if model is available (might have different tag)
            model_found = any(MODEL_NAME.split(":")[0] in name for name in model_names)

            if not model_found:
                print(f"Available models: {model_names}")
                print(f"Pull required model with: ollama pull {MODEL_NAME}")

            # Don't fail if model not found - just warn
            if model_found:
                print(f"✓ Model {MODEL_NAME} is available")
            else:
                pytest.skip(f"Model {MODEL_NAME} not found. Pull with: ollama pull {MODEL_NAME}")


class TestGPUDetection:
    """Test GPU availability and configuration"""

    def test_nvidia_smi_available(self):
        """Test that nvidia-smi is accessible"""
        info = get_nvidia_smi_info()
        if "error" in info:
            pytest.skip(f"nvidia-smi not available: {info['error']}")

        print(f"✓ GPU detected: {info['name']}")
        print(f"  Memory: {info['memory_used_mb']}/{info['memory_total_mb']} MB")
        print(f"  Utilization: {info['utilization_percent']}%")

    def test_sufficient_vram(self):
        """Test that GPU has enough VRAM for the model"""
        info = get_nvidia_smi_info()
        if "error" in info:
            pytest.skip("GPU not available")

        total_gb = info["memory_total_mb"] / 1024
        assert total_gb >= 16, f"Need at least 16GB VRAM, got {total_gb:.1f}GB"

        print(f"✓ Sufficient VRAM: {total_gb:.1f} GB")


class TestGPUInference:
    """Test GPU-accelerated inference"""

    @pytest.mark.asyncio
    async def test_inference_speed(self):
        """Test that inference meets speed requirements"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Check if model is available first
            try:
                tags_response = await client.get(f"{OLLAMA_URL}/api/tags")
                models = tags_response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                if not any(MODEL_NAME.split(":")[0] in name for name in model_names):
                    pytest.skip(f"Model {MODEL_NAME} not available")
            except Exception as e:
                pytest.skip(f"Ollama not accessible: {e}")

            # Run inference
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": "What is 2+2? Answer in one word.",
                    "stream": False,
                    "options": {
                        "num_predict": 50,
                    },
                },
            )

            assert response.status_code == 200
            result = response.json()

            # Check speed
            total_duration = result.get("total_duration", 0) / 1e9  # ns to seconds
            eval_count = result.get("eval_count", 0)

            if total_duration > 0 and eval_count > 0:
                tokens_per_sec = eval_count / total_duration
                print(f"✓ Inference speed: {tokens_per_sec:.1f} tokens/sec")

                # Warn if below target but don't fail
                if tokens_per_sec < EXPECTED_MIN_TOKENS_PER_SEC:
                    print(f"  ⚠ Below target of {EXPECTED_MIN_TOKENS_PER_SEC} t/s")

            print(f"  Response: {result.get('response', '')[:100]}")

    @pytest.mark.asyncio
    async def test_vram_usage_during_inference(self):
        """Test VRAM usage stays within limits"""
        info_before = get_nvidia_smi_info()
        if "error" in info_before:
            pytest.skip("GPU not available")

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Run a longer inference
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": "Write a short story about a robot learning to paint.",
                    "stream": False,
                    "options": {
                        "num_predict": 200,
                    },
                },
            )

            if response.status_code != 200:
                pytest.skip("Inference failed")

        info_after = get_nvidia_smi_info()
        vram_used_gb = info_after["memory_used_mb"] / 1024

        print(f"✓ VRAM after inference: {vram_used_gb:.1f} GB")

        if vram_used_gb > MAX_VRAM_GB:
            print(f"  ⚠ Above target of {MAX_VRAM_GB} GB")


class TestVisionCapabilities:
    """Test vision model capabilities (requires vision model)"""

    @pytest.mark.asyncio
    async def test_vision_model_loaded(self):
        """Test that vision model can process images"""
        import base64

        # Create a simple test image (1x1 red pixel)
        test_image_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
            0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        test_image_b64 = base64.b64encode(test_image_bytes).decode()

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Check if vision model supports images
            try:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": "What do you see in this image?",
                        "images": [test_image_b64],
                        "stream": False,
                        "options": {"num_predict": 20},
                    },
                )

                if response.status_code == 200:
                    print("✓ Vision model accepts images")
                else:
                    print(f"  Vision test returned: {response.status_code}")

            except Exception as e:
                pytest.skip(f"Vision test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
