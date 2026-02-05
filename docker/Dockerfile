# Vision-driven Autonomous Browser Agent
# Based on official Playwright image with GPU support
FROM mcr.microsoft.com/playwright:v1.45.0-noble

# Labels
LABEL maintainer="Autonomous Browser Agent"
LABEL description="Vision-driven browser automation with Qwen2.5-VL"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    curl \
    wget \
    gnupg2 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000 as specified)
RUN groupadd -g 1000 agentuser && \
    useradd -u 1000 -g agentuser -m -s /bin/bash agentuser

# Install NVIDIA drivers and CUDA runtime (for GPU passthrough)
# This assumes the host has NVIDIA drivers installed and nvidia-container-toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Create app directories
RUN mkdir -p /app /logs /data && \
    chown -R agentuser:agentuser /app /logs /data

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install uv && \
    uv pip install --system -e /app[dev]

# Install Playwright browsers
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code
COPY --chown=agentuser:agentuser src/ /app/src/
COPY --chown=agentuser:agentuser tests/ /app/tests/

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs/screenshots /app/logs/tasks /app/data/chromadb && \
    chown -R agentuser:agentuser /app/logs /app/data

# Switch to non-root user
USER agentuser

# Expose ports
EXPOSE 8501  # Streamlit
EXPOSE 9222  # Chrome DevTools Protocol

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["python3", "-m", "streamlit", "run", "src/ui/streamlit_app.py", "--server.address=0.0.0.0"]
