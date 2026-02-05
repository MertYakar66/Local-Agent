# Vision-Driven Autonomous Browser Agent

A local AI browser agent that performs autonomous web navigation using visual perception. Optimized for AMD Ryzen 9800X3D + RTX 4080 Super (16GB VRAM) + 64GB DDR5.

## Features

- **Visual Understanding**: Interprets screenshots to identify buttons, forms, links, and text fields
- **Multi-Step Planning**: Decomposes complex goals into actionable steps
- **Precise Interaction**: Clicks, scrolls, fills forms using coordinates derived from bounding boxes
- **Multi-Tab Management**: Operates 5-10 browser tabs simultaneously for parallel data gathering
- **Memory & Learning**: Remembers past interactions to avoid repeating mistakes
- **Safety Controls**: Requires human approval before financial/destructive actions
- **Streamlit Dashboard**: Real-time monitoring and human-in-the-loop control

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  - Live screenshots    - Agent thoughts    - HITL approval   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│  1. PLANNER → 2. VISION-ACTION → 3. EXECUTION → 4. VERIFY   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Browser Automation                         │
│         Playwright + Docker + Multi-Tab Management           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Persistence Layer                        │
│              ChromaDB + SQLite + JSON Logs                   │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 4080 Super (16GB VRAM) or equivalent
- **CPU**: AMD Ryzen 9800X3D or equivalent (16 cores recommended)
- **RAM**: 64GB DDR5 recommended (minimum 32GB)

### Software
- Python 3.11+
- Ollama with Qwen2.5-VL model
- Docker with NVIDIA Container Toolkit
- Playwright browsers

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd autonomous-browser-agent
```

2. **Install dependencies**:
```bash
pip install uv
uv pip install -e .[dev]
```

3. **Install Playwright browsers**:
```bash
playwright install chromium
```

4. **Set up Ollama**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Pull the vision model
ollama pull qwen2.5-vl:7b-instruct-q4_K_M
```

5. **Copy environment configuration**:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Streamlit Dashboard (Recommended)

```bash
python -m streamlit run src/ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Command Line

```bash
python -m src.main "Compare iPhone 15 prices on Amazon and B&H"
```

### Docker

```bash
# Build and run with Docker Compose
cd docker
docker-compose up -d

# Access dashboard at http://localhost:8501
```

## Project Structure

```
autonomous-browser-agent/
├── src/
│   ├── agents/
│   │   ├── planner.py          # Task decomposition
│   │   ├── vision_actor.py     # Screenshot analysis
│   │   ├── verifier.py         # Action verification
│   │   └── base_agent.py       # Shared LLM logic
│   ├── browser/
│   │   ├── tab_manager.py      # Multi-tab orchestration
│   │   ├── action_executor.py  # Playwright commands
│   │   └── screenshot_utils.py # Image capture
│   ├── memory/
│   │   ├── chroma_manager.py   # Vector store
│   │   └── logger.py           # Structured logging
│   ├── ui/
│   │   ├── streamlit_app.py    # Dashboard
│   │   └── components.py       # UI widgets
│   ├── utils/
│   │   ├── config.py           # Settings
│   │   ├── vision_utils.py     # Coordinate translation
│   │   └── error_handling.py   # Retry logic
│   └── main.py                 # Entry point
├── tests/
│   ├── test_ollama.py
│   ├── test_playwright.py
│   ├── test_agents.py
│   ├── test_memory.py
│   └── integration/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── seccomp_profile.json
├── data/                       # ChromaDB storage
├── logs/                       # Screenshots and logs
└── pyproject.toml
```

## Configuration

Key settings in `.env`:

```bash
# Model
OLLAMA_URL=http://localhost:11434
MODEL_NAME=qwen2.5-vl:7b-instruct-q4_K_M

# Browser
HEADLESS=true
MAX_TABS=10
VIEWPORT_WIDTH=1280
VIEWPORT_HEIGHT=720

# Safety
REQUIRE_APPROVAL_FOR_SENSITIVE=true
MAX_RETRIES_PER_ACTION=3
ACTION_CONFIDENCE_THRESHOLD=0.8
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_ollama.py -v -s
pytest tests/test_playwright.py -v -s
pytest tests/integration/ -v -s
```

## Performance Benchmarks

| Task | Latency | VRAM | RAM |
|------|---------|------|-----|
| Planner (decompose task) | 1-2s | 10.5GB | 1GB |
| Vision-Action (get action) | 2-4s | 14GB | 1GB |
| Playwright action (click) | 200-500ms | 0 | +200MB |
| Verifier (check success) | 1-2s | 14GB | 1GB |
| **Full step (end-to-end)** | **4-7s** | **14.5GB** | **9GB** |

## Human-in-the-Loop (HITL)

Sensitive actions automatically trigger approval:
- Purchase/checkout actions
- Delete/remove operations
- Payment submissions
- Contract signing

The dashboard displays:
- Screenshot with target highlighted
- Action details and reasoning
- Approve / Reject / Modify buttons

## Safety Features

- **Docker isolation**: Browser runs in sandboxed container
- **Seccomp profile**: Blocks dangerous syscalls
- **Non-root user**: Container runs as UID 1000
- **Sensitive action detection**: Pattern matching on actions and URLs
- **Retry limits**: Max 3 retries per action
- **Confidence threshold**: Actions below 0.8 confidence rejected

## Troubleshooting

### Ollama not responding
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
systemctl restart ollama
# or
ollama serve
```

### Out of VRAM
- Reduce `MAX_TABS` in config
- Close unused tabs during long sessions
- Restart Ollama to clear GPU memory

### Playwright timeout
- Increase `page_timeout_ms` in config
- Check network connectivity
- Some sites may block automated access

## License

MIT License
