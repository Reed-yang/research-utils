# Paper Ingestion & Analysis Toolkit

A high-performance toolkit for converting PDF research papers into AI-native Markdown formats. Designed for researchers who need to analyze papers with LLMs, preserving complex layouts, mathematical formulas, and extracting figures as standalone assets.

## Key Features

- **High-Fidelity Conversion**: Uses [MinerU](https://github.com/opendatalab/MinerU) (hybrid-auto-engine) for state-of-the-art PDF parsing.
- **10x Efficiency Boost**: Supports persistent API Server mode to eliminate model loading overhead (processing time reduced from ~130s to ~12s).
- **Asset Extraction**: Automatically extracts images and figures into a local `assets/` directory.
- **Math Support**: High-quality LaTeX formula recognition ($...$ and $$...$$).
- **Smart Organization**: Automatically renames and organizes output folders by date and paper title.
- **Modular Dependencies**: Engine dependencies are optional — install only what you need.

## Installation

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
cd paper-ingestion

# Base only (pypdf, pillow, requests)
uv sync

# Add MinerU engine (GPU-accelerated, from local mineru-fork)
uv sync --extra mineru

# Add Docling engine (CPU-friendly fallback)
uv sync --extra docling

# Install all engines
uv sync --all-extras
```

### Dependency Groups

| Group | Command | What's installed |
|-------|---------|------------------|
| Base | `uv sync` | `pypdf`, `pillow`, `requests` (~7 packages) |
| MinerU | `uv sync --extra mineru` | + `mineru[pipeline,api]` from local fork (~134 packages, includes torch + CUDA) |
| Docling | `uv sync --extra docling` | + `docling`, `torch` (~70 packages) |
| All | `uv sync --all-extras` | Everything (~199 packages) |

> The `mineru` extra installs from the local `mineru-fork/` subtree as an editable dependency. Any changes to the fork code take effect immediately.

## Usage Guide

### 1. Recommended: API Server Mode (Fastest)

For the best performance (10x speedup), run a persistent MinerU server. This loads the heavy ML models once into GPU memory, allowing subsequent papers to be processed in seconds.

**Step 1: Start the Server (Background)**

```bash
# Requires: uv sync --extra mineru
CUDA_VISIBLE_DEVICES=0 uv run mineru-api --host 127.0.0.1 --port 8000
```

**Step 2: Ingest Papers**

The script automatically detects if the server is running at `127.0.0.1:8000` and uses it.

```bash
# Process a paper from URL
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2512.05905"

# Process a local file
uv run scripts/ingest_paper.py papers/my_paper.pdf
```

### 2. Standard CLI Mode (Fallback)

If the server is not running, the script falls back to CLI mode. This handles model loading per-run (slower, ~1-2 mins per paper).

```bash
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2512.05905"
```

### 3. Docling Engine (No GPU)

```bash
# Requires: uv sync --extra docling
uv run scripts/ingest_paper.py paper.pdf --engine docling
```

## Output Structure

Papers are organized in timestamped folders:

```text
./20260202-My_Research_Paper_Title/
├── reference.pdf       # Original PDF
├── full_text.md        # Converted Markdown (with YAML frontmatter)
├── notes.md            # Empty notes file for your analysis
└── assets/             # Extracted figures and images
    ├── image_001.webp
    └── image_002.webp
```

## Performance

Based on benchmarks (H100 GPU, 15-page paper):

| Mode | Processing Time | Throughput | Notes |
|------|----------------|------------|-------|
| **API Server** | **~12s** | **~1.25 pg/s** | Recommended. Models loaded once. |
| CLI Mode | ~130s | ~0.11 pg/s | High overhead per run. |

## Configuration

Environment variables:

- `MINERU_API_HOST` / `MINERU_API_PORT`: API endpoint (default: `127.0.0.1:8000`)
- `MINERU_HYBRID_BATCH_RATIO`: Internal batch size (default: 16). Lower to 8 if OOM.
- `CUDA_VISIBLE_DEVICES`: GPU selection for server or CLI mode.

---

**Developed for AI-Native Research Workflows.**
