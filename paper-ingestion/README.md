# Paper Ingestion & Analysis Toolkit

A high-performance toolkit for converting PDF research papers into AI-native Markdown formats. Designed for researchers who need to analyze papers with LLMs, preserving complex layouts, mathematical formulas, and extracting figures as standalone assets.

## 🚀 Key Features

- **High-Fidelity Conversion**: Uses [MinerU](https://github.com/opendatalab/MinerU) (hybrid-auto-engine) for state-of-the-art PDF parsing.
- **10x Efficiency Boost**: Supports persistent API Server mode to eliminate model loading overhead (processing time reduced from ~130s to ~12s).
- **Asset Extraction**: Automatically extracts images and figures into a local `assets/` directory.
- **Math Support**: High-quality LaTeX formula recognition ($...$ and $$...$$).
- **Smart Organization**: Automatically renames and organizes output folders by date and paper title.

## 📦 Installation

Ensure you have `uv` installed for dependency management.

```bash
# Install dependencies
uv pip install -U "mineru[all]" requests psutil
```

## 🛠️ Usage Guide

### 1. ⚡ Recommended: API Server Mode (Fastest)

For the best performance (10x speedup), run a persistent MinerU server. This loads the heavy ML models once into GPU memory, allowing subsequent papers to be processed in seconds.

**Step 1: Start the Server (Background)**

```bash
# Start server on GPU 0 (adjust device as needed)
CUDA_VISIBLE_DEVICES=0 MINERU_HYBRID_BATCH_RATIO=16 \
    nohup mineru-api --host 127.0.0.1 --port 8000 > mineru.log 2>&1 &
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
# Useful for one-off tasks without setting up a server
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2512.05905"
```

### 3. Benchmarking

Use the included benchmark suite to test system performance and verify speedups.

```bash
uv run scripts/benchmark.py --gpus 0,1 --output-dir ./benchmarks
```

## 📂 Output Structure

Papers are organized in timestamps folders to keep your research clean:

```text
./20260202-My_Research_Paper_Title/
├── reference.pdf       # Original PDF
├── full_text.md        # Converted Markdown (with YAML frontmatter)
├── notes.md            # Empty notes file for your analysis
└── assets/             # Extracted figures and images
    ├── image_001.png
    └── image_002.png
```

## 📊 Performance Analysis

Based on benchmarks (H100 GPU):

| Mode | Processing Time (15 pages) | Throughput | Notes |
|------|---------------------------|------------|-------|
| **API Server** | **~12s** | **~1.25 pg/s** | **Recommended.** Models loaded once. |
| CLI Mode | ~130s | ~0.11 pg/s | High overhead per run. |

## ⚙️ Configuration

You can tweak performance via environment variables:

- `MINERU_HYBRID_BATCH_RATIO`: Controls internal batch size (Default: 16). Lower to 8 if encountering OOM.
- `MINERU_API_HOST` / `MINERU_API_PORT`: Configure API endpoint (Default: 127.0.0.1:8000).

---

**Developed for AI-Native Research Workflows.**
