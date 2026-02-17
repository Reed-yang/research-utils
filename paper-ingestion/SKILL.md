---
name: paper-ingestion
description: Ingest PDF research papers and convert to Markdown for AI-native analysis. Use when user wants to read, analyze, or process a PDF paper, or provides a PDF URL/path. Uses GLM-OCR (cloud) by default, MinerU (GPU) or docling as alternatives.
---

# Paper Ingestion Tool

Convert PDF research papers to Markdown with image extraction, organized for AI-native analysis.

## Quick Reference

```bash
cd paper-ingestion

# From local file (default: glm-ocr cloud engine)
uv run scripts/ingest_paper.py /path/to/paper.pdf

# From URL
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2401.12345.pdf"

# MinerU engine (highest quality, requires GPU)
uv run scripts/ingest_paper.py paper.pdf --engine mineru

# Docling engine (fast, no GPU, lower quality)
uv run scripts/ingest_paper.py paper.pdf --engine docling

# Custom output directory
uv run scripts/ingest_paper.py paper.pdf --output-dir /path/to/readings
```

## MinerU API Server (Recommended)

For the best performance, run a persistent MinerU API server. This loads models once into GPU memory, giving ~10x speedup on subsequent papers.

```bash
cd paper-ingestion

# Start server (requires --extra mineru)
CUDA_VISIBLE_DEVICES=0 uv run mineru-api --host 127.0.0.1 --port 8000
```

The ingestion script auto-detects the server at `127.0.0.1:8000` and uses it when available. If the server is not running, it falls back to the `mineru` CLI (slower, loads models per-run).

## Engine Selection

| Scenario | Engine | Extra needed | Notes |
|----------|--------|--------------|-------|
| Default (cloud, no GPU) | `glm-ocr` | None (base only) | Cloud API, requires `GLM_API_KEY`, max 100 pages |
| Highest quality (GPU) | `mineru` | `--extra mineru` | GPU-accelerated, excellent math/tables |
| Fallback (fast, no GPU) | `docling` | `--extra docling` | Lower quality, good for quick previews |

## Output Structure

Files organized at `{cwd}/{YYYYMMDD}-{Sanitized_Title}/`:

```
20260131-DeepSeek_V3_Technical_Report/
  reference.pdf    # Original PDF
  full_text.md     # Markdown with YAML frontmatter
  notes.md         # Empty notes file
  assets/          # Extracted images
    image_001.webp
    image_002.webp
```

**Naming rules:**
- Timestamped prefix: `YYYYMMDD-`
- Title source: Use detected paper title after conversion (not URL string)
- Windows-safe: No `:?/\*<>|"` characters
- Duplicate check: Aborts if same title exists (ignoring date)

## YAML Frontmatter

```yaml
---
title: "Paper Title"
date_ingested: 2026-01-31
source_pdf: reference.pdf
conversion_engine: mineru
tags:
  - paper
  - inbox
aliases: []
---
```

## JSON Output

**Success:**
```json
{"status": "success", "markdown_path": "...", "title": "...", "date": "2026-01-31", "paper_dir": "...", "engine_used": "mineru"}
```

**Error:**
```json
{"status": "error", "message": "...", "suggestion": "..."}
```

## Cloud Engine Configuration (GLM-OCR)

GLM-OCR uses the Zhipu AI cloud API. Set your API credentials:

```bash
# Option 1: Environment variables
export GLM_API_ID=your_id_here
export GLM_API_KEY=your_key_here

# Option 2: .env file
cp paper-ingestion/.env.template paper-ingestion/.env
# Edit .env and add your credentials
```

Get an API key at https://open.bigmodel.cn

**Limits**: PDF <= 50MB, max 100 pages per document.

## Error Handling

| Error | Action |
|-------|--------|
| Duplicate detected | Remove existing folder or use `--force` |
| MinerU timeout | Try `--engine docling` or `--engine glm-ocr` |
| Download failed | Check URL is accessible |
| GLM-OCR API key missing | Set `GLM_API_ID` and `GLM_API_KEY` in `.env` or environment |
| GLM-OCR timeout | Check network or try a local engine |
| GLM-OCR rate limited | Wait and retry (automatic), or reduce request frequency |

## Image Handling

- **All engines**: Extract images to `assets/` folder
- **Markdown references**: `![Fig1](./assets/image_001.webp)` (relative paths)
- **Syncthing compatible**: Small image files sync across devices

## Math Formatting

- Inline and display math are normalized to LaTeX using `$...$` / `$$...$$`

## Environment Setup (if there is no env yet)

Dependencies are managed via `pyproject.toml` with optional extras. Install only what you need:

```bash
cd paper-ingestion

# Base only (pypdf, pillow, requests — enough for mineru API mode)
uv sync

# MinerU engine (local mineru-fork, GPU-accelerated)
uv sync --extra mineru

# Docling engine (CPU-friendly fallback)
uv sync --extra docling

# Everything
uv sync --all-extras
```

> **Note:** `mineru` is installed as an editable dependency from the local `mineru-fork/` subtree. Changes to the fork take effect immediately without reinstalling.