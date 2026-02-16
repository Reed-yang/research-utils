---
name: paper-ingestion
description: Ingest PDF research papers and convert to Markdown for AI-native analysis. Use when user wants to read, analyze, or process a PDF paper, or provides a PDF URL/path. Uses MinerU (GPU) by default, docling as fallback.
---

# Paper Ingestion Tool

Convert PDF research papers to Markdown with image extraction, organized for AI-native analysis.

## Quick Reference

```bash
cd paper-ingestion

# From local file (default: mineru engine)
uv run scripts/ingest_paper.py /path/to/paper.pdf

# From URL
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2401.12345.pdf"

# Fallback engine (docling, fast but lower quality)
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
| Default (highest quality) | `mineru` | `--extra mineru` | GPU-accelerated, excellent math/tables |
| Fallback (fast, no GPU) | `docling` | `--extra docling` | Lower quality, good for quick previews |

## Output Structure

Files organized at `{cwd}/{YYYYMMDD}-{Sanitized_Title}/`:

```
20260131-DeepSeek_V3_Technical_Report/
  reference.pdf    # Original PDF
  full_text.md     # Markdown with YAML frontmatter
  notes.md         # Empty notes file
  assets/          # Extracted images
    image_001.png
    image_002.png
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

## Error Handling

| Error | Action |
|-------|--------|
| Duplicate detected | Remove existing folder or use `--force` |
| MinerU timeout | Try `--engine docling` |
| Download failed | Check URL is accessible |

## Image Handling

- **Both engines**: Extract images to `assets/` folder
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