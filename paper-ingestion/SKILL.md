---
name: paper-ingestion
description: Ingest PDF research papers and convert to Markdown for AI-native analysis. Use when user wants to read, analyze, or process a PDF paper, or provides a PDF URL/path. Uses MinerU (GPU) by default, docling as fallback.
---

# Paper Ingestion Tool

Convert PDF research papers to Markdown with image extraction, organized for AI-native analysis.

## Quick Reference

```bash
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

```bash
# Start server with uv + local mineru-fork
CUDA_VISIBLE_DEVICES=0 \
  uv run --python mineru-fork/.venv \
  python -m mineru.cli.fast_api --host 127.0.0.1 --port 8000
```

## Engine Selection

| Scenario | Engine | Notes |
|----------|--------|-------|
| Default (highest quality) | `mineru` | GPU-accelerated, excellent math/tables |
| Fallback (fast, no GPU) | `docling` | Lower quality, good for quick previews |

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
