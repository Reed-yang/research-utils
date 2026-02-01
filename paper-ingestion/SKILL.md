---
name: paper-ingestion
description: Ingest PDF research papers and convert to Markdown for AI-native analysis. Use when user wants to read, analyze, or process a PDF paper, or provides a PDF URL/path. Supports docling (fast, tables/figures) and nougat (heavy math) engines.
---

# Paper Ingestion Tool

Convert PDF research papers to Markdown with image extraction, organized for AI-native analysis.

## Quick Reference

```bash
# From local file (default: docling engine)
uv run scripts/ingest_paper.py /path/to/paper.pdf

# From URL
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2401.12345.pdf"

# Heavy math papers (nougat engine)
uv run scripts/ingest_paper.py paper.pdf --engine nougat

# Custom output directory
uv run scripts/ingest_paper.py paper.pdf --output-dir /path/to/readings
```

## Engine Selection

| Scenario | Engine | Notes |
|----------|--------|-------|
| General papers, tables, figures | `docling` (default) | Fast, extracts images |
| Heavy LaTeX math equations | `nougat` | GPU-intensive, slow |
| Math garbled with docling | `nougat` | Better equation rendering |

## Output Structure

Files organized at `{cwd}/{YYYYMMDD}-{Sanitized_Title}/`:

```
20260131-DeepSeek_V3_Technical_Report/
  reference.pdf    # Original PDF
  full_text.md     # Markdown with YAML frontmatter
  notes.md         # Empty notes file
  assets/          # Extracted images (docling only)
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
conversion_engine: docling
tags:
  - paper
  - inbox
aliases: []
---
```

## JSON Output

**Success:**
```json
{"status": "success", "markdown_path": "...", "title": "...", "date": "2026-01-31", "paper_dir": "...", "engine_used": "docling"}
```

**Error:**
```json
{"status": "error", "message": "...", "suggestion": "..."}
```

## Error Handling

| Error | Action |
|-------|--------|
| Duplicate detected | Remove existing folder or use `--force` |
| Nougat OOM | Use `--engine docling` |
| Math garbled | Re-run with `--engine nougat` |
| Download failed | Check URL is accessible |

## Image Handling

- **Docling**: Extracts images to `assets/` folder (default 4x resolution)
- **Markdown references**: `![Fig1](./assets/image_001.png)` (relative paths)
- **Syncthing compatible**: Small image files sync across devices

## Math Formatting

- Inline and display math are normalized to LaTeX using `$...$` / `$$...$$`
