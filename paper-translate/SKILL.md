---
name: paper-translate
description: Translate research paper markdown files to Chinese while preserving LaTeX, code, images, and formatting. Use after paper-ingestion to translate an ingested paper.
---

# Paper Translation Tool

Translate research paper markdown files (from paper-ingestion) to Chinese while preserving all formatting.

## Environment Setup

```bash
cd paper-translate
uv sync
```

API keys are loaded from `paper-translate/.env`:

```env
DEEPSEEK_API_KEY=sk-...
TENSORBLOCK_API_KEY=tb-...
```

## Quick Reference

```bash
cd paper-translate

# Translate using TensorBlock (default)
uv run scripts/translate_paper.py /path/to/paper_folder/full_text.md

# Translate using DeepSeek
uv run scripts/translate_paper.py /path/to/paper_folder/full_text.md --backend deepseek

# Specify target language (default: Chinese)
uv run scripts/translate_paper.py /path/to/paper_folder/full_text.md --target-lang Japanese
```

## Backend Selection

| Backend | API | Notes |
|---------|-----|-------|
| `tensorblock` (default) | TensorBlock Forge API | Alternative backend |
| `deepseek` | DeepSeek API | High quality, fast |

## What Gets Translated

| Element | Translated | Notes |
|---------|------------|-------|
| Body text | Yes | Main content |
| Headings | Yes | All heading levels |
| YAML frontmatter | No | Preserved, language tag added |
| LaTeX formulas | No | `$...$`, `$$...$$` preserved |
| Code blocks | No | Fenced and inline code preserved |
| Image references | No | Paths preserved exactly |
| Link URLs | No | URLs preserved, text translated |
| HTML tables | No | Structure preserved |
| Citations | No | `[1]`, `[17]` preserved |

## Output Structure

The translated file is saved alongside the original:

```
20260202-Paper_Title/
  reference.pdf        # Original PDF
  full_text.md         # Original markdown
  full_text_ch.md      # Translated markdown
  notes.md             # Notes file
  assets/              # Images (unchanged)
```

## JSON Output

**Success:**
```json
{"status": "success", "output_path": "...", "backend": "tensorblock", "target_lang": "Chinese"}
```

**Error:**
```json
{"status": "error", "message": "..."}
```

## Error Handling

| Error | Action |
|-------|--------|
| API rate limit | Automatic retry with backoff |
| Token limit exceeded | Text automatically chunked, with sub-chunk retry on marker loss |
| Network error | Retry up to 3 times |
