---
name: subtitle-translate
description: Translate SRT subtitle files to Chinese (or other languages) using LLM while preserving timestamps and structure. Supports multiple LLM backends with automatic integrity checks.
---

# Subtitle Translation Tool

Translate SRT subtitle files to Chinese while preserving all timestamps and formatting.

## Quick Reference

```bash
# Translate using TensorBlock (default)
uv run scripts/translate_subtitle.py /path/to/subtitle.srt

# Translate using DeepSeek
uv run scripts/translate_subtitle.py /path/to/subtitle.srt --backend deepseek

# Specify target language (default: Chinese)
uv run scripts/translate_subtitle.py /path/to/subtitle.srt --target-lang Japanese
```

## Backend Selection

| Backend | API | Notes |
|---------|-----|-------|
| `tensorblock` (default) | TensorBlock Forge API | Default backend |
| `deepseek` | DeepSeek API | High quality, fast |

## What Gets Translated

| Element | Translated | Notes |
|---------|------------|-------|
| Subtitle text | Yes | Main content |
| Sequence numbers | No | Preserved exactly |
| Timestamps | No | Preserved exactly |
| Speaker indicators | No | Preserved if present |

## Output Structure

The translated file is saved alongside the original:

```
subtitle.srt           # Original SRT
subtitle_zh.srt        # Translated SRT
```

## JSON Output

**Success:**
```json
{"status": "success", "output_path": "...", "backend": "tensorblock", "target_lang": "Chinese", "entries_translated": 150, "elapsed_seconds": 12.3}
```

**Error:**
```json
{"status": "error", "message": "..."}
```

## Error Handling

| Error | Action |
|-------|--------|
| API rate limit | Automatic retry with backoff |
| Missing entries in output | Automatic retry with smaller chunks |
| Network error | Retry up to 3 times |
