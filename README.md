# research-utils

An AI-native toolkit for academic research — a collection of composable agentic skills covering the full paper workflow from ingestion, reading, translation, summarization to peer review.

## Motivation

In academic research, acquiring papers, deep reading, managing knowledge, and accumulating insights are high-frequency, repetitive tasks. Traditional workflows rely on manual operations and fragmented tools, making it hard to form a continuous knowledge pipeline. This project packages these steps into composable agentic skills driven by Claude Code, so researchers can focus on thinking itself.

## Workflow

```
PDF / URL
  │
  ▼
paper-ingestion ──→ full_text.md + assets/
  │
  ├──→ summary ──→ notes.md (structured summary + keyword backfill)
  ├──→ paper-translate ──→ full_text_ch.md (Chinese translation)
  └──→ paper-validator ──→ peer-review style feedback
```

**subtitle-translate** is also included for translating academic video/lecture subtitles.

## Skills Overview

| Skill | Purpose | Engine / Backend |
|-------|---------|------------------|
| **paper-ingestion** | PDF → Markdown with image extraction and YAML frontmatter | GLM-OCR (cloud) / MinerU (GPU) / Docling |
| **paper-translate** | Translate paper markdown, preserving LaTeX, code, and image refs | DeepSeek / TensorBlock |
| **paper-summary** | Structured summary + keyword extraction + tag backfill | Claude (built-in skill) |
| **paper-validator** | Peer-review style critique targeting top-tier conferences | Claude (built-in skill) |
| **subtitle-translate** | SRT subtitle translation, preserving timestamps | DeepSeek / TensorBlock |
| **claude-sync** | Cross-device ~/.claude sync with interactive conflict resolution | claude-sync CLI (built-in skill) |

## Why DeepSeek

For translation and OCR-related tasks, **DeepSeek** is the recommended backend:

- **Extremely low cost** — among the cheapest LLM APIs available, ideal for batch-processing large volumes of papers
- **High quality** — strong accuracy on academic text translation with proper handling of technical terminology
- **Fast** — low response latency, smooth experience for concurrent long-document translation

For OCR, the default is Zhipu GLM-OCR cloud API — also chosen for its extremely low cost (processing 100 PDFs costs less than \$1).

## Getting Started

### 1. Configure API Keys

```bash
cp .env.template .env
# Edit .env and fill in your keys
```

`.env` supports two-level loading: **root-level takes priority**, with per-skill `.env` files as fallback. If you only use a few skills, you can configure keys in the corresponding subdirectory only.

### 2. Ingest a Paper

```bash
cd paper-ingestion
uv run scripts/ingest_paper.py "https://arxiv.org/pdf/2401.12345.pdf"
```

### 3. Summarize / Translate / Review

Invoke the corresponding skill directly in Claude Code:

```
/summary path/to/paper_folder
/paper-translate path/to/paper_folder/full_text.md
/paper-validator path/to/paper.tex
```

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) — each skill self-manages dependencies via `uv run`
- [Claude Code](https://claude.com/claude-code) — drives the agentic skills

Each skill has independent dependencies. Install only what you need — see `SKILL.md` in each subdirectory for details.

## Project Structure

```
research-utils/
├── .env.template          # Root-level API key template (shared across all skills)
├── paper-ingestion/       # PDF → Markdown conversion
│   ├── scripts/
│   ├── mineru-fork/       # Local MinerU fork (optional)
│   └── SKILL.md
├── paper-translate/       # Paper translation
│   ├── scripts/
│   └── SKILL.md
├── paper-summary/         # Structured summarization
│   └── SKILL.md
├── paper-validator/       # Paper review
│   └── SKILL.md
├── subtitle-translate/    # Subtitle translation
│   ├── scripts/
│   └── SKILL.md
└── claude-sync/           # Cross-device sync with conflict resolution
    └── SKILL.md
```
