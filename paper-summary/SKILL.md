---
name: summary
description: This skill should be used when providing concise summaries of research paper text. Use for quickly understanding the core content, arguments, and contributions of paper sections targeting top-tier computer science conferences.
---

# Paper Summary Skill

Generate structured summaries of research papers with keyword extraction. Reads ingested paper markdown, produces full-paper and section-level summaries, and backfills extracted keywords into YAML frontmatter tags.

## When to Use This Skill

- Summarizing a research paper after ingestion for quick understanding
- Extracting core arguments, contributions, and key findings
- Generating keywords for paper indexing and retrieval
- Preparing summaries for paper reviews or presentations
- As a follow-up step after `paper-ingestion`

## Pipeline Context

Typical workflow: `paper-ingestion` → **`summary`** → `paper-translate` / `paper-validator`

Input: A paper directory produced by `paper-ingestion`, containing `full_text.md` (and optionally `full_text_zh.md` after translation).

## Default Behavior

By default, produce **Full Paper Summary** only — a structured overview of the entire paper (concise, 1-3 sentences per section).

**Section-by-Section Summary** is available on request but not produced by default.

## Language Settings

- **Default output language**: Chinese (中文)
- Can be specified as English by the user
- **Chinese mode rules**:
  - Technical terms and proper nouns remain in English (e.g., "Transformer", "attention 机制", "MoE 架构")
  - Paper titles remain in original language
- **Keywords are always in English** regardless of output language

## Workflow

Follow these steps systematically:

### Step 1: Locate Paper Files

- User provides a paper directory path or `full_text.md` path
- Confirm the directory contains `full_text.md`
- Check if `full_text_zh.md` exists (for keyword backfill only)

### Step 2: Read Paper Content

- Use the **Read** tool to load `full_text.md` (the English original)
- Only read `full_text.md` — do NOT read `full_text_zh.md` for summarization

### Step 3: Generate Summary

- Produce Full Paper Summary and Section-by-Section Summary (see Output Format below)
- Follow the Language Settings above

### Step 4: Extract Keywords

- Extract at least 4 keywords from the paper (typically 4-8)
- Follow the Keyword Extraction Guidelines below
- Include keywords in the summary output

### Step 5: Write Summary to notes.md

- Use the **Write** tool to save the complete summary output (Full Paper Summary + Section Summary) into `notes.md` in the same paper directory
- `notes.md` is created empty by `paper-ingestion` — this step fills it with the summary content

### Step 6: Backfill Keywords to Tags

- Use the **Edit** tool to append keywords to the `tags:` list in `full_text.md` YAML frontmatter
- If `full_text_zh.md` exists, append the same keywords to its `tags:` list
- See Backfill Rules below for details

## Output Format — Full Paper Summary

```
# Paper Summary: {Title}

## Problem
[Core problem being addressed, 1-3 sentences]

## Method
[Key approach and techniques, 2-5 sentences]

## Key Results
[Main experimental findings, 2-4 sentences]

## Contributions
- [Contribution 1]
- [Contribution 2]
- ...

## Limitations
[Main limitations and shortcomings, 1-3 sentences]

## Keywords
keyword1, keyword2, keyword3, keyword4, ...
```

## Output Format — Section Summary

Summarize each major section in order of appearance:

```
# Section Summaries

### {Section Name}
[2-5 sentences capturing the essential content of this section]

### {Section Name}
[2-5 sentences]

...
```

## Keyword Extraction Guidelines

- **Quantity**: At least 4, typically 4-8
- **Coverage levels** (include a mix):
  - Domain-level (broad): e.g., `computer-vision`, `NLP`
  - Method-level: e.g., `diffusion-model`, `RL`
  - Technique-level (specific): e.g., `flash-attention`, `RoPE`
  - Task/application-level: e.g., `image-generation`, `code-completion`
- **Abbreviations**: Use widely recognized abbreviations directly (LLM, MoE, GAN, RL, NLP, CV, etc.); keep full names for uncommon or domain-specific terms — do not abbreviate arbitrarily
- **Format**: English, lowercase (capitalize proper nouns and abbreviations). **Tags MUST NOT contain spaces — use hyphens (`-`) to connect multi-word tags** (e.g., `diffusion-transformer`, `reward-hacking`, `text-to-video`). This is required for Obsidian tag compatibility.
- **Priority**: Paper's own keywords > key concepts from abstract > core terms from full text

## Keyword Backfill Rules

"Backfill" means appending extracted keywords into the `tags:` list in the YAML frontmatter of `full_text.md` (and `full_text_zh.md` if it exists). Keywords are inserted before the `aliases:` line.

**Before backfill:**
```yaml
tags:
  - paper
aliases: []
```

**After backfill:**
```yaml
tags:
  - paper
  - LLM
  - MoE
  - inference-optimization
  - transformer-architecture
aliases: []
```

**Rules:**
- **Backfill target**: The `tags:` block inside the YAML frontmatter of `full_text.md` (and `full_text_zh.md` if it exists)
- **No spaces in tags**: All tags MUST use hyphens (`-`) instead of spaces (e.g., `diffusion-transformer`, NOT `diffusion transformer`). Tags containing spaces will not be recognized by Obsidian.
- Use Edit tool to replace the `tags:` block (from `tags:` through the line before `aliases:`)
- If `tags:` already contains entries beyond `paper`, assume keywords were already backfilled — skip
- If `full_text_zh.md` does not exist, skip it silently
- Do not modify any content outside YAML frontmatter

## Summary Requirements

- **Conciseness**: Focus on core content, eliminate unnecessary details
- **Technical accuracy**: Preserve precise terminology, do not oversimplify
- **Clarity**: Use clear, precise, self-contained language
- **Content focus**: Highlight key contributions, novel aspects, and the "so what?"

## Important Constraints

- Do not introduce information not present in the source text
- Do not editorialize or add opinions
- Maintain technical precision appropriate for academic audience
- Only backfill keywords when producing a full paper summary (not for section-only requests)
- Preserve all existing YAML frontmatter fields when editing
