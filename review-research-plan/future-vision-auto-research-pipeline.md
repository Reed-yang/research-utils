# Future Vision: End-to-End Research Plan Iteration with Auto Paper Discovery

> This document captures a future direction for integrating review-research-plan with the
> broader research-utils ecosystem. NOT for current implementation — reference only.

## Core Idea

A closed-loop system for iterative research plan polishing that automatically discovers,
ingests, and integrates relevant papers as part of the review process:

```
brainstorm → plan draft → review → [auto-discover related work] → revise → re-review → ...
```

## Envisioned Pipeline

### Phase 1: During Novelty Scan
When review-research-plan's novelty scan identifies claims that need verification:

1. **Auto-search**: WebSearch for recent papers matching the claim's keywords
2. **Triage**: Rank results by relevance, filter to top-3 most threatening/supporting
3. **Auto-ingest**: Dispatch subagent to run paper-ingestion on discovered PDFs
4. **Auto-summarize**: Run paper-summary on ingested papers
5. **Integrate**: Feed summaries back into the review as evidence for/against novelty

### Phase 2: Deep Research Mode
A new command `/deep-review` that goes beyond `/full-review`:

- For each proposed direction in the plan, search for the 3-5 most relevant recent papers
- Ingest and summarize them into a local paper library
- Cross-reference claims against actual paper content (not just abstracts)
- Generate a "literature landscape" section showing where the plan sits

### Phase 3: Continuous Monitoring
- Track arxiv/OpenReview for new papers matching the plan's keywords
- Alert when a new paper threatens novelty or provides supporting evidence
- Integrate with re-review to update the plan's related work section

## Implementation Approach

- **Subagent-driven**: Each paper discovery + ingestion + summary runs as an independent
  subagent, parallelized for speed
- **Paper library**: Ingested papers stored in a shared `papers/` directory (or symlinked
  from agent-readings) with standardized metadata
- **Skill chaining**: review-research-plan orchestrates paper-ingestion → paper-summary
  as sub-workflows during the novelty scan phase

## Integration Points

```
review-research-plan (orchestrator)
├── WebSearch (paper discovery)
├── paper-ingestion (PDF → Markdown)
├── paper-summary (structured summary)
├── paper-translate (optional, for Chinese researchers)
└── paper-validator (optional, assess quality of discovered papers)
```

## Key Design Questions (for future brainstorming)

1. How to avoid ingesting too many papers (token/time budget)?
2. How to handle papers behind paywalls?
3. How to maintain the paper library across sessions?
4. Should discovered papers be automatically added to the plan's related work section?
5. How to distinguish between "threatening" papers (novelty risk) and "supporting" papers?
