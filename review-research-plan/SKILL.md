---
name: review-research-plan
description: >
  Critically review and iteratively polish research plans and roadmaps. Use this skill whenever
  the user asks to review, critique, validate, stress-test, or improve a research plan — even
  informally ("does this make sense", "what's wrong with my idea", "poke holes in this plan").
  Also trigger when the user wants to turn a rough brainstorm into a rigorous plan. Supports
  three commands: /quick-review (fast single-pass), /full-review (comprehensive multi-phase),
  and /re-review (iterative follow-up on a previously reviewed plan).
---

# Review Research Plan

Catch fatal flaws, challenge assumptions, and iteratively polish rough research ideas into
executable plans. Deliberately adversarial — seeks to falsify, not confirm.

## Architecture

```
SKILL.md (you are here — routing & maturity detection only)
├── commands/
│   ├── quick-review.md   ← Fast single-pass, ~5 min
│   ├── full-review.md    ← Comprehensive multi-phase, with pause point
│   └── re-review.md      ← Iterative follow-up on previously reviewed plans
└── references/
    ├── adversarial.md    ← Falsification & steelman protocols
    ├── novelty.md        ← Web search novelty verification procedure
    ├── feasibility.md    ← Resource & feasibility audit checklist
    └── report-format.md  ← Output templates & severity definitions
```

**Read only what you need.** Do NOT load all reference files upfront. Load a reference file
only when you reach the phase that requires it.

## Command Routing

Determine which command to run:

| User says | Command | What to read |
|-----------|---------|-------------|
| "review my plan", "poke holes", "what's wrong with this" | **quick-review** | `commands/quick-review.md` |
| "full review", "comprehensive review", "stress test this plan" | **full-review** | `commands/full-review.md` |
| "review again", "re-review", "check if I fixed the issues" | **re-review** | `commands/re-review.md` |
| Ambiguous or first time using the skill | Default to **quick-review** for early-stage plans, **full-review** for mid/late-stage plans | Assess maturity first (see below), then route |

If the user explicitly names a command (e.g., "/full-review"), use it directly.
Otherwise, assess maturity first and route accordingly.

## Context Detection (Run Alongside Maturity)

If the user provides a codebase path, paper directory, or documentation links, note them:

> **Context**: Codebase at `[path]`, papers at `[path]`
> Technical claims will be grounded against code and literature where possible.

This context is used during adversarial review — technical claims about existing systems
should be verified against actual implementations, not assessed through reasoning alone.
If no external context is provided, the review proceeds as pure plan analysis.

## Maturity Detection (Always Run First)

Before routing to a command, quickly assess the plan's maturity. This takes ~30 seconds of
reading — do not load any reference files for this step.

Scan the plan for four signals:

1. **Structural completeness** — Does it have: problem statement, approach, evaluation plan,
   related work, timeline? Count how many are present and substantive.
2. **Claim specificity** — Are contributions vague ("use diffusion for X") or precise
   ("inject identity embeddings at cross-attention layers with contrastive loss")?
3. **Empirical grounding** — Any preliminary results, pilot experiments, or proof-of-concept?
4. **Risk awareness** — Does it mention limitations, failure modes, or alternatives?

Classify:

| Level | Profile | Default Command |
|-------|---------|-----------------|
| **Early** | ≤3 elements, vague claims, no experiments, no risks | quick-review (constructive-first) |
| **Mid** | 4-6 elements, some specificity, limited evidence | full-review (balanced) |
| **Late** | 7+ elements, specific, some experiments, risk-aware | full-review (aggressive) |

State the detected level in your first line of output so the user can override:
> **Maturity: Mid-stage** — routing to full-review (balanced). Say "switch to quick/aggressive"
> to override.

## Severity Definitions (Used Across All Commands)

- 🔴 **FATAL** — Direction needs reconsideration (e.g., core novelty already published)
- 🟠 **SERIOUS** — Reviewer would likely reject if unaddressed
- 🟡 **CONCERN** — Worth noting, won't determine acceptance alone

## Key Principles (Apply to All Commands)

1. Every critique must come with at least one path forward.
2. Match the language of the input plan (Chinese plan → Chinese review).
3. Do not manufacture problems when the plan is strong — say so and move on.
4. Flag your own uncertainty honestly rather than asserting false confidence.
5. When the user pushes back with valid arguments, downgrade or close the finding.
