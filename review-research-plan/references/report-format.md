# Report Format & Templates

## Diagnosis Report Template

```markdown
## Review Diagnosis — [Plan Title]

**Maturity**: [Early/Mid/Late]-stage | **Intensity**: [Constructive/Balanced/Aggressive]
**Date**: [date] | **Novelty Scan**: [Performed / Skipped]

### Summary
[2-3 sentences: strongest aspect, most critical concern, overall viability assessment]

### Findings

#### 🔴 Fatal Issues
- **[F1] [Title]**: [Description. Why it's fatal. Evidence.]
- ...

#### 🟠 Serious Issues
- **[S1] [Title]**: [Description. Impact on reviewability.]
- ...

#### 🟡 Concerns
- **[C1] [Title]**: [Description.]
- ...

### Novelty Assessment
_(only if novelty scan was performed)_
- Claim 1: [🟢/🟡/🔴] [one-line assessment]
- Claim 2: [🟢/🟡/🔴] [one-line assessment]

### Feasibility Snapshot
- Estimated total GPU-hours: [range]
- Time-to-first-signal: [estimate]
- Critical path bottleneck: [what]
- Overall feasibility: [Feasible / Tight / At risk / Infeasible]
```

## Prescription Template

For each finding:

```markdown
### [ID] [Title]

**Problem**: [1-2 sentence restatement]

**Option A — [Label]**: [What to change, add, or remove]
  - *Tradeoff*: [cost or risk]

**Option B — [Label]** _(if applicable)_: [Alternative approach]
  - *Tradeoff*: [cost or risk]

**Recommended**: [A or B, brief reasoning]
```

- 🔴 Fatal: Always provide 2+ options, including possible pivot.
- 🟠 Serious: 1-2 options, at least one achievable within current timeline.
- 🟡 Concern: Brief note suffices, no full option analysis.

## Revised Draft Markers

When generating a revised plan, mark changes inline:

- `[NEW]` — Newly added content
- `[MODIFIED]` — Changed from original (with brief note of what changed)
- `[REMOVED: reason]` — Content deleted, with explanation

Do not rewrite sections that don't need changing. Preserve original voice and style.

## Review Log Format

Append this to the end of the plan document after each review round:

```markdown
---

## Review Log

### Round [N] — [Date]
**Maturity**: [Level] | **Intensity**: [Level] | **Novelty Scan**: [Yes/No]

| ID | Severity | Title | Status |
|----|----------|-------|--------|
| F1 | 🔴 | [title] | open / addressed / wont-fix |
| S1 | 🟠 | [title] | open / addressed / wont-fix |
| C1 | 🟡 | [title] | open / addressed / wont-fix |

**Summary**: [1-2 sentences on progress since last round]
```

If a Review Log section already exists, append a new Round entry — do not overwrite previous rounds.
