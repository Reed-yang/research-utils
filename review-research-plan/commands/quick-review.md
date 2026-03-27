# /quick-review — Fast Single-Pass Review

A lightweight review for early-stage ideas or when the user wants rapid feedback.
Takes ~5 minutes, produces a concise diagnosis without a revised draft.

## Procedure

### Step 1: Identify the Core Bet

In 1-2 sentences, state what must be true for this plan to work. This is the central
hypothesis — everything else depends on it.

### Step 2: Top-3 Risks

Without loading any reference files, identify the three most critical risks by asking:

1. **"What kills this?"** — Is there a single fact that, if true, makes this direction unviable?
   (e.g., someone already published it, the data doesn't exist, the compute is 100x too much)
2. **"What's the simplest alternative?"** — Would a senior researcher say "why not just do X
   instead?" If so, what's X, and why is this plan better?
3. **"What's the longest pole?"** — What's the single step that takes the most time or has the
   most uncertainty? Can it be de-risked early?

### Step 3: Verdict

Deliver one of:

- **🟢 Direction looks promising** — proceed to detailed planning. [Note 1-2 things to watch]
- **🟡 Promising but needs work** — [specific gaps to fill before committing]. Consider /full-review next.
- **🔴 Significant concerns** — [what needs to change]. May need to pivot.

### Step 4: Suggested Next Action

One concrete thing the user should do next. Examples:
- "Run a minimum viable experiment: [specific setup] to test [core assumption] in ~1 week"
- "Search arxiv for [specific query] to verify novelty of [specific claim]"
- "Flesh out the evaluation section, then run /full-review"

## Output Format

Keep it concise — aim for roughly one page. No section headers heavier than `###`.

```markdown
**Maturity**: [Early/Mid/Late] | **Mode**: Quick Review

**Core bet**: [1-2 sentences]

**Risk 1 — [title]** (🔴/🟠/🟡): [2-3 sentences]
**Risk 2 — [title]** (🔴/🟠/🟡): [2-3 sentences]
**Risk 3 — [title]** (🔴/🟠/🟡): [2-3 sentences]

**Verdict**: [🟢/🟡/🔴 + one sentence]

**Next action**: [one concrete step]
```
