# /re-review — Iterative Follow-Up Review

For plans that have been reviewed before. Focuses on verifying fixes, checking for new issues
introduced by revisions, and tracking convergence toward submission-readiness.

## Prerequisites

The plan should have a `## Review Log` section at the bottom from a previous review round.
If it doesn't, treat this as a /full-review instead.

## Procedure

### Step 1: Read Previous Review Log

Parse the most recent review log entry. Note:
- Which findings are marked `open` (still need attention)
- Which are marked `addressed` (claim to be fixed)
- Which are marked `wont-fix` (user decided not to fix)

### Step 2: Verify Fixes

For each `addressed` finding, check whether the fix is actually adequate:
- Does the fix resolve the root cause or just paper over the symptom?
- Did the fix introduce new problems?

If a fix is inadequate, **reopen** the finding with a note explaining why.

### Step 3: Skip Resolved Items

- `wont-fix` items: Skip unless the user explicitly asks to revisit.
- Adequately `addressed` items: Mark as verified and move on.

### Step 4: Scan for New Issues

The revision may have introduced new problems. Do a focused scan:
- New claims that weren't in the original (do they need novelty verification?)
- Changed methodology (does it introduce new assumptions?)
- Shifted scope (does the evaluation plan still match?)

Load reference files only if new issues warrant deep analysis (e.g., load
`references/novelty.md` if a new claim needs novelty checking).

### Step 5: Convergence Check

Count open Fatal + Serious issues across rounds:

- **Decreasing** → On track. Report progress.
- **Stable** → Stalling. Suggest which specific issues to prioritize.
- **Increasing** → Red flag. The plan may need a more fundamental rethink rather than
  incremental patches. Say so directly.

### Step 6: Output

Use the same report format as /full-review (see `references/report-format.md`) but add a
convergence summary at the top:

```markdown
**Re-review Round [N]** | Previous: [N-1] open issues → Current: [M] open issues
**Convergence**: [Improving / Stalling / Regressing]
```

Then follow the same Prescription → Pause → Revised Draft → Log Update flow.

## Target End State

The plan is ready to exit the review loop when:
- Zero 🔴 Fatal issues remain
- At most 1-2 🟠 Serious issues remain and have clear mitigation plans
- The user feels confident presenting to their advisor or starting execution
