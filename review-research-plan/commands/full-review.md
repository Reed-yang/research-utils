# /full-review — Comprehensive Multi-Phase Review

The thorough review for mid-to-late stage plans. Runs novelty scan, adversarial review, and
feasibility audit. Outputs a diagnosis report with prescriptions and optionally a revised draft.

## Execution Flow

```
Parse plan elements → Novelty Scan → Adversarial Review → Feasibility Audit
    → Diagnosis Report + Prescription → ⏸️ Pause → Revised Draft → Review Log
```

## Step-by-Step

### Step 1: Parse Plan Elements

Extract these from the plan (note which are missing):
- Problem statement & motivation
- Core hypothesis (the central bet)
- Key claims / contributions (list each explicitly)
- Proposed approach & methodology
- Evaluation plan (baselines, metrics, datasets)
- Related work awareness
- Resource requirements
- Timeline / milestones
- Risk acknowledgment

Missing elements are findings — flag them at appropriate severity based on maturity level.

### Step 2: Novelty Scan (Default ON)

Load `references/novelty.md` and follow its procedure.

Skip if the user says "skip novelty scan" or "I've already checked the literature."

### Step 3: Adversarial Review

Load `references/adversarial.md` and follow its procedure.
Intensity is determined by maturity level (see adversarial.md § Depth by Maturity).

### Step 4: Feasibility Audit

Load `references/feasibility.md` and follow its checklist.

### Step 5: Diagnosis Report + Prescription

Load `references/report-format.md` for report template and prescription format.

Compile all findings from Steps 1-4 into a structured report with severity grading,
then provide modification suggestions for each finding per the prescription template.

### Step 6: Pause (Default)

Pause and ask the user which suggestions to adopt. Match the plan's language.

**Skip pause** if the user said "don't pause" / "continuous mode" / "一口气跑完".

### Step 7: Revised Draft

Generate a revised plan incorporating adopted suggestions. Mark changes with:
- `[NEW]` — added content
- `[MODIFIED]` — changed content (note what changed)
- `[REMOVED: reason]` — deleted content

Preserve the original plan's voice and style. Only change what the suggestions call for.

### Step 8: Review Log

Append a review log entry to the plan. See `references/report-format.md` for the log format.
