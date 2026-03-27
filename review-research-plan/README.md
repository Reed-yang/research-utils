# review-research-plan

A Claude Code skill for critically reviewing research plans and roadmaps. It catches fatal flaws, challenges assumptions, and iteratively polishes rough ideas into executable plans.

## Design Philosophy

**Falsify, don't confirm.** The skill is deliberately adversarial — it tries to break your plan before reality does. Every critique comes with at least one path forward.

**Lazy-load architecture.** The skill is split into routing (SKILL.md), commands, and references. Only the files needed for the current phase are loaded, keeping the agent's context window focused on deep reasoning rather than bloated with instructions.

**Evidence over reasoning.** When a codebase or paper library is available, technical claims are verified against actual code — not just debated in the abstract. The most valuable findings come from grounding.

## How It Works

```
User prompt + plan file
        │
        ▼
┌─────────────────────┐
│  SKILL.md (router)  │  Detects maturity level + external context
└────────┬────────────┘
         │
    ┌────┴─────┬──────────────┐
    ▼          ▼              ▼
 /quick    /full           /re-review
 review    review          (iterative)
    │          │              │
    │    ┌─────┴─────────┐    │
    │    │ Lazy-load:     │    │
    │    │ novelty.md     │    │
    │    │ adversarial.md │    │
    │    │ feasibility.md │    │
    │    │ report-format  │    │
    │    └─────┬─────────┘    │
    │          │              │
    ▼          ▼              ▼
 Verdict   Diagnosis      Convergence
 + next    Report +       check +
 action    Revised Draft  re-diagnosis
```

## Three Commands

| Command | When to use | What you get | Time |
|---------|-------------|-------------|------|
| `/quick-review` | Early ideas, rough sketches | Core bet + top-3 risks + verdict + next action | ~5 min |
| `/full-review` | Mid/late-stage plans | Novelty scan + adversarial review + feasibility audit + diagnosis report + revised draft | ~30 min |
| `/re-review` | After revising a reviewed plan | Fix verification + new issue scan + convergence tracking | ~15 min |

You don't need to pick manually — the skill auto-routes based on plan maturity.

## Full Review Pipeline

```
Step 1   Parse plan elements (flag what's missing)
         │
Step 2   Novelty scan (web search against latest literature)
         │
Step 3   Adversarial review (falsify claims, steelman competitors)
         │
Step 4   Feasibility audit (compute, data, engineering, timeline)
         │
Step 5   Diagnosis report + prescription (with severity grading)
         │
Step 6   ⏸️  Pause — you choose which suggestions to adopt
         │
Step 7   Revised draft (changes marked inline)
         │
Step 8   Review log appended (enables /re-review tracking)
```

## Maturity Detection

The skill reads your plan and classifies it before choosing a command:

| Level | Signals | Auto-routes to |
|-------|---------|----------------|
| **Early** | Vague claims, no evaluation plan, no risks | quick-review (constructive) |
| **Mid** | Some specificity, limited evidence | full-review (balanced) |
| **Late** | Specific claims, preliminary results, risk-aware | full-review (aggressive) |

You can always override: *"switch to aggressive"* or *"use quick-review instead"*.

## Context Grounding

When you provide a codebase path or paper directory alongside your plan, the skill shifts from pure text analysis to evidence-based review:

```
Without context:  "This compression ratio change seems feasible"
With context:     "The architecture hardcodes factor=2 downsampling —
                   12× compression is architecturally impossible"   ← verified in code
```

This is the difference between a surface-level review and one that catches real blockers.

## Severity Levels

| Level | Meaning | Prescription |
|-------|---------|-------------|
| 🔴 **FATAL** | Direction needs reconsideration | 2+ options including possible pivot |
| 🟠 **SERIOUS** | Reviewer would likely reject | 1-2 options within existing timeline |
| 🟡 **CONCERN** | Worth noting, won't determine acceptance alone | Brief note |

## Iterative Convergence

The review loop tracks progress across rounds:

```
Round 1:  3 🔴  4 🟠  2 🟡  →  revise
Round 2:  0 🔴  2 🟠  3 🟡  →  improving ✓
Round 3:  0 🔴  1 🟠  1 🟡  →  ready to execute
```

Exit criteria: zero fatal issues, at most 1-2 serious issues with clear mitigation plans.

## File Structure

```
review-research-plan/
├── SKILL.md              # Router: maturity detection, context detection, command routing
├── commands/
│   ├── quick-review.md   # Fast single-pass procedure
│   ├── full-review.md    # Multi-phase procedure (8 steps)
│   └── re-review.md      # Iterative follow-up procedure
└── references/           # Lazy-loaded by commands as needed
    ├── adversarial.md    # Structured falsification + steelman protocol
    ├── novelty.md        # Web search novelty verification
    ├── feasibility.md    # Resource & architectural feasibility checklist
    └── report-format.md  # Output templates + severity definitions
```

## Usage

Invoke directly in Claude Code:

```
# Let the skill auto-detect the right command:
"Review this research plan: /path/to/roadmap.md"

# Or specify a command:
"/full-review /path/to/plan.md"

# With codebase grounding:
"Full review of /path/to/plan.md — the codebase is at /path/to/project/ and papers in /path/to/papers/"

# Iterative polishing:
"/re-review /path/to/revised-plan.md"
```
