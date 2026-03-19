# Feasibility Audit Checklist

Evaluate the plan against real-world constraints. Go through each item below and flag
issues at appropriate severity.

## Compute Budget

- Estimate GPU-hours for the plan's key experiments (training runs, ablations, baselines).
- Compare against the user's available resources. If mentioned in the plan or known from
  context, use those numbers. Otherwise ask.
- Flag if: total estimated compute > available budget, or a single experiment requires
  uninterrupted multi-day exclusive access to the full cluster.
- Consider: Is the cluster shared with production or other users?

## Data Requirements

- Are required datasets publicly available and accessible?
- If custom data collection or annotation is needed, estimate effort (days/weeks).
- Flag any data dependencies that could block progress for >1 week.
- Check: Are there licensing or ethical concerns with the data?

## Engineering Complexity

- How many new components need to be built from scratch vs. adapted from existing code?
- Is this a minor extension of existing infrastructure or a ground-up build?
- Estimate engineering time separately from experiment time — they're often conflated.

### Architectural Feasibility

If the plan proposes modifications to an existing system:
- Does the architecture actually support the proposed change? (e.g., a "12× compression"
  may be impossible if the system only supports power-of-2 ratios)
- Are there hard constraints (data formats, API contracts, compiled components) that block
  the proposal?
- What would need to be rebuilt vs. configured vs. fine-tuned?

This is distinct from complexity — a change can be simple but architecturally impossible.

## Time-to-First-Signal

This is one of the most important checks.

- How long until the first meaningful experimental result that tests the core hypothesis?
- If >2-3 weeks to get any signal → 🟠 Red flag.
- Look for a **minimum viable experiment**: a quick test of the core idea that can run in
  ~1 week with reduced scale (smaller model, subset of data, simplified pipeline).
- If no such experiment exists, suggest one.

## Dependency Chain

- Map which steps depend on which.
- Identify the **critical path** — the longest chain of sequential dependencies.
- Flag **single points of failure** — steps where one blockage delays everything downstream.
- Are there steps that can run in parallel to shorten the critical path?

## Human Bandwidth

- Is this a solo effort or a team? Who does what?
- Realistic weekly time allocation considering: coursework, advisor meetings, other projects,
  teaching/TA duties, personal commitments.
- Flag if the timeline assumes >60 hours/week sustained effort — this is unsustainable.

## Output

Summarize as a feasibility snapshot:

```markdown
### Feasibility Snapshot
- **Estimated total GPU-hours**: [range]
- **Time-to-first-signal**: [estimate]
- **Critical path bottleneck**: [what and why]
- **Minimum viable experiment**: [exists / suggested / not feasible]
- **Overall feasibility**: [Feasible / Tight but doable / At risk / Infeasible]
```
