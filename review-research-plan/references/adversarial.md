# Adversarial Review Protocol

Two sub-modules: Structured Falsification and Steelman the Competition.
Run both in sequence. Adjust depth based on maturity level.

## Structured Falsification

For each core hypothesis and key claim in the plan:

**1. State the assumption** being tested. Quote or paraphrase the plan's claim directly.

**2. Invert it**: "Assume this is wrong. Under what conditions would it fail?"

**3. Identify concrete failure scenarios** — not abstract worries, but specific technical
situations with enough detail to be actionable.

Example:
> Assumption: "Disentangling identity and motion enables identity-consistent editing."
>
> Failure scenarios:
> (a) Under extreme pose changes, identity features leak into motion subspace
> (b) Disentanglement works for near-frontal faces but fails on profile views
> (c) Contrastive loss achieves low training loss but separation doesn't transfer OOD

**4. Assess survivability**: If this assumption fails:
- Does the entire plan collapse? → Fatal dependency, 🔴 severity
- Can it degrade to a weaker-but-publishable claim? → Serious, 🟠
- Is it a nice-to-have that doesn't affect the core? → Concern, 🟡

**5. Ground against evidence** (when codebase or papers are available):
Don't just reason about whether an assumption could fail — check. If the plan claims
"technique X can be directly applied to system Y", examine Y's code to verify compatibility.
Dispatch subagents for targeted verification of specific claims rather than broad exploration.
A finding backed by code evidence is far stronger than one based on reasoning alone.

### Depth by Maturity

- **Early-stage**: Falsify only the 1-2 most central assumptions. Skip secondary claims.
- **Mid-stage**: Falsify all core claims (typically 3-5).
- **Late-stage**: Falsify core + secondary claims, and also challenge evaluation assumptions
  (e.g., "Is this metric actually measuring what you claim?").

## Steelman the Competition

For each key claim, ask:

> "What is the simplest alternative approach a senior researcher would suggest instead?"

This is the "Why not just do X?" test. Construct the strongest possible version of the
competing approach — don't strawman it.

Evaluate:
1. **Is the plan's added complexity justified** over this simpler alternative?
2. **Does the plan explicitly compare** against this alternative (or plan to)?
3. **Could a reviewer dismiss the plan** by pointing to this alternative?

If the plan lacks a convincing answer to "why not just X?", this is a 🟠 Serious finding.

Also check: Is there a concurrent/recent work doing something very close? If the plan's
approach is the "obvious next step" from a recent paper, scoop risk is high — flag it.
