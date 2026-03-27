# Novelty Scan Procedure

Uses web search to check the plan's key claims against the latest literature.
Default ON. Skip if user says "skip novelty scan" or confirms they've checked.

## Steps

### 1. Extract Key Claims

Identify 3-5 specific technical contributions from the plan. Focus on what the plan claims
is *new*, not on generic problem statements.

Good: "identity-consistent video editing via disentangled cross-attention"
Bad: "we work on video generation" (too generic to search meaningfully)

### 2. Construct Search Queries

For each claim, create 2-3 short, targeted queries (3-6 words each).
Combine the technique with the application domain.

Example:
- Claim: "camera reframing with motion-aware inpainting"
- Queries: `camera reframing video generation`, `motion aware inpainting video`,
  `video outpainting camera control`

### 3. Search and Analyze

For each claim, search and assess three dimensions:

**Direct overlap** — Does an existing paper do essentially the same thing?
- If yes → 🔴 Novelty at risk. This is a potential Fatal finding.
- Check both the method AND the problem formulation. Same method on a different problem
  (or vice versa) is partial overlap, not full.

**Recent baselines** — Any new SOTA methods in the last 3-6 months?
- If yes → The plan's experimental comparison may need updating.
- Not a novelty issue per se, but affects positioning.

**Trend direction** — Is the field moving toward or away from this approach?
- Toward: The idea has momentum but scoop risk is elevated.
- Away: May indicate known problems — search for negative results or critiques.

### 4. Output Per-Claim Assessment

For each claim, output one line:

- 🟢 **Clear**: No direct overlap found, meaningful delta from nearest work.
- 🟡 **Partial**: Related work exists but plan has differentiating aspects. Needs careful positioning.
- 🔴 **At risk**: Very similar work exists or was recently published. Needs significant repositioning.

### Caveats (Always State These)

- Web search is not exhaustive — papers in review or unpublished preprints won't appear.
- No results ≠ guaranteed novelty. It could mean the queries missed the right keywords.
- For fast-moving fields, even a 🟢 today may become 🔴 in a month. Flag "obvious next step" risk.
- Always recommend the user verify independently against key venues and author pages.
