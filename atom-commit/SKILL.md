---
name: atom-commit
description: Use when committing code changes to git, especially when working tree has multiple unrelated modifications. Analyzes diffs, writes Conventional Commits messages, and splits large changesets into decoupled atomic commits.
---

# Atomic Commit

Analyze uncommitted changes, write professional commit messages, and split large changesets into atomic commits. **Default: execute directly without asking for confirmation.**

## When to Use

- User says "commit", "提交", "atom-commit"
- After completing implementation work

## Workflow

1. **Gather**: `git status`, `git diff`, `git diff --cached`, `git log --oneline -10`
2. **Classify**: group changes by logical concern (feat/fix/refactor/docs/style/test/chore/perf)
3. **Split or not**: same concern across files → one commit; unrelated changes → split; formatting mixed with logic → separate (formatting first); feature + its tests → together
4. **Resolve attribution**: run the bundled resolver and use its exact trailer
5. **Execute**: stage specific files → commit with HEREDOC message → repeat → verify with `git log` + `git status`

## When to Pause

Default is **execute without preview**. Pause only in these three cases; ask the user only in the first two:

1. **Ambiguous atomicity** — the changeset is complex and there are multiple reasonable ways to split; present the proposed plan and ask for confirmation
2. **Uncertain files** — a file looks like it might not belong (generated files, `.env`, credentials, unrelated temp files); ask whether to include it
3. **Unresolved attribution** — the resolver cannot verify the current model and harness; stop before committing and report its diagnostics without asking the user to identify the model

Everything else: just commit.

## Commit Message Format

Conventional Commits, English, imperative mood:

```
<type>(<scope>): <subject>          ← max 72 chars, no period
                                     ← blank line
<body>                               ← what and why, wrap at 72
                                     ← blank line
Assisted-By: <model> via <harness>
```

Example:
```
feat(ingestion): add GLM-OCR cloud engine support

Integrate Zhipu GLM-OCR as an alternative OCR backend for documents
where local MinerU produces poor results.

Assisted-By: gpt-5.6-sol via Claude Code
```

## Commit Attribution

- Always append exactly one `Assisted-By: <model> via <harness>` trailer
- Identify the harness from the active client, such as `Claude Code`, `Codex CLI`, or `Codex App`
- Record the complete canonical model slug, such as `gpt-5.6-sol`; never shorten it to a family or display label such as `GPT-5` or `GPT-5.6`
- Resolve attribution with the bundled script before staging; do not rely on model self-identification
- Preserve the canonical slug's lowercase spelling, punctuation, and variant suffix
- Map a harness routing alias only when its underlying slug is explicit; for example, render local route `claude-gpt-5-6-sol` as `gpt-5.6-sol via Claude Code`
- Stop before committing when the complete slug cannot be verified; never guess, ask the user, or use `Unknown model` or a broader model family as fallback
- Do not add a second native or provider attribution trailer
- Replace the trailer with `Co-Authored-By: <model> via <harness> <email>` only when the user or repository explicitly requires GitHub co-author semantics and supplies the exact trusted email

## Resolve Attribution Deterministically

Run the bundled resolver by its path in the installed `atom-commit` skill immediately before staging:

```bash
python3 <atom-commit-skill-directory>/scripts/resolve_attribution.py --json
```

- Use the returned `trailer` verbatim
- In Codex, require `CODEX_THREAD_ID`; the resolver locates only that thread's rollout and reads the latest `turn_context.payload.model`
- Derive the Codex harness from `session_meta.payload.originator`; map `Codex Desktop` to `Codex App` and `codex-tui` or `codex_exec` to `Codex CLI`
- In Claude Code, require `CLAUDE_CODE_SESSION_ID`; the resolver locates only that session's transcript and reads the latest assistant message's `message.model`
- Convert only explicit routing aliases with a deterministic mapping, such as `claude-gpt-5-6-sol` to `gpt-5.6-sol via Claude Code`
- Use the latest turn rather than the persistent config because `/model` can change the model during a session
- Never scan another thread for a plausible model, read a supported-model catalog, or infer the slug from behavior, pricing, endpoint names, or model-family instructions
- For an unsupported harness, pass model and harness values obtained from its current-turn metadata using `--model <slug> --harness <name>`; do not pass remembered, inferred, or user-supplied values
- If the resolver exits nonzero, stop before committing and report the missing metadata and checked source. Do not ask the user which model is running

## Constraints

- Always stage specific files — never `git add .` or `git add -A`
- Use `git add -p` when splitting changes within a single file
- Never amend, never `--no-verify`
- Never commit `.env` or secrets — warn if detected
- Use HEREDOC for multi-line commit messages
- Check `git log` to match existing project commit style
