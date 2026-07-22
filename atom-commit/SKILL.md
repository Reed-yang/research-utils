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
4. **Execute**: stage specific files → commit with HEREDOC message → repeat → verify with `git log` + `git status`

## When to Pause and Ask

Default is **execute without preview**. Only ask the user in these two cases:

1. **Ambiguous atomicity** — the changeset is complex and there are multiple reasonable ways to split; present the proposed plan and ask for confirmation
2. **Uncertain files** — a file looks like it might not belong (generated files, `.env`, credentials, unrelated temp files); ask whether to include it

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

Assisted-By: GPT-5.6 SOL via Claude Code
```

## Commit Attribution

- Always append exactly one `Assisted-By: <model> via <harness>` trailer
- Identify the harness from the active client, such as `Claude Code`, `Codex CLI`, or `Codex App`
- Resolve the model from current session metadata first, then an explicit runtime environment or session setting, then persistent harness config when no session override is active
- Prefer the runtime display name; otherwise preserve the exact model ID and only normalize capitalization or separators when unambiguous
- When the harness is already named separately, remove an unambiguous harness routing prefix; for example, render `claude-gpt-5-6-sol` as `GPT-5.6 SOL via Claude Code`
- Use `Unknown model` rather than guessing when the exact model cannot be verified
- Do not add a second native or provider attribution trailer
- Replace the trailer with `Co-Authored-By: <model> via <harness> <email>` only when the user or repository explicitly requires GitHub co-author semantics and supplies the exact trusted email

## Constraints

- Always stage specific files — never `git add .` or `git add -A`
- Use `git add -p` when splitting changes within a single file
- Never amend, never `--no-verify`
- Never commit `.env` or secrets — warn if detected
- Use HEREDOC for multi-line commit messages
- Check `git log` to match existing project commit style
