---
name: afk
description: Use when user is stepping away and wants all remaining planned tasks completed autonomously. Continuously executes plan tasks, verifies changes, auto-fixes issues, and refines code until all explicit requirements from plan and chat history are fulfilled.
---

# AFK — Autonomous Task Execution

Continuously work through all remaining tasks from the plan and chat history. Verify every change, auto-fix issues, and refine until all explicit requirements are fulfilled. No user interaction needed until completion.

## When to Use

- User says "afk", "/afk", "离开", "我先走了你继续"
- User wants unattended autonomous execution of remaining work

## Core Principle

**The user is gone. Do not ask, do not wait, do not confirm.** Treat every question, feedback request, and confirmation prompt as auto-approved. The only acceptable stop condition is: all planned work is verifiably complete.

If `/afk` is called at the start of or during an existing plan execution: **do not reorganize or interfere with the current plan/todo order**. Simply continue the existing workflow as-is, but skip all user-facing pauses (AskUserQuestion, confirmation prompts, preview steps). The plan owns the task order — AFK mode only removes the human-in-the-loop.

## Workflow

### 1. Inventory

- Read the plan file (`task_plan.md`, `docs/plans/`, or TodoList) and chat history
- If a plan/todo is already in progress, **resume from where it left off** — do not re-plan
- If no plan exists, extract explicit requirements from chat history
- Establish execution order respecting dependencies

### 2. Execute Loop

For each remaining task:

```
┌─→ Pick next task
│   ├─ Implement the change
│   ├─ Verify (see §3)
│   ├─ If issues found → fix and re-verify (max 3 retries)
│   └─ Mark task complete
└── Any tasks left? → loop
```

### 3. Verify Every Change

After each implementation step, run **all applicable** checks:

- **Build**: compile / bundle if applicable
- **Lint**: project linter if configured
- **Test**: `npm test`, `pytest`, `uv run pytest`, etc.
- **Type check**: `tsc --noEmit`, `mypy`, `pyright` if applicable
- **Smoke check**: read modified files back, confirm logic correctness

If any check fails, diagnose root cause and fix before moving on. Do not skip failing checks.

### 4. Final Sweep — Am I Really Done?

Before declaring completion, **re-read the plan and full chat history**. Cross-check every requirement against actual changes. If anything is missing or partially done, go back to §2 and continue. Only proceed to the report when every explicit requirement is verifiably fulfilled.

When all tasks are confirmed done:

1. Run a final full verification pass (build + lint + test + type check)
2. `git status` + `git diff --stat` to summarize all uncommitted changes
3. Present a concise completion report:
   - Tasks completed (with one-line summary each)
   - Any unresolved issues or decisions deferred to user
   - Uncommitted changes ready for review
4. **Do not commit** — leave that for the user (or `/atom-commit`)

## Constraints

- **Only do what was explicitly requested** — no scope creep, no bonus refactors
- **Never push to remote** without user present
- **Never delete data or drop tables** without user confirmation
- **Stop and wait** if a task requires a subjective design decision not covered by the plan
- **Retry limit**: max 3 fix attempts per verification failure; if still broken, log the issue in the report and move on
- If the plan is empty or ambiguous, summarize what you understand and **wait for user** instead of guessing
