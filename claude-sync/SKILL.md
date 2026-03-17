---
name: claude-sync
description: Sync ~/.claude settings across devices using claude-sync CLI. Use when user wants to pull/push Claude Code configuration, resolve cross-device conflicts, merge plugin lists between machines, selectively sync specific paths, clean up remote storage, or manage .claudesyncignore rules. Handles platform-specific conflicts (Linux↔Windows), interactive diff review, selective field merging, parallel I/O, and glob-based remote deletion.
---

# Claude Sync Manager

Synchronize `~/.claude` configuration across devices with interactive conflict resolution, platform-aware merging, and user confirmation at every destructive step.

## When to Use This Skill

- User wants to sync Claude Code settings between machines
- User says "sync", "pull", "push" related to claude-sync
- User wants to check what changed on a remote device
- User needs to resolve conflicts between local and remote `~/.claude`
- User wants to merge specific fields (e.g., plugins) without full overwrite
- User wants to delete specific files from remote R2/S3/GCS storage
- User says "delete", "clean up", "remove" related to remote sync storage
- User wants to selectively push/pull only certain paths (e.g., "just sync plugins")
- User wants to set up or edit .claudesyncignore rules
- User wants to back up or restore specific remote files to a local directory
- User wants to remap session paths after moving project directories

## Prerequisites

- `claude-sync` CLI must be installed and initialized (`claude-sync init`)
- If `claude-sync` is not found, inform the user and offer to install from source:
  ```
  git clone https://github.com/tawanorg/claude-sync.git /tmp/claude-sync
  cd /tmp/claude-sync && go build -ldflags "-s -w" -o bin/claude-sync.exe ./cmd/claude-sync
  cp bin/claude-sync.exe "$(npm prefix -g)/claude-sync.exe"
  ```
- On Windows, `claude-sync status` may fail with "failed to hash skills" if `~/.claude/skills` is a symlink/junction — this is a known tool bug and does not affect pull/push operations

## New CLI Commands

### Selective Push/Pull

Push or pull specific paths instead of everything:

```bash
claude-sync push skills/ settings.json           # Push only these paths
claude-sync pull projects/ CLAUDE.md              # Pull only these paths
claude-sync pull --target /tmp/backup plugins/    # Download to custom dir (read-only)
claude-sync pull --dry-run                        # Preview what would change
```

Paths are relative to `~/.claude/`. All commands also respect `.claudesyncignore`.

### Delete Command

Remove remote files matching glob patterns:

```bash
claude-sync delete "plugins/cache/**" --dry-run           # Preview
claude-sync delete "plugins/marketplaces/*/.git/**"       # Delete .git dirs
claude-sync delete "projects/-home-*/**"                  # Delete old-path sessions
```

**Always run with `--dry-run` first.** Deletion requires typing 'yes' to confirm. Warns if deleting >50% of all remote files.

### .claudesyncignore

Place `~/.claude/.claudesyncignore` to permanently exclude paths from sync:

```gitignore
plugins/marketplaces/*/.git/
plugins/cache/
**/node_modules/
skills/paper-ingestion/mineru-fork/
```

Uses `.gitignore` syntax. Applied to both push and pull.

### Status and Diff with Path Filtering

```bash
claude-sync status plugins/                # Show changes in specific directory
claude-sync diff settings.json             # Compare specific file
```

## Workflow

### Phase 1: Pre-flight Check

1. Verify `claude-sync` is installed: run `claude-sync --version`
2. Run `claude-sync pull --dry-run` to preview all incoming changes
   - If user wants selective sync, add path arguments: `claude-sync pull --dry-run skills/ CLAUDE.md`
3. Parse the output and categorize files into:
   - **OVERWRITE**: files that exist locally and will be replaced by remote versions
   - **NEW**: files only on remote, will be added locally
   - **KEEP**: files only local, will not be touched
4. Present a summary table to the user:
   - Count of OVERWRITE / NEW / KEEP files
   - List all OVERWRITE files with their local vs remote dates

### Phase 2: Conflict Analysis

For each OVERWRITE file, determine its type and risk level:

#### Critical Config Files (always require user review)
- `CLAUDE.md` — global Claude behavior rules
- `settings.json` — model, plugins, permissions, statusLine
- `settings.local.json` — machine-specific settings

#### Low-risk Files (inform but don't block)
- `history.jsonl` — command history
- `plugins/` directory files — plugin caches and marketplace data
- `projects/` directory files — session recordings

For critical config files:
1. Back up each file: `cp <file> <file>.bak`
2. Read the local version content
3. Execute `claude-sync pull --force` to get the remote version
4. Read the updated (remote) version content
5. Show a **side-by-side diff** to the user
6. Ask the user how to handle each conflict (see Phase 3)

### Phase 3: Interactive Conflict Resolution

For each conflicting critical file, present the diff and ask the user to choose a resolution strategy:

#### Strategy A: Keep Local
Restore from backup: `cp <file>.bak <file>`

#### Strategy B: Keep Remote
No action needed (pull already applied the remote version).

#### Strategy C: Auto-Merge (for structured files)
Apply intelligent merging rules based on file type:

**For `CLAUDE.md`**:
- Compare rules/sections between local and remote
- For similar rules: keep the more detailed/specific version
- For rules only in remote: **always preserve them** (append to local)
- For rules only in local: keep them
- Reconstruct the file with merged content

**For `settings.json`**:
- Detect platform mismatch (Linux paths vs Windows paths)
- `statusLine`: if it contains hardcoded paths from another OS (e.g., `/home/...` on Windows or `C:\...` on Linux), **discard it** and warn the user
- `enabledPlugins`: **union merge** — combine all plugins from both local and remote
- `autoUpdatesChannel`: keep local value if remote doesn't have it
- `permissions`: keep local value (machine-specific)
- Other scalar fields: keep remote if newer, keep local otherwise

**For `settings.local.json`**:
- This file is inherently machine-specific
- If it arrives from a different platform, warn the user that it likely contains incompatible paths/commands
- Default recommendation: **do not merge**, keep local or discard remote

#### Strategy D: Manual Edit
Open the file content for the user to review and edit manually.

### Phase 4: Symlink & Platform Safety Checks

After pull and merge:
1. Verify `~/.claude/skills` symlink is intact (if it existed before): `ls -la ~/.claude/skills`
2. Check that the symlink target is still accessible
3. If the symlink was broken or replaced by a regular file/directory, alert the user immediately
4. Scan `settings.json` for platform-incompatible values:
   - Linux paths (`/home/...`) on Windows
   - Windows paths (`C:\...`) on Linux
   - Hardcoded node/python binary paths from another machine

### Phase 4b: Remote Cleanup (when requested)

If the user wants to clean up remote storage:
1. Identify what to clean: `.git/` dirs, cache, old-path sessions, etc.
2. Run `claude-sync delete [patterns...] --dry-run` to preview
3. Show the matched file count and total size
4. Ask user to confirm before executing actual deletion
5. After deletion, run `claude-sync pull --dry-run` to verify remaining files are correct

### Phase 5: Verification & Cleanup

1. Run `claude-sync diff` to confirm local and remote are now aligned on critical files
2. Show the user what was changed, merged, or skipped
3. Ask if they want to `claude-sync push` local changes back (to propagate merges to remote)
4. Clean up backup files: `rm ~/.claude/*.bak` (only after user confirms)

## Output Format

### Summary Table (Phase 1)
```
| Category  | Count | Details                          |
|-----------|-------|----------------------------------|
| OVERWRITE |     2 | CLAUDE.md, settings.json         |
| NEW       |   993 | plugins (865), projects (127)... |
| KEEP      |   590 | local-only files, not affected   |
```

### Conflict Resolution Report (Phase 3)
For each conflicting file, report:
```
## CLAUDE.md — Auto-Merged
- Kept local (more detailed): Language rule, Code Formatting rule
- Added from remote: "Comments in English" rule
- Discarded: none

## settings.json — Selective Merge
- Kept local: autoUpdatesChannel
- Merged (union): enabledPlugins (+7 new plugins)
- Discarded: statusLine (Linux path, incompatible with Windows)
```

### Final Status (Phase 5)
```
Sync complete.
- 2 files merged (CLAUDE.md, settings.json)
- 993 new files pulled
- 590 local files untouched
- Symlink ~/.claude/skills → intact
```

## Important Constraints

- **Never overwrite without user confirmation** for critical config files
- **Never auto-push** — always ask before running `claude-sync push`
- **Always back up** critical files before pulling
- **Always verify symlinks** after pull operations
- **Never silently discard** local-only rules or settings — present diffs first
- **Platform awareness is mandatory** — always check for cross-platform path incompatibilities
- If `claude-sync status` fails due to symlink issues, inform the user it's a known bug and proceed with pull/push
- If the user only wants to sync a specific field (e.g., "just sync plugins"), skip the full merge workflow and only merge that field
