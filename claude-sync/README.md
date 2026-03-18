# claude-sync Skill

Cross-device `~/.claude` configuration sync with interactive conflict resolution, powered by [claude-sync](https://github.com/tawanorg/claude-sync) CLI and Claude Code.

## Why This Skill

`claude-sync` CLI handles file transfer and encryption, but it treats sync as a bulk operation — pull overwrites, push uploads. When syncing across different platforms (e.g., Linux server ↔ Windows laptop), blind overwrites cause real problems:

- `settings.json` may contain hardcoded Linux paths (`/home/...`) that break on Windows
- `CLAUDE.md` global rules may differ per machine, and you don't want to lose either version
- Plugin lists diverge as you install different plugins on each device
- Symlinks (`~/.claude/skills`) can be silently broken by a careless pull

This skill wraps `claude-sync` with an intelligent, interactive layer that understands these conflicts and asks before acting.

With the new selective sync and delete features, the skill also helps with:
- Targeted sync of specific paths (e.g., "just pull my skills and settings")
- Remote storage cleanup with glob pattern matching and dry-run preview
- Setting up `.claudesyncignore` to prevent syncing unwanted files (cache, .git, node_modules)
- Session path remapping when moving projects to a new filesystem location

## Installing claude-sync

### Build from Source (recommended for this fork)

```bash
git clone https://github.com/tawanorg/claude-sync.git /tmp/claude-sync
cd /tmp/claude-sync
go build -ldflags "-s -w" -o bin/claude-sync ./cmd/claude-sync
sudo cp bin/claude-sync /usr/local/bin/   # or cp to ~/.local/bin/
```

### npm (upstream releases only)

```bash
npm install -g @tawandotorg/claude-sync
```

> Note: npm releases may not include the latest fork features (selective sync, delete, parallel I/O).

### Building from Source (Windows / unsupported platforms / glibc issues)

If `npm install` fails with a 404 or if you hit glibc version errors on older Linux, build from source.

#### Prerequisites

- **Go 1.21+** — the Go toolchain will automatically download the required Go 1.24 if needed
  - Windows: download from https://go.dev/dl/ or `winget install GoLang.Go`
  - Linux: `sudo apt install golang-go` or download from https://go.dev/dl/
  - macOS: `brew install go`
- **Git** — to clone the repository

#### Build Steps

```bash
# Clone
git clone https://github.com/tawanorg/claude-sync.git /tmp/claude-sync
cd /tmp/claude-sync

# Build for current platform
# Windows:
GOOS=windows GOARCH=amd64 go build -ldflags "-s -w" -o bin/claude-sync.exe ./cmd/claude-sync

# Linux:
GOOS=linux GOARCH=amd64 go build -ldflags "-s -w" -o bin/claude-sync ./cmd/claude-sync

# macOS (Apple Silicon):
GOOS=darwin GOARCH=arm64 go build -ldflags "-s -w" -o bin/claude-sync ./cmd/claude-sync
```

#### Install to PATH

```bash
# Windows (put it in npm global bin so it's on PATH):
cp bin/claude-sync.exe "$(npm prefix -g)/claude-sync.exe"

# Linux / macOS:
sudo cp bin/claude-sync /usr/local/bin/
```

#### Verify

```bash
claude-sync --version
# Expected: claude-sync version dev
```

> **Note**: Building from source shows `version dev` since there's no git tag context. This is cosmetic and does not affect functionality.

### First-Time Setup

After installing, run the interactive setup wizard:

```bash
claude-sync init
```

You'll configure:
1. **Storage provider** — Cloudflare R2 (recommended, 10GB free), AWS S3, or Google Cloud Storage
2. **Bucket credentials** — provider-specific API keys
3. **Encryption passphrase** — use the **same passphrase** on all devices for seamless sync

See the [upstream README](https://github.com/tawanorg/claude-sync#setup-guide) for detailed provider setup.

## Using the Skill with Claude Code

### Activating the Skill

This is a built-in skill — no installation beyond having it in your skills directory. Claude Code will automatically detect it when you:

```
/claude-sync
```

Or simply describe what you want in natural language:

```
"帮我同步一下远端的 claude 设置"
"sync my claude settings from the server"
"pull 远端的插件列表到本地"
```

### Recommended Interaction Patterns

#### Full Sync (first time or periodic)

```
"帮我执行一次完整的 claude-sync pull，检查冲突后再合并"
```

Claude will:
1. Run `pull --dry-run` and show you a summary
2. Identify conflicting files and show diffs
3. Ask how to handle each conflict
4. Execute the merge and verify integrity

#### Plugin-Only Sync

```
"只同步远端的插件列表到本地 settings.json"
```

Claude will only merge the `enabledPlugins` field, leaving everything else untouched.

#### Preview Without Changes

```
"看一下远端和本地有什么区别，不要做任何修改"
```

Claude will run `pull --dry-run` and `diff`, then present a detailed report.

#### Push Local Changes

```
"把我本地的配置推送到远端"
```

Claude will show what will be pushed and ask for confirmation before executing.

#### Selective Sync (specific paths only)

```
"只帮我同步 skills 和 settings.json"
"pull only my plugins directory"
```

Claude will use path arguments: `claude-sync pull skills/ settings.json`

#### Remote Cleanup

```
"帮我清理远端的 .git 目录和插件缓存"
"delete old session files from R2"
```

Claude will:
1. Run `claude-sync delete [patterns] --dry-run` to preview
2. Show matched files and total size
3. Ask for confirmation before executing

#### Set Up Ignore Rules

```
"帮我配置 .claudesyncignore 排除 node_modules 和 .git"
```

Claude will create/edit `~/.claude/.claudesyncignore` with appropriate patterns.

#### Backup to Local Directory

```
"把远端的 plugins 备份到 /tmp/backup"
```

Claude will run: `claude-sync pull --target /tmp/backup plugins/`

### Tips for Better Interactions

1. **Be specific about what you want to sync** — "sync plugins only" is better than "sync everything"
2. **Ask to preview first** — always safe to ask for `--dry-run` before committing
3. **Mention your platform** if Claude doesn't detect it — "I'm on Windows, the remote is Linux"
4. **Ask about specific files** — "show me the diff for settings.json" narrows the scope

## Known Issues

| Issue | Impact | Workaround |
|-------|--------|------------|
| `claude-sync status` fails with "failed to hash skills" | `status` command unusable | Caused by `~/.claude/skills` being a symlink/junction. Does not affect `pull`/`push`. Ignore safely. |
| `statusLine` contains hardcoded paths after pull | Claude Code HUD may break | The skill auto-detects cross-platform paths and discards incompatible `statusLine` entries |
| Windows binary not in npm releases | `npm install -g` fails with 404 | Build from source (see above) |
| `settings.local.json` pulled from remote | May contain machine-specific permissions | The skill warns about platform-specific content and recommends not merging |
| Delete patterns with leading dashes | `mv` in remap script fails | Use `mv --` prefix when handling paths starting with `-` |

## How It Works

```
claude-sync pull --dry-run
        │
        ▼
  Categorize files (OVERWRITE / NEW / KEEP)
        │
        ▼
  Critical files? ──no──→ Execute pull directly
        │ yes
        ▼
  Backup critical files (.bak)
        │
        ▼
  Pull remote versions
        │
        ▼
  Show diff for each conflict
        │
        ▼
  Ask user: Keep Local / Keep Remote / Auto-Merge / Manual Edit
        │
        ▼
  Apply resolution strategy
        │
        ▼
  Verify symlinks & platform safety
        │
        ▼
  Cleanup backups & optional push
```
