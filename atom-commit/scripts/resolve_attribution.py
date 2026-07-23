#!/usr/bin/env python3
"""Resolve exact commit attribution from current agent session metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

THREAD_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
CLAUDE_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,128}$")
ROUTED_GPT_MODEL_PATTERN = re.compile(
    r"^claude-gpt-(?P<major>[0-9]+)-(?P<minor>[0-9]+)(?P<variant>-.+)?$"
)

CODEX_HARNESS_NAMES = {
    "Codex Desktop": "Codex App",
    "codex_app": "Codex App",
    "codex-tui": "Codex CLI",
    "codex_exec": "Codex CLI",
    "Codex CLI": "Codex CLI",
}


class ResolutionError(RuntimeError):
    """Raised when exact current-session attribution is unavailable."""


@dataclass(frozen=True)
class Attribution:
    model: str
    harness: str
    source: str

    @property
    def trailer(self) -> str:
        return f"Assisted-By: {self.model} via {self.harness}"

    def to_dict(self) -> dict[str, str]:
        result = asdict(self)
        result["trailer"] = self.trailer
        return result


def nonempty(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def validate_explicit_value(value: str, label: str) -> str:
    if not value:
        raise ResolutionError(f"{label} is empty")
    if "\n" in value or "\r" in value:
        raise ResolutionError(f"{label} contains a newline")
    return value


def canonicalize_model(model: str) -> str:
    validated = validate_explicit_value(model.strip(), "model")
    routed_gpt_match = ROUTED_GPT_MODEL_PATTERN.fullmatch(validated)
    if routed_gpt_match:
        variant = routed_gpt_match.group("variant") or ""
        return (
            f"gpt-{routed_gpt_match.group('major')}."
            f"{routed_gpt_match.group('minor')}{variant}"
        )
    return validated


def codex_rollouts(codex_home: Path, thread_id: str) -> list[Path]:
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ResolutionError("CODEX_THREAD_ID is not a valid Codex thread UUID")

    filename_pattern = f"rollout-*-{thread_id}.jsonl"
    matches: list[Path] = []
    for directory_name in ("sessions", "archived_sessions"):
        directory = codex_home / directory_name
        if directory.is_dir():
            matches.extend(directory.rglob(filename_pattern))
    return sorted(set(matches), key=lambda path: (path.stat().st_mtime_ns, str(path)))


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError as error:
        raise ResolutionError(f"cannot read session record {path}: {error}") from error


def resolve_codex(codex_home: Path, thread_id: str) -> Attribution:
    paths = codex_rollouts(codex_home, thread_id)
    if not paths:
        raise ResolutionError(
            f"no rollout matched CODEX_THREAD_ID={thread_id} under {codex_home}"
        )

    model: str | None = None
    originator: str | None = None
    verified_thread = False

    for path in paths:
        for entry in iter_jsonl(path):
            entry_type = entry.get("type")
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue

            if entry_type == "session_meta":
                session_id = nonempty(payload.get("id"))
                if session_id == thread_id:
                    verified_thread = True
                    candidate_originator = nonempty(payload.get("originator"))
                    if candidate_originator:
                        originator = candidate_originator
            elif entry_type == "turn_context":
                candidate_model = nonempty(payload.get("model"))
                if candidate_model:
                    model = candidate_model

    if not verified_thread:
        raise ResolutionError(
            "matching rollout did not contain the expected session id"
        )
    if not model:
        raise ResolutionError(
            "current Codex rollout did not contain turn_context.payload.model"
        )
    if not originator:
        raise ResolutionError(
            "current Codex rollout did not contain session_meta.payload.originator"
        )

    harness = CODEX_HARNESS_NAMES.get(originator)
    if not harness:
        raise ResolutionError(f"unsupported Codex originator: {originator}")

    source_paths = ",".join(str(path) for path in paths)
    return Attribution(
        model=canonicalize_model(model),
        harness=harness,
        source=f"codex-rollout:{source_paths}",
    )


def claude_transcripts(claude_home: Path, session_id: str) -> list[Path]:
    if not CLAUDE_SESSION_ID_PATTERN.fullmatch(session_id):
        raise ResolutionError("CLAUDE_CODE_SESSION_ID has an invalid format")

    projects_directory = claude_home / "projects"
    if not projects_directory.is_dir():
        return []
    matches = projects_directory.rglob(f"{session_id}.jsonl")
    return sorted(set(matches), key=lambda path: (path.stat().st_mtime_ns, str(path)))


def resolve_claude(claude_home: Path, session_id: str) -> Attribution:
    paths = claude_transcripts(claude_home, session_id)
    if not paths:
        raise ResolutionError(
            f"no transcript matched CLAUDE_CODE_SESSION_ID={session_id} under {claude_home}"
        )

    model: str | None = None
    verified_session = False
    for path in paths:
        for entry in iter_jsonl(path):
            if nonempty(entry.get("sessionId")) != session_id:
                continue
            verified_session = True
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            candidate_model = nonempty(message.get("model"))
            if candidate_model:
                model = candidate_model

    if not verified_session:
        raise ResolutionError(
            "matching transcript did not contain the expected session id"
        )
    if not model:
        raise ResolutionError("current Claude transcript did not contain message.model")

    source_paths = ",".join(str(path) for path in paths)
    return Attribution(
        model=canonicalize_model(model),
        harness="Claude Code",
        source=f"claude-transcript:{source_paths}",
    )


def resolve_explicit(model: str, harness: str) -> Attribution:
    return Attribution(
        model=canonicalize_model(model),
        harness=validate_explicit_value(harness.strip(), "harness"),
        source="explicit-current-turn-metadata",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve an exact Assisted-By trailer without guessing."
    )
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    parser.add_argument("--model", help="exact model from current-turn metadata")
    parser.add_argument("--harness", help="harness from current-turn metadata")
    parser.add_argument(
        "--thread-id",
        default=os.environ.get("CODEX_THREAD_ID"),
        help="Codex thread UUID (default: CODEX_THREAD_ID)",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex"),
        help="Codex state directory (default: CODEX_HOME or ~/.codex)",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("CLAUDE_CODE_SESSION_ID"),
        help="Claude Code session ID (default: CLAUDE_CODE_SESSION_ID)",
    )
    parser.add_argument(
        "--claude-home",
        type=Path,
        default=Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"),
        help="Claude Code state directory (default: CLAUDE_CONFIG_DIR or ~/.claude)",
    )
    return parser.parse_args()


def resolve(args: argparse.Namespace) -> Attribution:
    if bool(args.model) != bool(args.harness):
        raise ResolutionError("--model and --harness must be supplied together")
    if args.model and args.harness:
        return resolve_explicit(args.model, args.harness)
    if args.thread_id and args.session_id:
        raise ResolutionError(
            "both Codex and Claude session identifiers are set; current harness is ambiguous"
        )
    if args.thread_id:
        return resolve_codex(args.codex_home.expanduser(), args.thread_id)
    if args.session_id:
        return resolve_claude(args.claude_home.expanduser(), args.session_id)
    raise ResolutionError(
        "no current-session metadata: CODEX_THREAD_ID and CLAUDE_CODE_SESSION_ID are unset"
    )


def main() -> int:
    args = parse_args()
    try:
        attribution = resolve(args)
    except ResolutionError as error:
        print(f"attribution resolution failed: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(attribution.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(attribution.trailer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
