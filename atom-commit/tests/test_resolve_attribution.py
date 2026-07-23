from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "resolve_attribution.py"
SPEC = importlib.util.spec_from_file_location("resolve_attribution", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResolveAttributionTests(unittest.TestCase):
    thread_id = "019f86d9-abc5-7b20-bb33-edd761faed3f"
    claude_session_id = "be751d5b-290a-4c96-b1ca-b1221e08a889"

    def write_rollout(self, root: Path, entries: list[dict[str, object]]) -> Path:
        rollout = (
            root
            / "sessions"
            / "2026"
            / "07"
            / "21"
            / f"rollout-2026-07-21T15-44-00-{self.thread_id}.jsonl"
        )
        rollout.parent.mkdir(parents=True)
        rollout.write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries),
            encoding="utf-8",
        )
        return rollout

    def test_resolves_latest_codex_desktop_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_rollout(
                root,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": self.thread_id,
                            "originator": "Codex Desktop",
                        },
                    },
                    {
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.5"},
                    },
                    {
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.6-sol"},
                    },
                ],
            )

            attribution = MODULE.resolve_codex(root, self.thread_id)

            self.assertEqual(attribution.model, "gpt-5.6-sol")
            self.assertEqual(attribution.harness, "Codex App")
            self.assertEqual(
                attribution.trailer,
                "Assisted-By: gpt-5.6-sol via Codex App",
            )

    def test_maps_codex_tui_to_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_rollout(
                root,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": self.thread_id,
                            "originator": "codex-tui",
                        },
                    },
                    {
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.6-sol"},
                    },
                ],
            )

            attribution = MODULE.resolve_codex(root, self.thread_id)

            self.assertEqual(attribution.harness, "Codex CLI")

    def test_rejects_rollout_with_wrong_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_rollout(
                root,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "019f86d9-abc5-7b20-bb33-000000000000",
                            "originator": "Codex Desktop",
                        },
                    },
                    {
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.6-sol"},
                    },
                ],
            )

            with self.assertRaisesRegex(MODULE.ResolutionError, "expected session id"):
                MODULE.resolve_codex(root, self.thread_id)

    def test_resolves_latest_claude_model_and_routing_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transcript = (
                root
                / "projects"
                / "-Users-reed-Code-project"
                / f"{self.claude_session_id}.jsonl"
            )
            transcript.parent.mkdir(parents=True)
            entries = [
                {
                    "type": "assistant",
                    "sessionId": self.claude_session_id,
                    "message": {"model": "claude-sonnet-4-6"},
                },
                {
                    "type": "assistant",
                    "sessionId": self.claude_session_id,
                    "message": {"model": "claude-gpt-5-6-sol"},
                },
            ]
            transcript.write_text(
                "".join(json.dumps(entry) + "\n" for entry in entries),
                encoding="utf-8",
            )

            attribution = MODULE.resolve_claude(root, self.claude_session_id)

            self.assertEqual(attribution.model, "gpt-5.6-sol")
            self.assertEqual(attribution.harness, "Claude Code")
            self.assertEqual(
                attribution.trailer,
                "Assisted-By: gpt-5.6-sol via Claude Code",
            )

    def test_cli_fails_without_current_session_metadata(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True,
            check=False,
            env={},
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("no current-session metadata", result.stderr)


if __name__ == "__main__":
    unittest.main()
