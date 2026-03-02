#!/usr/bin/env python3
"""Run translation benchmarks on a fixed 4-paper corpus.

CLI:
    # Run full benchmark (translate + validate)
    uv run scripts/run_benchmark.py --method bplus-v2

    # Validate existing translations only (no API calls)
    uv run scripts/run_benchmark.py --method bplus --skip-translate

    # Compare multiple methods
    uv run scripts/run_benchmark.py --compare baseline bplus bplus-v2

    # Run specific papers only
    uv run scripts/run_benchmark.py --method bplus-v2 --papers ttt3r cambrians
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Import validation API
from validate_translation import validate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Project root (research-utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_RESULTS_DIR = _PROJECT_ROOT / "benchmark-results"
_TRANSLATE_SCRIPT = Path(__file__).resolve().parent / "translate_paper.py"


@dataclass
class PaperInfo:
    short_name: str
    label: str
    dir_name: str
    difficulty: str

    @property
    def source_path(self) -> Path:
        return _OUTPUT_DIR / self.dir_name / "full_text.md"

    @property
    def translated_path(self) -> Path:
        return _OUTPUT_DIR / self.dir_name / "full_text_ch.md"


# Hardcoded test corpus registry
CORPUS: dict[str, PaperInfo] = {
    "vace": PaperInfo(
        short_name="vace",
        label="VACE",
        dir_name="20260301-VACE_All_in_One_Video_Creation_and_Editing",
        difficulty="easy",
    ),
    "ttt-e2e": PaperInfo(
        short_name="ttt-e2e",
        label="TTT-E2E",
        dir_name="20260301-End_to_End_Test_Time_Training_for_Long_Context",
        difficulty="easy",
    ),
    "ttt3r": PaperInfo(
        short_name="ttt3r",
        label="TTT3R",
        dir_name="20260301-TTT3R_3D_RECONSTRUCTION_AS_TEST_TIME_TRAINING",
        difficulty="hard",
    ),
    "cambrians": PaperInfo(
        short_name="cambrians",
        label="CambrianS",
        dir_name="20260301-CambrianS_Towards_Spatial_Supersensing_in_Video",
        difficulty="hard",
    ),
}


def _get_git_commit() -> str:
    """Get short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Translation runner
# ---------------------------------------------------------------------------

def _run_translate(paper: PaperInfo, backend: str) -> dict | None:
    """Run translate_paper.py and return parsed JSON output, or None on error."""
    cmd = [
        "uv", "run", str(_TRANSLATE_SCRIPT),
        str(paper.source_path),
        "--backend", backend,
    ]
    print(f"  Translating {paper.label}...", file=sys.stderr)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=_TRANSLATE_SCRIPT.parent.parent,  # paper-translate/
        )
        # translate_paper.py outputs JSON to stdout
        if result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if data.get("status") == "success":
                return data
            print(f"  ERROR: {data.get('message', 'unknown')}", file=sys.stderr)
            return None
        print(f"  ERROR: No stdout from translate_paper.py", file=sys.stderr)
        if result.stderr:
            # Print last few lines of stderr for debugging
            lines = result.stderr.strip().splitlines()
            for line in lines[-5:]:
                print(f"    {line}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Translation timed out (600s)", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON from translate_paper.py: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _run_benchmark(
    method: str, papers: list[PaperInfo], backend: str, skip_translate: bool,
) -> dict:
    """Run benchmark on specified papers.

    Returns the full result dict ready for JSON serialization.
    """
    meta = {
        "method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "git_commit": _get_git_commit(),
    }

    paper_results = []
    total_cost = 0.0
    total_time = 0.0
    total_score = 0.0
    passed_count = 0

    for paper in papers:
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Paper: {paper.label} ({paper.difficulty})", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        # Translate (or skip)
        perf_data = None
        if not skip_translate:
            perf_data = _run_translate(paper, backend)
            if perf_data is None:
                print(f"  Skipping validation (translation failed)", file=sys.stderr)
                paper_results.append({
                    "paper": {
                        "short_name": paper.short_name,
                        "label": paper.label,
                        "difficulty": paper.difficulty,
                    },
                    "performance": None,
                    "quality": None,
                    "error": "Translation failed",
                })
                continue

        # Validate
        if not paper.translated_path.exists():
            print(f"  Skipping validation (no translated file)", file=sys.stderr)
            paper_results.append({
                "paper": {
                    "short_name": paper.short_name,
                    "label": paper.label,
                    "difficulty": paper.difficulty,
                },
                "performance": perf_data,
                "quality": None,
                "error": "Translated file not found",
            })
            continue

        print(f"  Validating {paper.label}...", file=sys.stderr)
        quality = validate(paper.source_path, paper.translated_path)
        quality_dict = quality.to_dict()

        # Accumulate stats
        total_score += quality.score
        if quality.passed:
            passed_count += 1
        if perf_data:
            cost = perf_data.get("estimated_cost_usd", 0)
            total_cost += cost if cost else 0
            total_time += perf_data.get("timing", {}).get("total_sec", 0)

        print(
            f"  Score: {quality.score}/100 "
            f"({'PASSED' if quality.passed else 'FAILED'})",
            file=sys.stderr,
        )
        if quality.hard_failures:
            for hf in quality.hard_failures:
                print(f"    HARD FAIL: {hf}", file=sys.stderr)

        paper_results.append({
            "paper": {
                "short_name": paper.short_name,
                "label": paper.label,
                "difficulty": paper.difficulty,
            },
            "performance": perf_data,
            "quality": quality_dict,
        })

    # Summary
    n = len([p for p in paper_results if p.get("quality")])
    summary = {
        "total_cost_usd": round(total_cost, 4) if total_cost > 0 else None,
        "total_time_sec": round(total_time, 1) if total_time > 0 else None,
        "avg_score": round(total_score / n, 1) if n > 0 else None,
        "papers_passed": passed_count,
        "papers_failed": n - passed_count,
    }

    return {"meta": meta, "papers": paper_results, "summary": summary}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _load_result(method: str) -> dict | None:
    """Load benchmark result JSON for a method."""
    # Try exact filename, then _quality variant
    for suffix in ["", "_quality"]:
        path = _RESULTS_DIR / f"{method}{suffix}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _build_comparison_table(methods: list[str]) -> str:
    """Build markdown comparison table from method results."""
    results: dict[str, dict] = {}
    for m in methods:
        data = _load_result(m)
        if data is None:
            print(f"WARNING: No result found for method '{m}'", file=sys.stderr)
            continue
        results[m] = data

    if len(results) < 2:
        return "ERROR: Need at least 2 methods with results to compare.\n"

    # Collect all paper names across methods
    all_papers: list[str] = []
    for data in results.values():
        for p in data.get("papers", []):
            name = p["paper"]["short_name"]
            if name not in all_papers:
                all_papers.append(name)

    # Build table
    method_names = list(results.keys())
    lines: list[str] = []
    lines.append(f"# Benchmark Comparison: {' vs '.join(method_names)}")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # Score comparison table
    header = "| Paper     |"
    sep = "|-----------|"
    for i, m in enumerate(method_names):
        header += f" {m} score |"
        sep += "------------|"
        if i < len(method_names) - 1:
            header += f" {method_names[i+1]} score |"
            sep += "------------|"
            header += " Δ     |"
            sep += "-------|"
            break  # For now support pairwise comparison
    lines.append(header)
    lines.append(sep)

    # Build pairwise comparison rows
    if len(method_names) >= 2:
        m1, m2 = method_names[0], method_names[1]
        scores1: dict[str, float] = {}
        scores2: dict[str, float] = {}
        times1: dict[str, float] = {}
        times2: dict[str, float] = {}

        for p in results[m1].get("papers", []):
            name = p["paper"]["short_name"]
            if p.get("quality"):
                scores1[name] = p["quality"]["score"]
            if p.get("performance"):
                times1[name] = p["performance"].get("timing", {}).get("total_sec", 0)

        for p in results[m2].get("papers", []):
            name = p["paper"]["short_name"]
            if p.get("quality"):
                scores2[name] = p["quality"]["score"]
            if p.get("performance"):
                times2[name] = p["performance"].get("timing", {}).get("total_sec", 0)

        # Score table
        lines_scores = [
            f"## Quality Scores",
            "",
            f"| Paper     | {m1} score | {m2} score | Δ     |",
            f"|-----------|------------|------------|-------|",
        ]
        avg1, avg2, count = 0.0, 0.0, 0
        for name in all_papers:
            s1 = scores1.get(name)
            s2 = scores2.get(name)
            s1_str = f"{s1:.1f}" if s1 is not None else "—"
            s2_str = f"{s2:.1f}" if s2 is not None else "—"
            if s1 is not None and s2 is not None:
                delta = s2 - s1
                d_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
                avg1 += s1
                avg2 += s2
                count += 1
            else:
                d_str = "—"
            lines_scores.append(
                f"| {name:<9} | {s1_str:>10} | {s2_str:>10} | {d_str:>5} |"
            )
        if count > 0:
            a1 = avg1 / count
            a2 = avg2 / count
            d = a2 - a1
            d_str = f"+{d:.1f}" if d >= 0 else f"{d:.1f}"
            lines_scores.append(
                f"| **Avg**   | **{a1:.1f}**    | **{a2:.1f}**    | {d_str} |"
            )
        lines_scores.append("")

        # Time table (if available)
        lines_time = []
        if times1 or times2:
            lines_time = [
                f"## Performance",
                "",
                f"| Paper     | {m1} time | {m2} time | Δ    |",
                f"|-----------|-----------|-----------|------|",
            ]
            tavg1, tavg2, tcount = 0.0, 0.0, 0
            for name in all_papers:
                t1 = times1.get(name)
                t2 = times2.get(name)
                t1_str = f"{t1:.1f}s" if t1 is not None else "—"
                t2_str = f"{t2:.1f}s" if t2 is not None else "—"
                if t1 is not None and t2 is not None and t1 > 0:
                    pct = ((t2 - t1) / t1) * 100
                    d_str = f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%"
                    tavg1 += t1
                    tavg2 += t2
                    tcount += 1
                else:
                    d_str = "—"
                lines_time.append(
                    f"| {name:<9} | {t1_str:>9} | {t2_str:>9} | {d_str:>4} |"
                )
            if tcount > 0:
                lines_time.append(
                    f"| **Avg**   | **{tavg1/tcount:.1f}s** | **{tavg2/tcount:.1f}s** |      |"
                )
            lines_time.append("")

        lines = lines_scores + lines_time

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run translation benchmarks.")
    parser.add_argument(
        "--method", type=str,
        help="Method name for labeling results (e.g., bplus, bplus-v2)",
    )
    parser.add_argument(
        "--backend", type=str, default="tensorblock",
        help="Translation backend (default: tensorblock)",
    )
    parser.add_argument(
        "--skip-translate", action="store_true",
        help="Skip translation, validate existing translations only",
    )
    parser.add_argument(
        "--papers", nargs="+", type=str,
        help="Specific papers to run (e.g., ttt3r cambrians). Default: all",
    )
    parser.add_argument(
        "--compare", nargs="+", type=str, metavar="METHOD",
        help="Compare results from multiple methods (e.g., --compare baseline bplus)",
    )
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        table = _build_comparison_table(args.compare)
        print(table)
        # Save to file
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        comp_path = _RESULTS_DIR / "comparison.md"
        comp_path.write_text(table, encoding="utf-8")
        print(f"Saved to {comp_path}", file=sys.stderr)
        return

    # Benchmark mode requires --method
    if not args.method:
        parser.error("--method is required (unless using --compare)")

    # Select papers
    if args.papers:
        papers = []
        for name in args.papers:
            if name not in CORPUS:
                parser.error(
                    f"Unknown paper '{name}'. Available: {list(CORPUS.keys())}"
                )
            papers.append(CORPUS[name])
    else:
        papers = list(CORPUS.values())

    # Run benchmark
    result = _run_benchmark(args.method, papers, args.backend, args.skip_translate)

    # Save results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if not args.skip_translate else "_quality"
    out_path = _RESULTS_DIR / f"{args.method}{suffix}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Print summary
    summary = result["summary"]
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"BENCHMARK SUMMARY: {args.method}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    if summary["avg_score"] is not None:
        print(f"  Avg Score:     {summary['avg_score']}/100", file=sys.stderr)
    print(
        f"  Passed:        {summary['papers_passed']}/{summary['papers_passed'] + summary['papers_failed']}",
        file=sys.stderr,
    )
    if summary["total_time_sec"]:
        print(f"  Total Time:    {summary['total_time_sec']}s", file=sys.stderr)
    if summary["total_cost_usd"]:
        print(f"  Total Cost:    ${summary['total_cost_usd']}", file=sys.stderr)
    print(f"  Results saved: {out_path}", file=sys.stderr)

    # JSON to stdout
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
