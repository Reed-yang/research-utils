#!/usr/bin/env python3
"""Validate translation quality by comparing source and translated markdown files.

CLI:
    uv run scripts/validate_translation.py <source.md> <translated.md> [--json] [--verbose]

API:
    from validate_translation import validate, ValidationResult
    result = validate(source_path, translated_path)
    print(result.score, result.passed, result.hard_failures)

Not exposed in SKILL.md — this is an internal validation tool for the benchmark system.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from markdown_it import MarkdownIt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unicode math symbols that might appear outside $...$ in poorly formatted papers
UNICODE_MATH_RE = re.compile(
    r"[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ∇∈∀∃∑∏∫∞≈≠≤≥±×÷·]"
)

# Partial-wrap pattern: $β$ t style (single Greek letter in $ followed by space+text)
PARTIAL_WRAP_RE = re.compile(r"\$[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]\$\s*\w")

# GLYPH marker from ingestion
GLYPH_RE = re.compile(r"\[GLYPH:<[^>]+>\]")

# Code block regions
CODE_BLOCK_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)

# Display math regions
DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)

# Inline math
INLINE_MATH_RE = re.compile(r"\$([^$\n]+?)\$")

# Image references
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Table rows (pipes)
TABLE_ROW_RE = re.compile(r"^\|.*\|$", re.MULTILINE)

# YAML frontmatter
FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StructuralMetrics:
    display_math_count: int = 0
    inline_math_count: int = 0
    inline_formula_fingerprints: set[str] = field(default_factory=set)
    heading_count: int = 0
    heading_breakdown: dict[int, int] = field(default_factory=dict)
    image_count: int = 0
    code_block_count: int = 0
    paragraph_count: int = 0
    table_row_count: int = 0
    char_count: int = 0
    line_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["inline_formula_fingerprints"] = sorted(d["inline_formula_fingerprints"])
        return d


@dataclass
class MathIssues:
    missing_formulas: list[str] = field(default_factory=list)
    extra_formulas: list[str] = field(default_factory=list)
    unicode_math_outside_dollar: int = 0
    partial_wrap_count: int = 0
    glyph_marker_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectionInfo:
    heading: str
    level: int
    source_chars: int
    translated_chars: int
    ratio: float


@dataclass
class ContentMetrics:
    length_ratio: float = 0.0
    sections: list[SectionInfo] = field(default_factory=list)
    empty_sections: list[str] = field(default_factory=list)
    untranslated_lines: int = 0

    def to_dict(self) -> dict:
        return {
            "length_ratio": self.length_ratio,
            "sections": [
                {
                    "heading": s.heading,
                    "level": s.level,
                    "source_chars": s.source_chars,
                    "translated_chars": s.translated_chars,
                    "ratio": s.ratio,
                }
                for s in self.sections
            ],
            "empty_sections": self.empty_sections,
            "untranslated_lines": self.untranslated_lines,
        }


@dataclass
class ValidationResult:
    source: str
    translated: str
    score: float
    passed: bool
    hard_failures: list[str]
    issues: list[str]
    warnings: list[str]
    structural_source: StructuralMetrics
    structural_translated: StructuralMetrics
    math_issues: MathIssues
    content: ContentMetrics

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "translated": self.translated,
            "score": round(self.score, 1),
            "passed": self.passed,
            "hard_failures": self.hard_failures,
            "issues": self.issues,
            "warnings": self.warnings,
            "structural": {
                "source": self.structural_source.to_dict(),
                "translated": self.structural_translated.to_dict(),
            },
            "math_issues": self.math_issues.to_dict(),
            "content": self.content.to_dict(),
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter."""
    return FRONTMATTER_RE.sub("", text, count=1)


def _strip_code_blocks(text: str) -> str:
    """Remove code blocks from text for analysis."""
    return CODE_BLOCK_RE.sub("", text)


def _strip_display_math(text: str) -> str:
    """Remove display math blocks from text."""
    return DISPLAY_MATH_RE.sub("", text)


def _collect_headings(text: str) -> list[tuple[int, int, int, str]]:
    """Collect markdown headings using markdown-it-py.

    Returns list of (start_line, end_line, level, heading_text).
    Lines are 0-indexed.
    """
    md = MarkdownIt("commonmark")
    tokens = md.parse(text)
    headings: list[tuple[int, int, int, str]] = []
    for i, token in enumerate(tokens):
        if token.type != "heading_open" or not token.map:
            continue
        level = int(token.tag[1]) if token.tag and token.tag.startswith("h") else 6
        inline = tokens[i + 1] if i + 1 < len(tokens) else None
        heading_text = inline.content if inline and inline.type == "inline" else ""
        start_line, end_line = token.map
        headings.append((start_line, end_line, level, heading_text))
    return headings


def _normalize_formula(s: str) -> str:
    """Normalize a formula fingerprint for comparison.

    Strips whitespace and trailing punctuation that OCR/LLM may add/remove.
    """
    s = s.strip()
    # Strip trailing sentence punctuation (period, comma, semicolon)
    s = s.rstrip(".,;:")
    return s


def _strip_references_section(text: str) -> str:
    """Remove the References section from text for analysis.

    References contain many formulas and author names that pollute fingerprints.
    """
    lines = text.splitlines()
    headings = _collect_headings(text)
    for idx, (_start, _end, _level, heading_text) in enumerate(headings):
        h_lower = heading_text.strip().lower()
        h_stripped = _SECTION_KEY_RE.sub("", h_lower).strip()
        if h_stripped in ("references", "bibliography", "参考文献"):
            ref_start = _start
            # Find end: next heading of any level, or end of document
            ref_end = len(lines)
            for next_start, _ne, _nl, _ in headings[idx + 1:]:
                ref_end = next_start
                break
            # Remove the References lines
            return "\n".join(lines[:ref_start] + lines[ref_end:])
    return text


def _extract_structural(text: str, exclude_references: bool = False) -> StructuralMetrics:
    """Extract structural metrics from markdown text."""
    body = _strip_frontmatter(text)
    m = StructuralMetrics()

    m.char_count = len(body)
    m.line_count = body.count("\n") + 1

    # Display math
    m.display_math_count = len(DISPLAY_MATH_RE.findall(body))

    # Code blocks
    m.code_block_count = len(CODE_BLOCK_RE.findall(body))

    # For inline math, strip display math and code blocks first
    # Optionally exclude References section (formulas in citations pollute fingerprints)
    analysis_body = _strip_references_section(body) if exclude_references else body
    no_display = _strip_display_math(analysis_body)
    no_code = _strip_code_blocks(no_display)
    inline_matches = INLINE_MATH_RE.findall(no_code)
    m.inline_math_count = len(inline_matches)
    m.inline_formula_fingerprints = {_normalize_formula(s) for s in inline_matches if _normalize_formula(s)}

    # Headings — exclude References heading when exclude_references is True
    # (translation pipeline strips References section, so it's expected to be missing)
    headings = _collect_headings(body)
    if exclude_references:
        headings = [
            h for h in headings
            if _SECTION_KEY_RE.sub("", h[3].strip().lower()).strip()
            not in ("references", "bibliography", "参考文献")
        ]
    m.heading_count = len(headings)
    for _, _, level, _ in headings:
        m.heading_breakdown[level] = m.heading_breakdown.get(level, 0) + 1

    # Images
    m.image_count = len(IMAGE_RE.findall(body))

    # Paragraphs: blocks of text separated by blank lines (rough estimate)
    no_code_body = _strip_code_blocks(body)
    paragraphs = [p.strip() for p in no_code_body.split("\n\n") if p.strip()]
    m.paragraph_count = len(paragraphs)

    # Table rows
    m.table_row_count = len(TABLE_ROW_RE.findall(body))

    return m


def _extract_math_issues(
    source_text: str, translated_text: str,
    source_metrics: StructuralMetrics, translated_metrics: StructuralMetrics,
) -> MathIssues:
    """Detect math-related issues."""
    issues = MathIssues()

    # Formula fingerprint diff
    missing = source_metrics.inline_formula_fingerprints - translated_metrics.inline_formula_fingerprints
    extra = translated_metrics.inline_formula_fingerprints - source_metrics.inline_formula_fingerprints
    issues.missing_formulas = sorted(missing)
    issues.extra_formulas = sorted(extra)

    # Unicode math outside $ in source (informational)
    source_body = _strip_frontmatter(source_text)
    source_no_math = _strip_display_math(_strip_code_blocks(source_body))
    # Remove inline math too
    source_no_inline = INLINE_MATH_RE.sub("", source_no_math)
    issues.unicode_math_outside_dollar = len(UNICODE_MATH_RE.findall(source_no_inline))

    # Partial-wrap patterns in source (informational)
    issues.partial_wrap_count = len(PARTIAL_WRAP_RE.findall(source_no_math))

    # GLYPH markers in translated
    translated_body = _strip_frontmatter(translated_text)
    issues.glyph_marker_count = len(GLYPH_RE.findall(translated_body))

    return issues


def _split_sections(text: str) -> list[tuple[str, int, str]]:
    """Split text into sections at heading boundaries.

    Returns list of (heading_text, level, body_text).
    First entry has heading="" level=0 for content before first heading.
    """
    body = _strip_frontmatter(text)
    lines = body.splitlines()
    headings = _collect_headings(body)

    sections: list[tuple[str, int, str]] = []

    if not headings:
        return [("", 0, body)]

    # Content before first heading
    first_start = headings[0][0]
    if first_start > 0:
        pre = "\n".join(lines[:first_start])
        sections.append(("", 0, pre))

    for idx, (start, _end, level, heading_text) in enumerate(headings):
        if idx + 1 < len(headings):
            next_start = headings[idx + 1][0]
        else:
            next_start = len(lines)
        section_body = "\n".join(lines[start:next_start])
        sections.append((heading_text, level, section_body))

    return sections


# Standard heading translation mappings for section alignment
_HEADING_TRANSLATIONS = {
    "abstract": "摘要",
    "introduction": "引言",
    "related work": "相关工作",
    "method": "方法",
    "methods": "方法",
    "methodology": "方法论",
    "experiments": "实验",
    "experiment": "实验",
    "results": "结果",
    "result": "结果",
    "discussion": "讨论",
    "conclusion": "结论",
    "conclusions": "结论",
    "acknowledgments": "致谢",
    "acknowledgements": "致谢",
    "references": "参考文献",
    "appendix": "附录",
    "supplementary material": "补充材料",
    "limitations": "局限性",
    "broader impact": "更广泛的影响",
    "evaluation": "评估",
    "analysis": "分析",
    "approach": "方法",
    "implementation": "实现",
    "overview": "概述",
    "background": "背景",
    "preliminaries": "预备知识",
    "setup": "设置",
    "ablation": "消融实验",
    "ablation study": "消融研究",
    "visualization": "可视化",
    "qualitative results": "定性结果",
    "quantitative results": "定量结果",
    "contents": "目录",
    "table of contents": "目录",
    "acknowledgment": "致谢",
    "related works": "相关工作",
    "training": "训练",
    "inference": "推理",
    "dataset": "数据集",
    "datasets": "数据集",
    "implementation details": "实现细节",
    "training recipe": "训练方案",
    "infrastructure": "基础设施",
    "model architecture": "模型架构",
    "author contributions": "作者贡献",
    "ethics statement": "伦理声明",
    "reproducibility": "可复现性",
    "reproducibility statement": "可复现性声明",
    "broader impacts": "更广泛的影响",
    "societal impacts": "社会影响",
    "societal impact": "社会影响",
    "training details": "训练细节",
    "more results": "更多结果",
    "additional results": "补充结果",
    "additional details": "补充细节",
    "state reset": "状态重置",
}

# Section key regex — extracts the number/letter prefix from headings
# Matches: "1", "2.3", "A.1", "B.2.1", "A.", "B.", "A " (single letter appendix)
_SECTION_KEY_RE = re.compile(
    r"^("
    r"[A-Z]\.\d[\d.]*"     # A.1, A.1.2, B.2.1
    r"|[A-Z](?=[\s.])"     # A, B, C (single letter before space or dot)
    r"|\d[\d.]*"            # 1, 2.3, 3.2.1
    r")[\s.]*"
)


def _extract_heading_key(heading: str) -> str | None:
    """Extract a matching key from a heading for cross-language alignment.

    Returns a normalized section number prefix if present, otherwise None.
    Examples: "3.2 METHOD" -> "3.2", "A.1. Hyperparameters" -> "A.1",
              "A. Implementation" -> "A", "1 INTRODUCTION" -> "1"
    """
    m = _SECTION_KEY_RE.match(heading.strip())
    if m:
        return m.group(1).rstrip(".")
    return None


def _align_sections(
    source_sections: list[tuple[str, int, str]],
    translated_sections: list[tuple[str, int, str]],
) -> list[tuple[tuple[str, int, str] | None, tuple[str, int, str] | None]]:
    """Align source and translated sections using heading text matching.

    Uses a multi-strategy approach:
    1. Section number prefix matching (e.g., "3.2" in "3.2 METHOD" vs "3.2 方法")
    2. Known heading translation table
    3. Fallback: unmatched sections reported as source-only or translation-only

    Returns list of (source_section, translated_section) pairs.
    Either element can be None if unmatched.
    """
    # Build lookup for translated sections
    tgt_by_num: dict[str, int] = {}   # number prefix -> index
    tgt_by_kw: dict[str, int] = {}    # normalized keyword -> index
    tgt_used: set[int] = set()

    for i, (heading, _level, _body) in enumerate(translated_sections):
        key = _extract_heading_key(heading)
        if key and key not in tgt_by_num:
            tgt_by_num[key] = i
        # Check if heading matches a known translation
        h_lower = heading.strip().lower()
        # Strip number prefix for keyword matching
        h_stripped = _SECTION_KEY_RE.sub("", h_lower).strip()
        for _eng, cn in _HEADING_TRANSLATIONS.items():
            if cn in h_stripped:
                if cn not in tgt_by_kw:
                    tgt_by_kw[cn] = i
                break

    aligned: list[tuple[tuple[str, int, str] | None, tuple[str, int, str] | None]] = []

    # Track first real heading for title matching (any level)
    first_src_heading_idx = None
    first_tgt_heading_idx = None
    for i, (h, l, _b) in enumerate(source_sections):
        if h:
            first_src_heading_idx = i
            break
    for i, (h, l, _b) in enumerate(translated_sections):
        if h:
            first_tgt_heading_idx = i
            break

    for src_idx, src_section in enumerate(source_sections):
        s_heading = src_section[0]
        matched_idx = None

        # Strategy 1: Match by section number prefix
        s_key = _extract_heading_key(s_heading)
        if s_key and s_key in tgt_by_num:
            idx = tgt_by_num[s_key]
            if idx not in tgt_used:
                matched_idx = idx

        # Strategy 2: Match by known heading translation
        if matched_idx is None:
            s_lower = s_heading.strip().lower()
            s_stripped = _SECTION_KEY_RE.sub("", s_lower).strip()
            for eng, cn in _HEADING_TRANSLATIONS.items():
                if eng in s_stripped:
                    if cn in tgt_by_kw and tgt_by_kw[cn] not in tgt_used:
                        matched_idx = tgt_by_kw[cn]
                    break

        # Strategy 3: Match preamble (empty heading) to preamble
        if matched_idx is None and s_heading == "":
            for i, (th, _tl, _tb) in enumerate(translated_sections):
                if th == "" and i not in tgt_used:
                    matched_idx = i
                    break

        # Strategy 4: Match paper title (first level-1 heading) by position
        if matched_idx is None and first_src_heading_idx == src_idx:
            if first_tgt_heading_idx is not None and first_tgt_heading_idx not in tgt_used:
                matched_idx = first_tgt_heading_idx

        # Strategy 5: "References" section — skip it (stripped from translation)
        if matched_idx is None:
            s_lower = s_heading.strip().lower()
            s_stripped = _SECTION_KEY_RE.sub("", s_lower).strip()
            if s_stripped in ("references", "bibliography"):
                # Expected to be missing in translation — don't count as empty
                aligned.append((src_section, src_section))  # self-match to avoid false empty
                continue

        if matched_idx is not None:
            tgt_used.add(matched_idx)
            aligned.append((src_section, translated_sections[matched_idx]))
        else:
            aligned.append((src_section, None))

    # Append unmatched translated sections
    for i, tgt_section in enumerate(translated_sections):
        if i not in tgt_used:
            aligned.append((None, tgt_section))

    return aligned


def _is_untranslated_line(line: str) -> bool:
    """Heuristic: line is untranslated English if >60% ASCII alpha and len > 20.

    Excludes math-heavy lines, table rows, HTML, headings, and image refs.
    """
    stripped = line.strip()
    if len(stripped) <= 20:
        return False
    # Skip lines that look like markdown artifacts or structural elements
    if stripped.startswith(("![", "http", "|", "```", "$$", "<", "#")):
        return False
    # Skip lines with significant inline math (>3 dollar signs)
    if stripped.count("$") >= 3:
        return False
    # Strip inline math before checking ASCII ratio
    no_math = INLINE_MATH_RE.sub("", stripped)
    no_math = DISPLAY_MATH_RE.sub("", no_math)
    if len(no_math) <= 10:
        return False
    alpha_chars = sum(1 for c in no_math if c.isascii() and c.isalpha())
    total_chars = len(no_math)
    return (alpha_chars / total_chars) > 0.6 if total_chars > 0 else False


def _count_untranslated_lines(translated_text: str) -> int:
    """Count lines that appear to be untranslated English in the translation."""
    body = _strip_frontmatter(translated_text)
    # Remove code blocks and display math
    cleaned = _strip_code_blocks(body)
    cleaned = _strip_display_math(cleaned)

    count = 0
    in_code = False
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if _is_untranslated_line(stripped):
            count += 1
    return count


def _extract_content(
    source_text: str, translated_text: str,
) -> ContentMetrics:
    """Extract content quality metrics."""
    cm = ContentMetrics()

    source_body = _strip_frontmatter(source_text)
    translated_body = _strip_frontmatter(translated_text)

    # Overall length ratio
    if len(source_body) > 0:
        cm.length_ratio = round(len(translated_body) / len(source_body), 3)

    # Per-section analysis with heading-text alignment
    source_sections = _split_sections(source_text)
    translated_sections = _split_sections(translated_text)
    aligned = _align_sections(source_sections, translated_sections)

    for src_sec, tgt_sec in aligned:
        if src_sec is not None:
            s_heading, s_level, s_body = src_sec
        else:
            s_heading, s_level, s_body = "", 0, ""
        if tgt_sec is not None:
            t_heading, t_level, t_body = tgt_sec
        else:
            t_heading, t_level, t_body = "", 0, ""

        heading = s_heading or t_heading
        level = s_level or t_level
        s_chars = len(s_body.strip())
        t_chars = len(t_body.strip())
        ratio = round(t_chars / s_chars, 3) if s_chars > 0 else 0.0

        si = SectionInfo(
            heading=heading, level=level,
            source_chars=s_chars, translated_chars=t_chars, ratio=ratio,
        )
        cm.sections.append(si)

        # Empty section detection — only for source sections with no match
        # AND only for major sections (Introduction, Method, etc.)
        # Non-major unmatched sections are likely merged or table captions
        if src_sec is not None and tgt_sec is None:
            if level <= 2 and s_chars > 100 and _is_major_section(heading):
                cm.empty_sections.append(f"## {heading}" if heading else "(preamble)")

    # Untranslated residue
    cm.untranslated_lines = _count_untranslated_lines(translated_text)

    return cm


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_MAJOR_SECTION_KEYWORDS = [
    "introduction", "method", "experiment", "result", "conclusion",
    "approach", "evaluation", "discussion", "abstract",
    "引言", "方法", "实验", "结果", "结论", "摘要",
]


def _is_major_section(heading: str) -> bool:
    """Check if a heading is a major section (word-boundary matching)."""
    h_lower = heading.lower()
    for kw in _MAJOR_SECTION_KEYWORDS:
        # Use word boundary to avoid "model" matching "LARGE LANGUAGE MODELS"
        if re.search(rf"\b{re.escape(kw)}\b", h_lower):
            return True
    return False


def _compute_score(
    src: StructuralMetrics, tgt: StructuralMetrics,
    math: MathIssues, content: ContentMetrics,
) -> tuple[float, bool, list[str], list[str], list[str]]:
    """Compute validation score (0-100, penalty-based).

    Returns (score, passed, hard_failures, issues, warnings).
    """
    score = 100.0
    hard_failures: list[str] = []
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Heading preservation (20 pts)
    if src.heading_count > 0:
        if tgt.heading_count < src.heading_count:
            lost = src.heading_count - tgt.heading_count
            penalty = min(20, 20 * (lost / src.heading_count))
            score -= penalty
            hard_failures.append(
                f"Headings lost: {src.heading_count} → {tgt.heading_count} (-{lost})"
            )
            issues.append(f"Heading count: {src.heading_count} → {tgt.heading_count}")
        elif tgt.heading_count > src.heading_count:
            extra = tgt.heading_count - src.heading_count
            penalty = min(10, 5 * extra)
            score -= penalty
            warnings.append(
                f"Extra headings: {src.heading_count} → {tgt.heading_count} (+{extra})"
            )

    # 2. Display math preservation (15 pts)
    if src.display_math_count > 0:
        diff = abs(src.display_math_count - tgt.display_math_count)
        if diff > 0:
            penalty = min(15, 15 * (diff / src.display_math_count))
            score -= penalty
            issues.append(
                f"Display math: {src.display_math_count} → {tgt.display_math_count}"
            )

    # 3. Missing formulas / fingerprint (15 pts)
    if math.missing_formulas:
        count = len(math.missing_formulas)
        total = len(src.inline_formula_fingerprints) or 1
        penalty = min(15, 15 * (count / total))
        score -= penalty
        hard_failures.append(f"Missing formulas: {count} lost")
        preview = math.missing_formulas[:5]
        issues.append(f"Missing formulas ({count}): {preview}")

    # 4. Extra $ wrapping (10 pts) — soft, reduced penalty
    # Unicode→LaTeX conversions (α→$\alpha$) are quality improvements,
    # so only lightly penalize large counts.
    if math.extra_formulas:
        count = len(math.extra_formulas)
        # Gentle penalty: 0.5 per extra formula, capped at 10
        penalty = min(10, 0.5 * count)
        score -= penalty
        if count > 10:
            issues.append(f"Extra $ wrapping: {count} instances")
        else:
            warnings.append(f"Extra $ wrapping: {count} instances")

    # 5. Overall length ratio (10 pts)
    # Chinese text is typically 0.3-0.7x the char count of English.
    # References section is often stripped, further reducing the ratio.
    lr = content.length_ratio
    if lr < 0.1:
        score -= 10
        hard_failures.append(f"Catastrophic length ratio: {lr:.2f}")
    elif lr < 0.2:
        score -= 7
        warnings.append(f"Very low length ratio: {lr:.2f}")
    elif lr < 0.25:
        score -= 4
        warnings.append(f"Low length ratio: {lr:.2f}")
    elif lr > 2.0:
        score -= 5
        warnings.append(f"High length ratio: {lr:.2f}")

    # 6. Empty major sections (10 pts)
    for section in content.sections:
        if section.level <= 2 and _is_major_section(section.heading):
            if section.source_chars > 100 and section.translated_chars == 0:
                score -= 10
                hard_failures.append(f"Empty section: ## {section.heading}")
                break  # count once for hard fail

    # Additional penalty for non-major empty sections
    if content.empty_sections:
        non_major = [
            s for s in content.empty_sections
            if not any(kw in s.lower() for kw in _MAJOR_SECTION_KEYWORDS)
        ]
        if non_major:
            penalty = min(5, 2 * len(non_major))
            score -= penalty
            issues.append(f"Empty sections: {non_major}")

    # 7. Image preservation (5 pts)
    if src.image_count > 0:
        lost = src.image_count - tgt.image_count
        if lost > 0:
            penalty = min(5, 5 * (lost / src.image_count))
            score -= penalty
            hard_failures.append(
                f"Images lost: {src.image_count} → {tgt.image_count} (-{lost})"
            )
            issues.append(f"Image count: {src.image_count} → {tgt.image_count}")

    # 8. Paragraph structure (5 pts)
    if src.paragraph_count > 0:
        ratio = tgt.paragraph_count / src.paragraph_count
        if ratio < 0.5 or ratio > 2.0:
            penalty = min(5, 5 * abs(1 - ratio))
            score -= penalty
            issues.append(
                f"Paragraph count: {src.paragraph_count} → {tgt.paragraph_count} "
                f"(ratio {ratio:.2f})"
            )

    # 9. Untranslated residue (5 pts)
    if content.untranslated_lines > 0:
        penalty = min(5, content.untranslated_lines * 0.5)
        score -= penalty
        if content.untranslated_lines > 5:
            issues.append(f"Untranslated lines: {content.untranslated_lines}")
        else:
            warnings.append(f"Untranslated lines: {content.untranslated_lines}")

    # 10. Table preservation (5 pts)
    if src.table_row_count > 0:
        lost = src.table_row_count - tgt.table_row_count
        if lost > 0:
            penalty = min(5, 5 * (lost / src.table_row_count))
            score -= penalty
            issues.append(
                f"Table rows: {src.table_row_count} → {tgt.table_row_count} (-{lost})"
            )

    score = max(0, round(score, 1))
    passed = len(hard_failures) == 0

    return score, passed, hard_failures, issues, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(source_path: str | Path, translated_path: str | Path) -> ValidationResult:
    """Validate a translated markdown file against its source.

    Args:
        source_path: Path to the original markdown file.
        translated_path: Path to the translated markdown file.

    Returns:
        ValidationResult with score, pass/fail, and detailed metrics.
    """
    source_path = Path(source_path)
    translated_path = Path(translated_path)

    source_text = source_path.read_text(encoding="utf-8")
    translated_text = translated_path.read_text(encoding="utf-8")

    # Use exclude_references=True for formula fingerprint to avoid
    # polluting fingerprints with citation formulas from References section
    src_metrics = _extract_structural(source_text, exclude_references=True)
    tgt_metrics = _extract_structural(translated_text, exclude_references=True)
    math_issues = _extract_math_issues(source_text, translated_text, src_metrics, tgt_metrics)
    content = _extract_content(source_text, translated_text)

    score, passed, hard_failures, issues, warnings = _compute_score(
        src_metrics, tgt_metrics, math_issues, content,
    )

    return ValidationResult(
        source=str(source_path),
        translated=str(translated_path),
        score=score,
        passed=passed,
        hard_failures=hard_failures,
        issues=issues,
        warnings=warnings,
        structural_source=src_metrics,
        structural_translated=tgt_metrics,
        math_issues=math_issues,
        content=content,
    )


# ---------------------------------------------------------------------------
# CLI: Rich output
# ---------------------------------------------------------------------------

# ANSI color helpers
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print_rich(result: ValidationResult) -> None:
    """Print colored summary to stderr."""
    # Score header
    color = _GREEN if result.passed else _RED
    status = "PASSED" if result.passed else "FAILED"
    sys.stderr.write(
        f"\n{_BOLD}Validation: {color}{status}{_RESET}  "
        f"Score: {_BOLD}{result.score}/100{_RESET}\n\n"
    )

    # Structural comparison
    src = result.structural_source
    tgt = result.structural_translated
    sys.stderr.write(f"{_CYAN}Structural Comparison:{_RESET}\n")
    rows = [
        ("Headings", src.heading_count, tgt.heading_count),
        ("Display math ($$)", src.display_math_count, tgt.display_math_count),
        ("Inline math ($)", src.inline_math_count, tgt.inline_math_count),
        ("Images", src.image_count, tgt.image_count),
        ("Code blocks", src.code_block_count, tgt.code_block_count),
        ("Paragraphs", src.paragraph_count, tgt.paragraph_count),
        ("Table rows", src.table_row_count, tgt.table_row_count),
        ("Characters", src.char_count, tgt.char_count),
    ]
    sys.stderr.write(f"  {'Metric':<20} {'Source':>8} {'Translated':>12} {'Delta':>8}\n")
    sys.stderr.write(f"  {'─'*20} {'─'*8} {'─'*12} {'─'*8}\n")
    for name, s_val, t_val in rows:
        delta = t_val - s_val
        d_str = f"+{delta}" if delta > 0 else str(delta)
        d_color = _GREEN if delta == 0 else (_RED if delta < 0 else _YELLOW)
        sys.stderr.write(
            f"  {name:<20} {s_val:>8} {t_val:>12} {d_color}{d_str:>8}{_RESET}\n"
        )

    # Math issues
    mi = result.math_issues
    if mi.missing_formulas or mi.extra_formulas or mi.glyph_marker_count:
        sys.stderr.write(f"\n{_CYAN}Math Issues:{_RESET}\n")
        if mi.missing_formulas:
            sys.stderr.write(f"  {_RED}Missing formulas: {len(mi.missing_formulas)}{_RESET}\n")
            for f in mi.missing_formulas[:10]:
                sys.stderr.write(f"    - ${f}$\n")
        if mi.extra_formulas:
            sys.stderr.write(f"  {_YELLOW}Extra formulas: {len(mi.extra_formulas)}{_RESET}\n")
            for f in mi.extra_formulas[:10]:
                sys.stderr.write(f"    - ${f}$\n")
        if mi.glyph_marker_count:
            sys.stderr.write(f"  GLYPH markers: {mi.glyph_marker_count}\n")

    # Content
    sys.stderr.write(f"\n{_CYAN}Content:{_RESET}\n")
    sys.stderr.write(f"  Length ratio: {result.content.length_ratio:.3f}\n")
    if result.content.untranslated_lines:
        sys.stderr.write(
            f"  {_YELLOW}Untranslated lines: {result.content.untranslated_lines}{_RESET}\n"
        )

    # Hard failures
    if result.hard_failures:
        sys.stderr.write(f"\n{_RED}{_BOLD}Hard Failures:{_RESET}\n")
        for hf in result.hard_failures:
            sys.stderr.write(f"  {_RED}✗ {hf}{_RESET}\n")

    # Issues
    if result.issues:
        sys.stderr.write(f"\n{_YELLOW}Issues:{_RESET}\n")
        for issue in result.issues:
            sys.stderr.write(f"  • {issue}\n")

    # Warnings
    if result.warnings:
        sys.stderr.write(f"\nWarnings:\n")
        for w in result.warnings:
            sys.stderr.write(f"  ◦ {w}\n")

    sys.stderr.write("\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate translation quality by comparing source and translated markdown."
    )
    parser.add_argument("source", type=Path, help="Path to source markdown file")
    parser.add_argument("translated", type=Path, help="Path to translated markdown file")
    parser.add_argument("--json", action="store_true", help="Force JSON output to stdout")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print colored summary to stderr (auto-enabled for TTY)",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(json.dumps({"status": "error", "message": f"Source not found: {args.source}"}))
        sys.exit(1)
    if not args.translated.exists():
        print(json.dumps({"status": "error", "message": f"Translated not found: {args.translated}"}))
        sys.exit(1)

    result = validate(args.source, args.translated)

    # Rich output: --verbose or TTY stderr
    if args.verbose or (sys.stderr.isatty() and not args.json):
        _print_rich(result)

    # JSON output: always to stdout
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
