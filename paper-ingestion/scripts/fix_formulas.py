"""
Post-processing fixes for OCR-introduced LaTeX formula errors.

Rules are conservative and deterministic — only high-confidence patterns.
"""

from __future__ import annotations

import re

# Commands whose brace content should have single-letter spaces merged
_TEXT_COMMANDS = (
    r"\text",
    r"\mathrm",
    r"\operatorname",
    r"\textbf",
    r"\textit",
    r"\mathit",
    r"\mathbf",
    r"\mathcal",
)

# Matches: \command { s p a c e d } — captures command name and brace content
_TEXT_CMD_PATTERN = re.compile(
    r"(\\(?:text|mathrm|operatorname|textbf|textit|mathit|mathbf|mathcal))"
    r"\s*\{([^}]*)\}",
)

# Detects single-letter-space sequences: "i f", "m e a n", "d a t a"
_SINGLE_LETTER_SPACE = re.compile(r"^(?:[a-zA-Z] )+[a-zA-Z]$")


def _fix_text_command_splitting(formula: str) -> tuple[str, list[str]]:
    """R1: Merge single-letter sequences inside text commands."""
    log: list[str] = []

    def replacer(m: re.Match) -> str:
        cmd, content = m.group(1), m.group(2)
        stripped = content.strip()
        if _SINGLE_LETTER_SPACE.match(stripped):
            merged = stripped.replace(" ", "")
            log.append(f"R1: {cmd}{{{stripped}}} -> {cmd}{{{merged}}}")
            return f"{cmd}{{{merged}}}"
        return m.group(0)

    fixed = _TEXT_CMD_PATTERN.sub(replacer, formula)
    return fixed, log


# ── R4: Abbreviation spacing ─────────────────────────────────────────────────

# Pattern: single letter, space, dot, space — repeated (i . e ., e . g .)
_ABBREV_SPACING = re.compile(r"([a-zA-Z]) \. ([a-zA-Z]) \.")


def _fix_abbreviation_spacing(formula: str) -> tuple[str, list[str]]:
    """R4: Fix spaced-out abbreviations inside text/mathrm commands."""
    log: list[str] = []

    def cmd_replacer(m: re.Match) -> str:
        cmd, content = m.group(1), m.group(2)
        if _ABBREV_SPACING.search(content):
            fixed = re.sub(r"([a-zA-Z]) \. ?", r"\1.", content)
            log.append(f"R4: {cmd}{{{content.strip()}}} -> {cmd}{{{fixed.strip()}}}")
            return f"{cmd}{{{fixed}}}"
        return m.group(0)

    fixed = _TEXT_CMD_PATTERN.sub(cmd_replacer, formula)
    return fixed, log


# ── R2: Bare letter splitting in math mode ───────────────────────────────────

_WORD_WHITELIST = [
    "subject to", "such that", "otherwise", "where", "given",
    "with", "over", "then", "for", "and", "if", "or",
]
# Sort longest-first for greedy matching
_WORD_WHITELIST.sort(key=len, reverse=True)

# Matches >=3 consecutive single-letter-space sequences: "s u c h t h a t"
_BARE_LETTERS = re.compile(r"(?<![a-zA-Z\\])([a-zA-Z] ){2,}[a-zA-Z](?![a-zA-Z{])")


def _fix_bare_letter_splitting(formula: str) -> tuple[str, list[str]]:
    """R2: Merge bare letter sequences that match whitelisted words."""
    log: list[str] = []

    def replacer(m: re.Match) -> str:
        raw = m.group(0)
        collapsed = raw.replace(" ", "")
        remaining = collapsed
        parts: list[str] = []

        while remaining:
            matched = False
            for word in _WORD_WHITELIST:
                no_space = word.replace(" ", "")
                if remaining.startswith(no_space):
                    parts.append(rf"\text{{{word}}}")
                    remaining = remaining[len(no_space):]
                    matched = True
                    break
            if not matched:
                # Leave unmatched trailing chars as-is
                if len(remaining) < 3:
                    parts.append(" ".join(remaining))
                    remaining = ""
                else:
                    return raw  # Can't match — leave unchanged

        result = " ".join(parts)
        log.append(f"R2: '{raw}' -> '{result}'")
        return result

    fixed = _BARE_LETTERS.sub(replacer, formula)
    return fixed, log


# ── R3: Brace balance repair ─────────────────────────────────────────────────


def _fix_brace_balance(formula: str) -> tuple[str, list[str]]:
    """R3: Fix brace mismatch when exactly ±1."""
    log: list[str] = []
    # Strip \{ and \} (LaTeX escaped braces) before counting —
    # they are literal brace glyphs, not structural grouping.
    stripped = re.sub(r"\\[{}]", "", formula)
    opens = stripped.count("{")
    closes = stripped.count("}")
    diff = opens - closes

    if diff == 0:
        return formula, log
    if abs(diff) >= 2:
        return formula, log  # too ambiguous

    if diff == -1:
        # Extra closing brace — remove the last }
        idx = formula.rfind("}")
        fixed = formula[:idx] + formula[idx + 1:]
        log.append(f"R3: removed extra '}}' at position {idx}")
        return fixed, log

    if diff == 1:
        # Missing closing brace — append at end
        fixed = formula + "}"
        log.append("R3: appended missing '}'")
        return fixed, log

    return formula, log


# ── R5: Pseudocode nested $ repair ───────────────────────────────────────────

_PSEUDOCODE_KEYWORDS = re.compile(
    r"\b(?:if|for|while|do|Sample|Input|Output|Return|repeat|until|end|Set|Compute|Initialize)\b",
    re.IGNORECASE,
)

# Matches $$<short_expr>$ — double-dollar open, single-dollar close
_NESTED_DOLLAR = re.compile(r"\$\$([a-zA-Z_\\][^$]{0,30}?)\$(?!\$)")


def _fix_pseudocode_delimiters(text: str) -> tuple[str, list[str]]:
    """R5: Fix confused $/$$ delimiters in pseudocode lines.

    Unlike R1-R4, this operates at the TEXT level (not within formula boundaries).
    Only triggers on lines containing pseudocode keywords.
    """
    log: list[str] = []
    lines = text.split("\n")
    fixed_lines = []

    for line in lines:
        if _PSEUDOCODE_KEYWORDS.search(line) and _NESTED_DOLLAR.search(line):
            fixed_line = _NESTED_DOLLAR.sub(r"$\1$", line)
            if fixed_line != line:
                log.append("R5: fixed nested $ in pseudocode line")
                line = fixed_line
        fixed_lines.append(line)

    return "\n".join(fixed_lines), log


# ── Formula boundary helpers ─────────────────────────────────────────────────

# Matches inline $...$ and display $$...$$, capturing the delimiters and content
_FORMULA_PATTERN = re.compile(
    r"(\$\$)(.*?)(\$\$)"   # display math (may span lines)
    r"|"
    r"(\$)((?!\$)[^\n$]*?)(\$(?!\$))",  # inline math (single line only, no nested $)
    re.DOTALL,  # DOTALL only affects display math branch; inline uses [^\n$]
)


def _apply_rules_to_formulas(
    text: str,
    rules: list,
) -> tuple[str, list[str]]:
    """Apply a list of rule functions to each formula in text.

    Each rule: (formula_content: str) -> (fixed: str, log: list[str])
    Operates only within $...$ and $$...$$ boundaries.
    """
    all_logs: list[str] = []

    def replacer(m: re.Match) -> str:
        if m.group(1):  # display math $$...$$
            open_delim, content, close_delim = m.group(1), m.group(2), m.group(3)
        else:  # inline math $...$
            open_delim, content, close_delim = m.group(4), m.group(5), m.group(6)

        for rule_fn in rules:
            content, logs = rule_fn(content)
            all_logs.extend(logs)

        return f"{open_delim}{content}{close_delim}"

    fixed = _FORMULA_PATTERN.sub(replacer, text)
    return fixed, all_logs


def fix_latex_formulas(text: str) -> tuple[str, list[str]]:
    """Apply all formula repair rules. Returns (fixed_text, list_of_changes)."""
    all_logs: list[str] = []

    # R5 first — operates at text level to fix delimiter structure
    text, r5_logs = _fix_pseudocode_delimiters(text)
    all_logs.extend(r5_logs)

    # R4, R1, R2, R3 — operate within formula boundaries
    text, rule_logs = _apply_rules_to_formulas(text, [
        _fix_abbreviation_spacing,  # R4
        _fix_text_command_splitting,  # R1
        _fix_bare_letter_splitting,  # R2
        _fix_brace_balance,  # R3 (last — after text fixes)
    ])
    all_logs.extend(rule_logs)

    return text, all_logs
