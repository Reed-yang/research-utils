import sys
from pathlib import Path

# Add scripts/ to path so fix_formulas can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fix_formulas import fix_latex_formulas


class TestR1TextCommandWordSplitting:
    """R1: Merge single-letter sequences in \\text{}, \\mathrm{}, \\operatorname{}, etc."""

    def test_text_if(self):
        result, log = fix_latex_formulas(r"$\text{i f}$")
        assert result == r"$\text{if}$"
        assert len(log) == 1

    def test_operatorname_mean(self):
        result, log = fix_latex_formulas(r"$\operatorname{m e a n}$")
        assert result == r"$\operatorname{mean}$"

    def test_mathrm_data(self):
        result, log = fix_latex_formulas(r"$\mathrm{d a t a}$")
        assert result == r"$\mathrm{data}$"

    def test_mathrm_KL_no_change(self):
        """KL is only 2 chars — should NOT be merged (not single-letter-space pattern)."""
        result, _ = fix_latex_formulas(r"$D_{\mathrm{KL}}$")
        assert result == r"$D_{\mathrm{KL}}$"

    def test_text_normal_word_no_change(self):
        """Already correct text should not be modified."""
        result, _ = fix_latex_formulas(r"$\text{otherwise}$")
        assert result == r"$\text{otherwise}$"

    def test_outside_formula_untouched(self):
        """Plain text outside $ must never be modified."""
        input_text = "This is plain text with s p a c e d letters"
        result, _ = fix_latex_formulas(input_text)
        assert result == input_text

    def test_display_math(self):
        result, _ = fix_latex_formulas(r"$$\operatorname{c l i p}(x)$$")
        assert result == r"$$\operatorname{clip}(x)$$"

    def test_multiple_commands_one_formula(self):
        result, _ = fix_latex_formulas(
            r"$\text{i f} x > 0 \text{o t h e r w i s e}$"
        )
        assert result == r"$\text{if} x > 0 \text{otherwise}$"


class TestR4AbbreviationSpacing:
    """R4: Fix spaced abbreviations like i . e . inside text commands."""

    def test_ie_in_mathrm(self):
        result, _ = fix_latex_formulas(r"$\mathrm{i . e .}$")
        assert result == r"$\mathrm{i.e.}$"

    def test_eg_in_text(self):
        result, _ = fix_latex_formulas(r"$\text{e . g .}$")
        assert result == r"$\text{e.g.}$"

    def test_no_change_normal_dots(self):
        result, _ = fix_latex_formulas(r"$\text{etc.}$")
        assert result == r"$\text{etc.}$"


class TestR3BraceBalance:
    """R3: Fix brace mismatch when difference is exactly ±1."""

    def test_extra_closing_brace(self):
        result, _ = fix_latex_formulas(r"$x_{0}^{<i}}$")
        assert result == r"$x_{0}^{<i}$"

    def test_missing_closing_brace(self):
        result, _ = fix_latex_formulas(r"$\frac{a{b}$")
        assert result == r"$\frac{a{b}}$"

    def test_balanced_no_change(self):
        result, _ = fix_latex_formulas(r"$\frac{a}{b}$")
        assert result == r"$\frac{a}{b}$"

    def test_mismatch_ge2_skip(self):
        """Mismatch >= 2 should be skipped (too ambiguous)."""
        original = r"$\frac{a{b$"
        result, _ = fix_latex_formulas(original)
        assert result == original  # unchanged

    def test_extra_brace_in_complex_formula(self):
        result, _ = fix_latex_formulas(
            r"$p_{\theta}\left(\boldsymbol{x}_{0}^{<i}}\right)$"
        )
        assert r"}}" not in result  # extra brace removed

    def test_escaped_braces_ignored(self):
        r"""R3 must not count \{ and \} as structural braces."""
        original = r"$x^{[N]\backslash\{i]}$"
        result, _ = fix_latex_formulas(original)
        assert result == original  # already balanced, no change


class TestR2BareLetterSplitting:
    """R2: Merge >=3 consecutive single-letter-space sequences into \\text{}."""

    def test_such_that(self):
        result, _ = fix_latex_formulas(r"$x > 0, s u c h t h a t y < 1$")
        assert r"\text{such that}" in result

    def test_and(self):
        result, _ = fix_latex_formulas(r"$x = 1 a n d y = 2$")
        assert r"\text{and}" in result

    def test_otherwise(self):
        result, _ = fix_latex_formulas(r"$0, o t h e r w i s e$")
        assert r"\text{otherwise}" in result

    def test_two_letters_no_change(self):
        """Only 2 single letters — should NOT trigger (could be math variables)."""
        original = r"$a b$"
        result, _ = fix_latex_formulas(original)
        assert result == original

    def test_math_variables_untouched(self):
        """Single variables like 'x + y' should not be merged."""
        original = r"$x + y$"
        result, _ = fix_latex_formulas(original)
        assert result == original


class TestR5PseudocodeDelimiterRepair:
    """R5: Fix confused $/$$ delimiters in pseudocode lines."""

    def test_double_dollar_variable(self):
        result, _ = fix_latex_formulas("Sample $$x_0$ \\sim \\mathcal{N}(0, I)")
        assert result == "Sample $x_0$ \\sim \\mathcal{N}(0, I)"

    def test_if_line(self):
        result, _ = fix_latex_formulas(
            "if $$t_i$ \\in [S_{t_{\\min}}, S_{t_{\\max}}]$"
        )
        assert "$$t_i$" not in result
        assert "$t_i$" in result

    def test_display_math_untouched(self):
        """Legitimate display math should NOT be modified."""
        original = "$$E = mc^2$$"
        result, _ = fix_latex_formulas(original)
        assert result == original

    def test_no_keyword_no_change(self):
        """Lines without pseudocode keywords should not trigger R5."""
        original = "$$x_0$ is defined"
        result, _ = fix_latex_formulas(original)
        assert result == original

    def test_for_loop_line(self):
        result, _ = fix_latex_formulas("For $i \\in \\{0, \\dots, N-1\\}$ do")
        # Already correct — should not change
        assert result == "For $i \\in \\{0, \\dots, N-1\\}$ do"
