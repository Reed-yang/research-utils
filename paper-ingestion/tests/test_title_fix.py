"""Fix 2 verification: section-heading titles must not be mistaken for the
paper title (regression: page-1 OCR failure produced titles like "2").

No network / no GLM API spend — pure function checks.
"""
import sys
from pathlib import Path

# Add scripts/ to path so ingest_paper can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ingest_paper as ip


# Real paper titles that MUST survive (looks_like_section_heading -> False)
SAFE_TITLES = [
    "DreamBooth: Fine Tuning Text-to-Image Diffusion Models",
    "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
    "4D-Humans: Reconstructing and Tracking Humans",
    "2D or not 2D",
    "Introduction to Diffusion Models",
    "Background Subtraction via Deep Learning",
    "A Survey of Large Language Models",
    "A Neural Algorithm of Artistic Style",
    "A ConvNet for the 2020s",
]

# Body section headings / OCR garbage that MUST be rejected (-> True)
SECTION_HEADINGS = [
    "2. Related Work",
    "2. Video Diffusion Distillation",
    "23",
    "2",
    "3.1 Method",
    "3.1. Model Architecture",
    "5.1.1 Image Understanding",
    "Related Work",
    "References",
    "Conclusion",
    "A.1 Implementation Details",
]


class TestLooksLikeSectionHeading:
    def test_safe_titles_not_section(self):
        for t in SAFE_TITLES:
            assert ip.looks_like_section_heading(t) is False, t

    def test_defect_strings_are_section(self):
        for t in SECTION_HEADINGS:
            assert ip.looks_like_section_heading(t) is True, t


class TestExtractTitleFromMarkdown:
    def test_normal_title(self):
        md = "# DreamBooth: Fine Tuning\n\n## 1. Introduction\n\nbody"
        assert ip.extract_title_from_markdown(md) == "DreamBooth: Fine Tuning"

    def test_page1_failed_returns_none(self):
        md = (
            "\n\n<!-- PAGE 1 FAILED: timeout -->\n\n"
            "## 2. Related Work\n\n## 3. Method\n"
        )
        assert ip.extract_title_from_markdown(md) is None

    def test_safe_titles_roundtrip(self):
        for t in SAFE_TITLES:
            md = f"# {t}\n\n## 1. Introduction\n\nbody"
            assert ip.extract_title_from_markdown(md) == t, t

    def test_skips_leading_section_headings(self):
        md = "## 2. Related Work\n\n# Actual Paper Title\n\nbody"
        assert ip.extract_title_from_markdown(md) == "Actual Paper Title"


class TestLooksLikePlaceholderTitle:
    def test_section_heading_is_placeholder(self):
        assert ip.looks_like_placeholder_title("2. Related Work") is True

    def test_real_title_not_placeholder(self):
        assert ip.looks_like_placeholder_title("DreamBooth: X") is False

    def test_arxiv_id_still_placeholder(self):
        assert ip.looks_like_placeholder_title("2401.12345") is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
