"""Fix 1 verification: when URL mode exhausts retries (incl. 429 rate-limit),
convert_with_glm_ocr must fall back to per-page image mode instead of
output_error()+sys.exit().

No network / no GLM API spend — requests.post and the per-page helper are
stubbed; time.sleep is neutralized so backoff is instant.
"""
import sys
import time as _time
from pathlib import Path

import pytest
import requests as _requests

# Add scripts/ to path so ingest_paper can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ingest_paper as ip


class _StopFallback(Exception):
    """Raised by the stubbed per-page helper to halt before the downstream
    bbox/file pipeline — we only need to prove the fallback path ran."""


class _Resp429:
    status_code = 429
    text = "rate limited"

    def json(self):
        return {}


def test_url_mode_falls_back_to_page_image(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 dummy small file")  # tiny -> URL mode route

    monkeypatch.setenv("GLM_API_KEY", "test-key")
    monkeypatch.setenv("GLM_API_ID", "test-id")

    # Avoid needing a real/parseable PDF
    monkeypatch.setattr(ip, "get_pdf_page_count", lambda p: 3)

    # Force every URL attempt to 429; neutralize backoff sleep (instant, no spend)
    monkeypatch.setattr(_requests, "post", lambda *a, **k: _Resp429())
    monkeypatch.setattr(_time, "sleep", lambda *a, **k: None)

    called = {"v": False}

    def _recorder(pdf_path, auth_token, page_count, debug):
        called["v"] = True
        raise _StopFallback()

    monkeypatch.setattr(ip, "_run_glm_page_image_mode", _recorder)

    with pytest.raises(_StopFallback):
        ip.convert_with_glm_ocr(
            pdf,
            tmp_path / "assets",
            "webp",
            90,
            False,
            debug=False,
            source_url="https://arxiv.org/pdf/2401.00001",
        )

    # Fallback helper was invoked, and the mode was relabeled before the call
    assert called["v"] is True
    assert ip._conversion_metadata["mode"] == "page-images (url-fallback)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
