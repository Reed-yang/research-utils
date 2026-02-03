#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "docling>=2.5.0",
#     "torch>=2.0.0",
#     "pypdf>=4.0.0",
#     "pillow>=10.0.0",
#     "requests>=2.31.0",
# ]
# [tool.uv]
# exclude-newer = "2025-12-01"
# ///
"""
Paper Ingestion Tool - Convert PDF papers to Markdown for AI-native research workflow.

Dual-backend strategy:
  - docling (default): Fast, CPU/GPU, layout-aware, great for tables, extracts images
  - nougat: Slow, GPU-intensive, end-to-end Transformer, perfect for heavy LaTeX math

Usage:
  uv run ingest_paper.py <pdf_path_or_url> [--engine docling|nougat] [--output-dir <path>] [--images-scale <float>]
    [--image-format source|png|jpg|jpeg|webp] [--image-quality <1-100>] [--image-lossless] [--force]

Output:
  - Files organized in {cwd}/{YYYYMMDD}-{Sanitized_Title}/
  - Images saved to assets/ subfolder with relative paths in markdown
  - Single JSON object to stdout with status, paths, and metadata.
"""

import argparse
import csv
import io
import math
import os
import subprocess
import json
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote


# ============================================================================
# Configuration
# ============================================================================


def get_output_root(output_dir: str | None = None) -> Path:
    """Get output root directory. Defaults to current working directory."""
    if output_dir:
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root
    return Path.cwd()


# ============================================================================
# Utility Functions
# ============================================================================


def sanitize_filename(name: str) -> str:
    r"""
    Convert filename to a Windows-safe folder name.
    Removes invalid chars: : ? / \ * < > | "
    Replaces spaces with underscores.
    """
    # Remove extension
    stem = Path(name).stem
    # Remove Windows-invalid characters: : ? / \ * < > | "
    sanitized = re.sub(r'[:\?/\\*<>|"]', "", stem)
    # Replace spaces and multiple hyphens with underscores
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    # Remove any other non-word characters except underscores
    sanitized = re.sub(r"[^\w_]", "", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


def extract_title_from_markdown(markdown_content: str) -> str | None:
    """Extract title from markdown heading."""
    title_match = re.search(r"^\s*#{1,3}\s+(.+?)\s*$", markdown_content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()
    return None


def extract_pdf_metadata_title(pdf_path: Path) -> str | None:
    """Extract title from PDF metadata if available."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        if reader.metadata and reader.metadata.title:
            return str(reader.metadata.title).strip()
    except Exception:
        return None
    return None


def resolve_paper_title(
    detected_title: str | None, markdown_content: str, pdf_path: Path
) -> str | None:
    """Resolve best available title for folder naming."""
    markdown_title = extract_title_from_markdown(markdown_content)
    if markdown_title and (
        not detected_title or looks_like_placeholder_title(detected_title)
    ):
        return markdown_title
    if detected_title:
        return detected_title
    metadata_title = extract_pdf_metadata_title(pdf_path)
    if metadata_title:
        return metadata_title
    return None


def looks_like_placeholder_title(title: str) -> bool:
    """Check if title is likely a filename or arXiv id."""
    normalized = title.strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if lowered.endswith(".pdf"):
        lowered = lowered[:-4]
    if "http" in lowered or "/" in lowered or "\\" in lowered:
        return True
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", lowered):
        return True
    if re.fullmatch(r"[0-9._-]+", lowered):
        return True
    return False


def apply_outside_code_blocks(text: str, transform) -> str:
    """Apply a transform to text outside fenced code blocks."""
    out = []
    buffer = []
    in_code_block = False
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            if not in_code_block:
                if buffer:
                    out.append(transform("".join(buffer)))
                    buffer = []
                in_code_block = True
            else:
                if buffer:
                    out.append("".join(buffer))
                    buffer = []
                in_code_block = False
            out.append(line)
        else:
            buffer.append(line)
    if buffer:
        if in_code_block:
            out.append("".join(buffer))
        else:
            out.append(transform("".join(buffer)))
    return "".join(out)


def normalize_math_delimiters(text: str) -> str:
    """Normalize math delimiters to $...$ and $$...$$ for Markdown."""
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    env_pattern = re.compile(
        r"\\begin\{(equation\*?|align\*?|multline\*?|gather\*?|split|cases)\}"
        r"(.*?)\\end\{\1\}",
        re.DOTALL,
    )

    def wrap_env(match: re.Match) -> str:
        start, end = match.span()
        before = text[max(0, start - 2) : start]
        after = text[end : end + 2]
        if before == "$$" and after == "$$":
            return match.group(0)
        return f"$$\n{match.group(0)}\n$$"

    return env_pattern.sub(wrap_env, text)


def replace_image_placeholders(
    markdown_content: str, image_count: int, image_ext: str = ".png"
) -> str:
    """Replace <!-- image --> placeholders with actual image references."""
    counter = [0]  # Use list to allow mutation in nested function

    def replacer(match):
        counter[0] += 1
        if counter[0] <= image_count:
            return f"![Figure {counter[0]}](./assets/image_{counter[0]:03d}{image_ext})"
        return match.group(0)

    return re.sub(r"<!--\s*image\s*-->", replacer, markdown_content)


def normalize_image_format(image_format: str | None) -> str:
    """Normalize image format selection."""
    if not image_format:
        return "source"
    normalized = image_format.strip().lower()
    if normalized in ("source", "auto", "original", "keep"):
        return "source"
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in ("png", "jpeg", "webp"):
        raise ValueError(f"Unsupported image format: {image_format}")
    return normalized


def clamp_image_quality(quality: int) -> int:
    """Clamp image quality to 1-100."""
    return max(1, min(100, int(quality)))


def get_image_extension(
    image_format: str, source_ext: str | None = None, default_for_source: str = ".png"
) -> str:
    """Resolve output image extension."""
    if image_format == "source":
        if source_ext:
            source_ext = source_ext.lower()
            return source_ext if source_ext.startswith(".") else f".{source_ext}"
        return default_for_source
    if image_format == "jpeg":
        return ".jpg"
    return f".{image_format}"


def save_pil_image(
    pil_image,
    output_path: Path,
    image_format: str,
    image_quality: int,
    image_lossless: bool,
) -> None:
    """Save PIL image to target format."""
    fmt = "JPEG" if image_format == "jpeg" else image_format.upper()
    save_kwargs = {}

    if image_format == "jpeg":
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        save_kwargs.update(
            {"quality": image_quality, "optimize": True, "progressive": True}
        )
    elif image_format == "webp":
        save_kwargs.update({"method": 6, "lossless": image_lossless})
        save_kwargs["quality"] = 100 if image_lossless else image_quality
    elif image_format == "png":
        save_kwargs.update({"optimize": True})

    pil_image.save(str(output_path), format=fmt, **save_kwargs)


def save_image_bytes(
    img_bytes: bytes,
    output_path: Path,
    image_format: str,
    image_quality: int,
    image_lossless: bool,
) -> None:
    """Save image bytes in desired format."""
    if image_format == "source":
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        return
    from PIL import Image

    with Image.open(io.BytesIO(img_bytes)) as img:
        save_pil_image(img, output_path, image_format, image_quality, image_lossless)


def reencode_image_file(
    src_path: Path,
    dest_path: Path,
    image_format: str,
    image_quality: int,
    image_lossless: bool,
) -> None:
    """Re-encode an on-disk image to the target format."""
    if image_format == "source":
        shutil.copy2(src_path, dest_path)
        return
    from PIL import Image

    with Image.open(src_path) as img:
        save_pil_image(img, dest_path, image_format, image_quality, image_lossless)


def wrap_inline_math(text: str) -> str:
    """
    Wrap common inline math patterns in $...$ delimiters.
    Applied outside code blocks only.
    """

    def transform(segment: str) -> str:
        # Variables with subscripts: x_i, P_ref, z_0, etc.
        segment = re.sub(r"\b([A-Za-z])\s*([_])\s*(\w+)\b", r"$\1_\3$", segment)

        # Greek letters as standalone
        greek = r"[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΘΛΞΠΣΦΨΩ∈∀∃∅∇∞∝∑∏∫∂]"
        segment = re.sub(rf"(?<![\$A-Za-z])({greek})(?![A-Za-z])", r"$\1$", segment)

        # Comparison operators with numbers: 1 ≤ j ≤ N
        segment = re.sub(r"(\d+)\s*([≤≥≈≠<>])\s*(\w+)", r"$\1 \2 \3$", segment)
        segment = re.sub(r"(\w+)\s*([≤≥≈≠])\s*(\d+)", r"$\1 \2 \3$", segment)

        return segment

    return apply_outside_code_blocks(text, transform)


def output_json(data: dict) -> None:
    """Print JSON to stdout (agent interface)."""
    print(json.dumps(data, ensure_ascii=False))


def output_error(message: str, suggestion: str = None) -> None:
    """Output error JSON and exit."""
    error_data = {"status": "error", "message": message}
    if suggestion:
        error_data["suggestion"] = suggestion
    output_json(error_data)
    sys.exit(1)


def is_url(path: str) -> bool:
    """Check if the input is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def download_pdf(url: str) -> Path:
    """Download PDF from URL to a temporary file."""
    import requests

    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        # Try to get filename from Content-Disposition or URL
        filename = None
        if "Content-Disposition" in response.headers:
            cd = response.headers["Content-Disposition"]
            if "filename=" in cd:
                filename = cd.split("filename=")[-1].strip("\"'")

        if not filename:
            # Extract from URL path
            url_path = urlparse(url).path
            filename = unquote(Path(url_path).name)

        if not filename or not filename.lower().endswith(".pdf"):
            filename = "downloaded_paper.pdf"

        # Save to temp file
        temp_dir = Path(tempfile.mkdtemp())
        temp_path = temp_dir / filename

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return temp_path

    except requests.RequestException as e:
        output_error(f"Failed to download PDF: {e}")


def check_duplicate(sanitized_title: str, output_root: Path) -> bool:
    """
    Check if a folder with the same title already exists (ignoring date prefix).
    Returns True if duplicate found.
    """
    # Pattern: {YYYYMMDD}-{title} or just {title}
    for item in output_root.iterdir():
        if item.is_dir():
            dir_name = item.name
            # Remove date prefix if present (format: YYYYMMDD-)
            if re.match(r"^\d{8}-", dir_name):
                existing_title = dir_name[9:]  # Skip "YYYYMMDD-"
            else:
                existing_title = dir_name

            if existing_title == sanitized_title:
                return True
    return False


# ============================================================================
# Docling Backend (with Image Extraction)
# ============================================================================


def get_all_free_gpus(
    max_memory_mb: int = 2000, max_util_percent: int = 10
) -> list[str]:
    """
    Get indices of all free GPUs, sorted by memory usage (lowest first).

    Priority:
    1. If CUDA_VISIBLE_DEVICES is set, use those GPU indices directly
    2. Otherwise, auto-detect free GPUs via nvidia-smi

    Returns empty list if no GPU is free or nvidia-smi fails.
    """
    # Priority 1: Respect CUDA_VISIBLE_DEVICES if already set by user/container
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip():
        # User explicitly set GPU indices - use them as-is
        # These are already the logical indices visible to this process
        indices = [idx.strip() for idx in cuda_visible.split(",") if idx.strip()]
        print(f"Using CUDA_VISIBLE_DEVICES: {indices}", file=sys.stderr)
        return indices

    # Priority 2: Auto-detect free GPUs via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []

        reader = csv.reader(io.StringIO(result.stdout.strip()))
        free_gpus = []
        for row in reader:
            try:
                idx, mem, util = map(int, row)
                if mem < max_memory_mb and util < max_util_percent:
                    free_gpus.append((idx, mem))
            except ValueError:
                continue

        # Sort by memory usage (ascending)
        free_gpus.sort(key=lambda x: x[1])
        return [str(gpu[0]) for gpu in free_gpus]

    except FileNotFoundError:
        return []  # nvidia-smi not found
    except Exception:
        return []


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get total page count of a PDF using pypdf."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return 0


def split_pdf_to_chunks(
    pdf_path: Path, num_chunks: int, tmpdir: Path
) -> list[tuple[Path, int, int]]:
    """
    Split PDF into chunks for parallel processing.

    Returns list of (chunk_pdf_path, start_page, end_page) tuples.
    Pages are 0-indexed.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    if total_pages == 0:
        return []

    # Limit chunks to page count
    num_chunks = min(num_chunks, total_pages)
    pages_per_chunk = math.ceil(total_pages / num_chunks)

    chunks = []
    for i in range(num_chunks):
        start_page = i * pages_per_chunk
        end_page = min((i + 1) * pages_per_chunk - 1, total_pages - 1)

        if start_page > end_page:
            break

        # Create chunk PDF
        writer = PdfWriter()
        for page_num in range(start_page, end_page + 1):
            writer.add_page(reader.pages[page_num])

        chunk_path = tmpdir / f"chunk_{i:02d}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)

        chunks.append((chunk_path, start_page, end_page))

    return chunks


def run_mineru_on_chunk(
    chunk_pdf: Path,
    gpu_idx: str,
    output_dir: Path,
    chunk_idx: int,
) -> tuple[int, str, Path | None]:
    """
    Run MinerU on a single PDF chunk using specified GPU.

    Returns (chunk_idx, markdown_content, images_dir or None).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["MINERU_HYBRID_BATCH_RATIO"] = "8"

    try:
        result = subprocess.run(
            [
                "mineru",
                "-p",
                str(chunk_pdf),
                "-o",
                str(output_dir),
                "-b",
                "hybrid-auto-engine",
                "-l",
                "en",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per chunk
        )

        if result.returncode != 0:
            return (chunk_idx, "", None)

        # Find generated markdown
        md_files = list(output_dir.rglob("*.md"))
        if not md_files:
            return (chunk_idx, "", None)

        md_file = md_files[0]
        markdown_content = md_file.read_text(encoding="utf-8")

        # Find images directory
        images_dir = md_file.parent / "images"
        if not images_dir.exists():
            images_dir = None

        return (chunk_idx, markdown_content, images_dir)

    except Exception:
        return (chunk_idx, "", None)


def merge_chunk_results(
    chunk_results: list[tuple[int, str, Path | None]],
    assets_dir: Path,
    image_format: str,
    image_quality: int,
    image_lossless: bool,
) -> tuple[str, int]:
    """
    Merge markdown from multiple chunks and renumber images sequentially.

    Returns (merged_markdown, total_image_count).
    """
    # Sort by chunk index
    chunk_results.sort(key=lambda x: x[0])

    image_format = normalize_image_format(image_format)
    image_quality = clamp_image_quality(image_quality)

    assets_dir.mkdir(parents=True, exist_ok=True)
    merged_parts = []
    global_image_counter = 0

    for chunk_idx, markdown_content, images_dir in chunk_results:
        if not markdown_content:
            continue

        # Build image mapping for this chunk
        image_map = {}  # old_ref -> new_ref

        if images_dir and images_dir.exists():
            for img_file in sorted(images_dir.iterdir()):
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    global_image_counter += 1
                    source_ext = img_file.suffix.lower()
                    target_ext = get_image_extension(
                        image_format, source_ext, default_for_source=".png"
                    )
                    new_name = f"image_{global_image_counter:03d}{target_ext}"
                    dest_path = assets_dir / new_name
                    reencode_image_file(
                        img_file,
                        dest_path,
                        image_format,
                        image_quality,
                        image_lossless,
                    )

                    # Map various possible references
                    old_ref = f"images/{img_file.name}"
                    image_map[old_ref] = f"./assets/{new_name}"
                    image_map[img_file.name] = f"./assets/{new_name}"

        # Rewrite image paths in this chunk's markdown
        def replace_img_path(match):
            alt_text = match.group(1)
            old_path = match.group(2)

            if old_path in image_map:
                return f"![{alt_text}]({image_map[old_path]})"

            # Try filename only
            filename = Path(old_path).name
            if filename in image_map:
                return f"![{alt_text}]({image_map[filename]})"

            return match.group(0)

        updated_content = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            replace_img_path,
            markdown_content,
        )

        merged_parts.append(updated_content)

    # Join with page breaks
    merged_markdown = "\n\n---\n\n".join(merged_parts)

    return merged_markdown, global_image_counter


def convert_with_docling(
    pdf_path: Path,
    assets_dir: Path,
    images_scale: float,
    image_format: str,
    image_quality: int,
    image_lossless: bool,
) -> tuple[str, str | None]:
    """
    Convert PDF to Markdown using IBM Docling with image extraction.
    Returns (markdown_content, detected_title).
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        image_format = normalize_image_format(image_format)
        image_quality = clamp_image_quality(image_quality)
        output_format = "png" if image_format == "source" else image_format
        image_ext = get_image_extension(output_format)

        # Configure pipeline with image extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.do_formula_enrichment = True
        # Higher value => higher resolution (1.0 ~= 72 DPI)
        pipeline_options.images_scale = images_scale

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(str(pdf_path))
        doc = result.document

        # Extract title from document metadata
        detected_title = None
        if hasattr(doc, "name") and doc.name:
            detected_title = doc.name

        # Export markdown
        markdown_content = doc.export_to_markdown(image_mode="referenced")

        # Save images to assets directory and rewrite paths
        assets_dir.mkdir(parents=True, exist_ok=True)
        image_counter = 0

        if hasattr(doc, "pictures") and doc.pictures:
            for pic in doc.pictures:
                if hasattr(pic, "image") and pic.image:
                    image_counter += 1
                    image_name = f"image_{image_counter:03d}{image_ext}"
                    image_path = assets_dir / image_name
                    save_pil_image(
                        pic.image.pil_image,
                        image_path,
                        output_format,
                        image_quality,
                        image_lossless,
                    )

        # Replace <!-- image --> placeholders with actual image references
        markdown_content = replace_image_placeholders(
            markdown_content, image_counter, image_ext
        )

        return markdown_content, detected_title

    except ImportError as e:
        output_error(
            f"Docling import failed: {e}",
            "Ensure docling is installed: uv pip install docling",
        )
    except Exception as e:
        output_error(f"Docling conversion failed: {e}")


# ============================================================================
# MinerU Backend (High-Quality GPU-Accelerated)
# ============================================================================

# Default mineru-api server configuration
MINERU_API_HOST = os.environ.get("MINERU_API_HOST", "127.0.0.1")
MINERU_API_PORT = int(os.environ.get("MINERU_API_PORT", "8000"))


def check_mineru_api_server(host: str = None, port: int = None) -> bool:
    """
    Check if mineru-api server is running and healthy.

    Returns True if server is available.
    """
    import requests

    host = host or MINERU_API_HOST
    port = port or MINERU_API_PORT

    try:
        response = requests.get(f"http://{host}:{port}/docs", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def convert_via_mineru_api(
    pdf_path: Path,
    assets_dir: Path,
    host: str = None,
    port: int = None,
    image_format: str = "source",
    image_quality: int = 95,
    image_lossless: bool = False,
) -> tuple[str, str | None]:
    """
    Convert PDF via mineru-api REST server (persistent model, no reload overhead).

    Requires mineru-api server to be running:
        CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000

    Returns (markdown_content, detected_title).
    """
    import requests

    host = host or MINERU_API_HOST
    port = port or MINERU_API_PORT
    image_format = normalize_image_format(image_format)
    image_quality = clamp_image_quality(image_quality)

    try:
        # Upload PDF and convert via API
        # Endpoint is /file_parse per MinerU's fast_api.py
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"http://{host}:{port}/file_parse",
                files={"files": (pdf_path.name, f, "application/pdf")},
                data={
                    "backend": "hybrid-auto-engine",
                    "parse_method": "auto",
                    "lang_list": "en",
                    "return_md": "true",
                    "return_images": "true",
                    "formula_enable": "true",
                    "table_enable": "true",
                },
                timeout=600,  # 10 min timeout for large PDFs
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"API error: {response.status_code} - {response.text[:200]}"
            )

        result = response.json()

        # API response format:
        # {"backend": "...", "version": "...", "results": {
        #   "pdf_stem": {"md_content": "...", "images": {"img.jpg": "data:image/jpeg;base64,..."}}
        # }}
        results = result.get("results", {})
        if not results:
            raise RuntimeError("API returned empty results")

        # Get first (only) result since we send one file
        pdf_stem = pdf_path.stem
        pdf_result = results.get(pdf_stem, {})
        if not pdf_result:
            # Try to get first available result
            pdf_result = next(iter(results.values()), {})

        # Extract markdown content
        markdown_content = pdf_result.get("md_content", "")
        if not markdown_content:
            raise RuntimeError("API returned no markdown content")

        # Handle images from API response
        assets_dir.mkdir(parents=True, exist_ok=True)
        image_counter = 0
        image_map = {}

        # API returns images as base64 data URIs: {"img.jpg": "data:image/jpeg;base64,..."}
        images_dict = pdf_result.get("images", {})
        for img_filename, img_data_uri in images_dict.items():
            image_counter += 1
            # Determine extension from filename or default to jpg
            source_ext = Path(img_filename).suffix or ".jpg"
            target_ext = get_image_extension(
                image_format, source_ext, default_for_source=".jpg"
            )
            img_name = f"image_{image_counter:03d}{target_ext}"
            img_path = assets_dir / img_name

            # Parse data URI: data:image/jpeg;base64,/9j/4AAQ...
            if isinstance(img_data_uri, str) and ";base64," in img_data_uri:
                import base64

                base64_data = img_data_uri.split(";base64,", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                save_image_bytes(
                    img_bytes,
                    img_path,
                    image_format,
                    image_quality,
                    image_lossless,
                )
                # Map original reference to new path
                image_map[f"images/{img_filename}"] = f"./assets/{img_name}"
                image_map[img_filename] = f"./assets/{img_name}"

        # Rewrite image paths in markdown
        def replace_img_path(match):
            alt_text = match.group(1)
            old_path = match.group(2)
            if old_path in image_map:
                return f"![{alt_text}]({image_map[old_path]})"
            for old_ref, new_ref in image_map.items():
                if old_ref in old_path or old_path in old_ref:
                    return f"![{alt_text}]({new_ref})"
            return match.group(0)

        markdown_content = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            replace_img_path,
            markdown_content,
        )

        detected_title = extract_title_from_markdown(markdown_content)
        return markdown_content, detected_title

    except Exception as e:
        raise RuntimeError(f"mineru-api conversion failed: {e}")


# Global to track conversion metadata for logging
_conversion_metadata = {}


def get_conversion_metadata() -> dict:
    """Get metadata about the last conversion for logging."""
    return _conversion_metadata.copy()


def convert_with_mineru(
    pdf_path: Path,
    assets_dir: Path,
    image_format: str,
    image_quality: int,
    image_lossless: bool,
) -> tuple[str, str | None]:
    """
    Convert PDF to Markdown using MinerU (hybrid-auto-engine for best accuracy).

    Strategy:
    1. Try mineru-api server first (persistent model, no reload overhead)
    2. Fall back to subprocess CLI if server not available

    Automatically detects all free GPUs and uses parallel page-chunking
    when multiple GPUs are available.

    Returns (markdown_content, detected_title).
    """
    global _conversion_metadata
    image_format = normalize_image_format(image_format)
    image_quality = clamp_image_quality(image_quality)
    page_count = get_pdf_page_count(pdf_path)
    # === PRIORITY 1: Try mineru-api server (persistent model) ===
    if check_mineru_api_server():
        print(
            f"MinerU: Using API server at {MINERU_API_HOST}:{MINERU_API_PORT} (persistent model)",
            file=sys.stderr,
        )
        try:
            _conversion_metadata = {
                "backend": "mineru",
                "mode": "api",
                "api_host": MINERU_API_HOST,
                "api_port": MINERU_API_PORT,
                "page_count": page_count,
                "image_format": image_format,
                "image_quality": image_quality,
                "image_lossless": image_lossless,
            }
            return convert_via_mineru_api(
                pdf_path,
                assets_dir,
                image_format=image_format,
                image_quality=image_quality,
                image_lossless=image_lossless,
            )
        except Exception as e:
            print(f"MinerU API failed: {e}, falling back to CLI", file=sys.stderr)

    # === PRIORITY 2: Fall back to subprocess CLI ===
    MAX_WORKERS = 4  # Cap parallel workers to avoid OOM

    # Detect all free GPUs
    free_gpus = get_all_free_gpus()
    page_count = get_pdf_page_count(pdf_path)

    if not free_gpus:
        print("MinerU: No free GPU found, using default/CPU", file=sys.stderr)
        free_gpus = [""]  # Empty string = use default CUDA device

    # Determine optimal worker count
    num_workers = min(len(free_gpus), page_count, MAX_WORKERS)
    num_workers = max(1, num_workers)  # At least 1 worker

    # Use selected GPUs (take first N)
    selected_gpus = free_gpus[:num_workers]

    # Report configuration
    batch_ratio = 8
    if num_workers > 1:
        print(
            f"MinerU using {num_workers} GPUs: {selected_gpus} "
            f"({num_workers} workers, batch_ratio={batch_ratio})",
            file=sys.stderr,
        )
    else:
        gpu_str = selected_gpus[0] if selected_gpus[0] else "default"
        print(
            f"MinerU using GPU: {gpu_str} (batch_ratio={batch_ratio})", file=sys.stderr
        )

    # Store metadata for logging
    _conversion_metadata = {
        "backend": "mineru",
        "mode": "cli",
        "gpus": ",".join(selected_gpus) if selected_gpus[0] else "default",
        "num_workers": num_workers,
        "batch_ratio": batch_ratio,
        "page_count": page_count,
        "image_format": image_format,
        "image_quality": image_quality,
        "image_lossless": image_lossless,
    }

    # Create temporary directory
    tmpdir = Path(tempfile.mkdtemp())

    try:
        # === SINGLE WORKER PATH ===
        if num_workers == 1:
            env = os.environ.copy()
            env["MINERU_HYBRID_BATCH_RATIO"] = "8"
            if selected_gpus[0]:
                env["CUDA_VISIBLE_DEVICES"] = selected_gpus[0]

            result = subprocess.run(
                [
                    "mineru",
                    "-p",
                    str(pdf_path),
                    "-o",
                    str(tmpdir),
                    "-b",
                    "hybrid-auto-engine",
                    "-l",
                    "en",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                output_error(
                    f"MinerU conversion failed: {result.stderr[:500]}",
                    "Check MinerU installation: uv pip install -U 'mineru[all]'",
                )

            # Find generated markdown
            md_files = list(tmpdir.rglob("*.md"))
            if not md_files:
                output_error("MinerU produced no markdown output")

            pdf_stem = pdf_path.stem
            md_file = next((f for f in md_files if f.stem == pdf_stem), md_files[0])
            markdown_content = md_file.read_text(encoding="utf-8")

            # Copy images to assets directory
            assets_dir.mkdir(parents=True, exist_ok=True)
            image_map = {}
            image_counter = 0

            images_dir = md_file.parent / "images"
            if images_dir.exists():
                for img_file in sorted(images_dir.iterdir()):
                    if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        image_counter += 1
                        source_ext = img_file.suffix.lower()
                        target_ext = get_image_extension(
                            image_format, source_ext, default_for_source=".png"
                        )
                        new_name = f"image_{image_counter:03d}{target_ext}"
                        dest_path = assets_dir / new_name
                        reencode_image_file(
                            img_file,
                            dest_path,
                            image_format,
                            image_quality,
                            image_lossless,
                        )
                        old_ref = f"images/{img_file.name}"
                        image_map[old_ref] = f"./assets/{new_name}"

            # Rewrite image paths
            def replace_img_path(match):
                alt_text = match.group(1)
                old_path = match.group(2)
                if old_path in image_map:
                    return f"![{alt_text}]({image_map[old_path]})"
                for old_ref, new_ref in image_map.items():
                    if old_ref.endswith(Path(old_path).name):
                        return f"![{alt_text}]({new_ref})"
                return match.group(0)

            markdown_content = re.sub(
                r"!\[([^\]]*)\]\(([^)]+)\)",
                replace_img_path,
                markdown_content,
            )

            detected_title = extract_title_from_markdown(markdown_content)
            return markdown_content, detected_title

        # === MULTI-WORKER PARALLEL PATH ===
        chunks_dir = tmpdir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Split PDF into chunks
        chunks = split_pdf_to_chunks(pdf_path, num_workers, chunks_dir)

        if not chunks:
            output_error("Failed to split PDF into chunks")

        print(
            f"  Split into {len(chunks)} chunks for parallel processing",
            file=sys.stderr,
        )

        # Process chunks in parallel
        chunk_results = []

        def process_chunk(args):
            chunk_idx, chunk_info, gpu_idx = args
            chunk_pdf, start_pg, end_pg = chunk_info
            output_dir = tmpdir / f"output_{chunk_idx:02d}"
            output_dir.mkdir(exist_ok=True)
            return run_mineru_on_chunk(chunk_pdf, gpu_idx, output_dir, chunk_idx)

        # Prepare work items (round-robin GPU assignment if more chunks than GPUs)
        work_items = []
        for i, chunk in enumerate(chunks):
            gpu_idx = selected_gpus[i % len(selected_gpus)]
            work_items.append((i, chunk, gpu_idx))

        # Run parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk, item) for item in work_items]
            for future in as_completed(futures):
                result = future.result()
                chunk_results.append(result)

        # Merge results
        markdown_content, total_images = merge_chunk_results(
            chunk_results,
            assets_dir,
            image_format,
            image_quality,
            image_lossless,
        )

        if not markdown_content:
            output_error("All MinerU chunk processes failed")

        print(f"  Merged {len(chunks)} chunks, {total_images} images", file=sys.stderr)

        detected_title = extract_title_from_markdown(markdown_content)
        return markdown_content, detected_title

    except subprocess.TimeoutExpired:
        output_error(
            "MinerU conversion timed out (>10 minutes)",
            "PDF may be too large. Try --engine docling instead.",
        )
    except FileNotFoundError:
        output_error(
            "MinerU command not found",
            "Install MinerU: uv pip install -U 'mineru[all]'",
        )
    except Exception as e:
        output_error(f"MinerU conversion failed: {e}")


# ============================================================================
# File Organization
# ============================================================================


def setup_paper_directory(
    pdf_path: Path,
    markdown_content: str,
    engine: str,
    detected_title: str | None,
    output_dir: str | None = None,
    allow_duplicate: bool = False,
) -> dict:
    """
    Organize files with timestamped folder naming.

    Structure:
      {cwd}/{YYYYMMDD}-{Sanitized_Title}/
        reference.pdf    - Original PDF (copied)
        full_text.md     - Converted Markdown with YAML frontmatter
        notes.md         - Empty file for analysis notes
        assets/          - Extracted images (if any)
    """
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    date_iso = today.strftime("%Y-%m-%d")

    # Use detected title or fall back to filename
    title_source = detected_title if detected_title else pdf_path.stem
    sanitized_title = sanitize_filename(title_source)

    output_root = get_output_root(output_dir)

    # Check for duplicates
    if not allow_duplicate and check_duplicate(sanitized_title, output_root):
        output_error(
            f"Duplicate detected: A folder with title '{sanitized_title}' already exists",
            "Remove the existing folder or rename if you want to re-ingest",
        )

    # Create timestamped folder name
    folder_name = f"{date_str}-{sanitized_title}"
    paper_dir = output_root / folder_name

    # Create directory
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Copy original PDF
    reference_pdf = paper_dir / "reference.pdf"
    shutil.copy2(pdf_path, reference_pdf)

    # Create YAML frontmatter
    display_title = (
        detected_title if detected_title else sanitized_title.replace("_", " ")
    )
    frontmatter = f"""---
title: "{display_title}"
date_ingested: {date_iso}
source_pdf: reference.pdf
conversion_engine: {engine}
tags:
  - paper
  - inbox
aliases: []
---

"""

    # Save Markdown with frontmatter
    full_text_path = paper_dir / "full_text.md"
    full_text_path.write_text(frontmatter + markdown_content, encoding="utf-8")

    # Create empty notes file
    notes_path = paper_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(f"# Notes: {display_title}\n\n", encoding="utf-8")

    return {
        "paper_dir": str(paper_dir),
        "markdown_path": str(full_text_path),
        "reference_pdf": str(reference_pdf),
        "notes_path": str(notes_path),
        "title": sanitized_title,
        "date": date_iso,
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF paper and convert to Markdown"
    )
    start_time = time.time()
    parser.add_argument(
        "pdf_source", type=str, help="Path to local PDF file OR URL to download"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["mineru", "docling"],
        default="mineru",
        help="Conversion engine: mineru (default, highest quality, GPU), docling (fallback, fast)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: current working directory)",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=4.0,
        help="Image scale factor for extraction (1.0 ~= 72 DPI). Use >1 for higher resolution.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="source",
        choices=["source", "png", "jpg", "jpeg", "webp"],
        help="Output image format. 'source' keeps original format when available.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="Image quality for lossy formats (1-100).",
    )
    parser.add_argument(
        "--image-lossless",
        action="store_true",
        help="Use lossless encoding when supported (webp).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting when a folder with the same title exists",
    )

    args = parser.parse_args()

    try:
        # Handle URL or local path
        temp_pdf = None
        source_is_url = is_url(args.pdf_source)
        if source_is_url:
            pdf_path = download_pdf(args.pdf_source)
            temp_pdf = pdf_path.parent  # Remember temp dir for cleanup
        else:
            pdf_path = Path(args.pdf_source).resolve()
            if not pdf_path.exists():
                output_error(f"File not found: {pdf_path}")
            if not pdf_path.suffix.lower() == ".pdf":
                output_error(f"Not a PDF file: {pdf_path}")

        image_format = normalize_image_format(args.image_format)
        image_quality = clamp_image_quality(args.image_quality)
        image_lossless = args.image_lossless

        # Convert based on engine
        engine = args.engine
        temp_assets = Path(tempfile.mkdtemp()) / "assets"
        temp_assets.mkdir(parents=True, exist_ok=True)

        if engine == "mineru":
            markdown_content, detected_title = convert_with_mineru(
                pdf_path,
                temp_assets,
                image_format,
                image_quality,
                image_lossless,
            )
        elif engine == "docling":
            markdown_content, detected_title = convert_with_docling(
                pdf_path,
                temp_assets,
                args.images_scale,
                image_format,
                image_quality,
                image_lossless,
            )

        # Normalize math delimiters to $...$ / $$...$$
        markdown_content = apply_outside_code_blocks(
            markdown_content, normalize_math_delimiters
        )
        # Wrap common inline math patterns (subscripts, greek letters, etc.)
        markdown_content = wrap_inline_math(markdown_content)
        resolved_title = resolve_paper_title(detected_title, markdown_content, pdf_path)
        if not resolved_title and not source_is_url:
            resolved_title = pdf_path.stem
        if not resolved_title:
            resolved_title = "untitled_paper"

        # Organize files
        paths = setup_paper_directory(
            pdf_path,
            markdown_content,
            engine,
            resolved_title,
            args.output_dir,
            args.force,
        )

        # Move assets to final location (for docling and mineru)
        if engine in ("docling", "mineru") and temp_assets and temp_assets.exists():
            final_assets = Path(paths["paper_dir"]) / "assets"
            if any(temp_assets.iterdir()):
                shutil.copytree(temp_assets, final_assets, dirs_exist_ok=True)
            shutil.rmtree(temp_assets.parent, ignore_errors=True)

        # Output success JSON
        output_json(
            {
                "status": "success",
                "markdown_path": paths["markdown_path"],
                "engine_used": engine,
                "title": paths["title"],
                "date": paths["date"],
                "paper_dir": paths["paper_dir"],
            }
        )

    finally:
        # Cleanup temp download
        if temp_pdf:
            shutil.rmtree(temp_pdf, ignore_errors=True)

        elapsed_time = time.time() - start_time

        # Enhanced logging with conversion details
        print("\n" + "=" * 50, file=sys.stderr)
        print("Conversion Summary", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

        # Get conversion metadata
        if engine == "mineru":
            meta = get_conversion_metadata()
            mode = meta.get("mode", "cli")
            print(f"  Backend: MinerU ({mode} mode)", file=sys.stderr)
            if mode == "api":
                print(
                    f"  API Server: {meta.get('api_host')}:{meta.get('api_port')}",
                    file=sys.stderr,
                )
            else:
                print(f"  GPUs: {meta.get('gpus', 'default')}", file=sys.stderr)
                print(f"  Workers: {meta.get('num_workers', 1)}", file=sys.stderr)
            print(f"  Batch Ratio: {meta.get('batch_ratio', 8)}", file=sys.stderr)
            print(f"  Pages: {meta.get('page_count', 'unknown')}", file=sys.stderr)
            print(f"  Image Format: {meta.get('image_format', image_format)}", file=sys.stderr)
            print(f"  Image Quality: {meta.get('image_quality', image_quality)}", file=sys.stderr)
            if meta.get("image_lossless", image_lossless):
                print("  Image Lossless: true", file=sys.stderr)
        elif engine == "docling":
            print("  Backend: Docling", file=sys.stderr)
            print(f"  Images Scale: {args.images_scale}", file=sys.stderr)
            print(f"  Image Format: {image_format}", file=sys.stderr)
            print(f"  Image Quality: {image_quality}", file=sys.stderr)
            if image_lossless:
                print("  Image Lossless: true", file=sys.stderr)

        print(f"  Total Time: {elapsed_time:.2f}s", file=sys.stderr)

        # Calculate pages per second if we have page count
        if engine == "mineru":
            meta = get_conversion_metadata()
            page_count = meta.get("page_count", 0)
            if page_count > 0 and elapsed_time > 0:
                pps = page_count / elapsed_time
                print(f"  Speed: {pps:.2f} pages/sec", file=sys.stderr)

        print("=" * 50, file=sys.stderr)


if __name__ == "__main__":
    main()
