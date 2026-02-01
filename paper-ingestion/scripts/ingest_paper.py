#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "docling>=2.5.0",
#     "nougat-ocr>=0.1.17",
#     "torch>=2.0.0",
#     "pypdf>=4.0.0",
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
  uv run ingest_paper.py <pdf_path_or_url> [--engine docling|nougat] [--output-dir <path>] [--images-scale <float>] [--force]

Output:
  - Files organized in {cwd}/{YYYYMMDD}-{Sanitized_Title}/
  - Images saved to assets/ subfolder with relative paths in markdown
  - Single JSON object to stdout with status, paths, and metadata.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path
from urllib.parse import urlparse, unquote


# ============================================================================
# Configuration
# ============================================================================

def get_output_root(output_dir: str | None = None) -> Path:
    """Get output root directory. Defaults to current working directory."""
    if output_dir:
        return Path(output_dir).resolve()
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
    sanitized = re.sub(r'[:\?/\\*<>|"]', '', stem)
    # Replace spaces and multiple hyphens with underscores
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    # Remove any other non-word characters except underscores
    sanitized = re.sub(r'[^\w_]', '', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized


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
    return parsed.scheme in ('http', 'https')


def download_pdf(url: str) -> Path:
    """Download PDF from URL to a temporary file."""
    import requests
    
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Try to get filename from Content-Disposition or URL
        filename = None
        if 'Content-Disposition' in response.headers:
            cd = response.headers['Content-Disposition']
            if 'filename=' in cd:
                filename = cd.split('filename=')[-1].strip('"\'')
        
        if not filename:
            # Extract from URL path
            url_path = urlparse(url).path
            filename = unquote(Path(url_path).name)
        
        if not filename or not filename.lower().endswith('.pdf'):
            filename = 'downloaded_paper.pdf'
        
        # Save to temp file
        temp_dir = Path(tempfile.mkdtemp())
        temp_path = temp_dir / filename
        
        with open(temp_path, 'wb') as f:
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
            if re.match(r'^\d{8}-', dir_name):
                existing_title = dir_name[9:]  # Skip "YYYYMMDD-"
            else:
                existing_title = dir_name
            
            if existing_title == sanitized_title:
                return True
    return False


# ============================================================================
# Docling Backend (with Image Extraction)
# ============================================================================

def convert_with_docling(
    pdf_path: Path,
    assets_dir: Path,
    images_scale: float
) -> tuple[str, str | None]:
    """
    Convert PDF to Markdown using IBM Docling with image extraction.
    Returns (markdown_content, detected_title).
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        
        # Configure pipeline with image extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
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
        if hasattr(doc, 'name') and doc.name:
            detected_title = doc.name
        
        # Export markdown
        markdown_content = doc.export_to_markdown(image_mode="referenced")
        
        # Save images to assets directory and rewrite paths
        assets_dir.mkdir(parents=True, exist_ok=True)
        image_counter = 0
        
        if hasattr(doc, 'pictures') and doc.pictures:
            for pic in doc.pictures:
                if hasattr(pic, 'image') and pic.image:
                    image_counter += 1
                    image_name = f"image_{image_counter:03d}.png"
                    image_path = assets_dir / image_name
                    pic.image.pil_image.save(str(image_path))
        
        # Rewrite image paths to relative format ./assets/
        # Docling may use various formats, normalize to ./assets/image_xxx.png
        markdown_content = re.sub(
            r'!\[([^\]]*)\]\([^)]+/([^/)]+)\)',
            r'![\1](./assets/\2)',
            markdown_content
        )
        
        return markdown_content, detected_title
        
    except ImportError as e:
        output_error(
            f"Docling import failed: {e}",
            "Ensure docling is installed: uv pip install docling"
        )
    except Exception as e:
        output_error(f"Docling conversion failed: {e}")


# ============================================================================
# Nougat Backend
# ============================================================================

def convert_with_nougat(pdf_path: Path) -> tuple[str, str | None]:
    """
    Convert PDF to Markdown using Meta Nougat (GPU-intensive).
    Returns (markdown_content, detected_title).
    """
    try:
        import torch
        from nougat import NougatModel
        from nougat.utils.dataset import LazyDataset
        from nougat.utils.checkpoint import get_checkpoint
        from nougat.postprocessing import markdown_compatible
        
        # Check GPU availability
        if not torch.cuda.is_available():
            output_error(
                "Nougat requires CUDA GPU but none detected",
                "Try using --engine docling instead"
            )
        
        # Load model
        checkpoint = get_checkpoint(None, model_tag="0.1.0-small")
        model = NougatModel.from_pretrained(checkpoint)
        model = model.to(torch.bfloat16)
        model = model.cuda()
        model.eval()
        
        # Process PDF
        dataset = LazyDataset(
            pdf_path,
            partial(model.encoder.prepare_input, random_padding=False),
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )
        
        predictions = []
        for sample, is_last_page in dataloader:
            model_output = model.inference(image_tensors=sample.cuda())
            output = model_output["predictions"][0]
            if model.config.model_type == "nougat":
                output = markdown_compatible(output)
            predictions.append(output)
            if is_last_page:
                break
        
        markdown_content = "\n\n".join(predictions)
        
        # Try to extract title from first heading
        detected_title = None
        title_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
        if title_match:
            detected_title = title_match.group(1).strip()
        
        return markdown_content, detected_title
        
    except Exception as e:
        if 'OutOfMemoryError' in str(type(e).__name__):
            output_error(
                "Nougat failed: CUDA Out of Memory",
                "Try using --engine docling which is more memory-efficient"
            )
        elif isinstance(e, ImportError):
            output_error(
                f"Nougat import failed: {e}",
                "Ensure nougat-ocr is installed with GPU support"
            )
        else:
            output_error(
                f"Nougat conversion failed: {e}",
                "Try using --engine docling instead"
            )


# ============================================================================
# File Organization
# ============================================================================

def setup_paper_directory(
    pdf_path: Path,
    markdown_content: str,
    engine: str,
    detected_title: str | None,
    output_dir: str | None = None,
    allow_duplicate: bool = False
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
            "Remove the existing folder or rename if you want to re-ingest"
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
    display_title = detected_title if detected_title else sanitized_title.replace('_', ' ')
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
    parser.add_argument(
        "pdf_source",
        type=str,
        help="Path to local PDF file OR URL to download"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["docling", "nougat"],
        default="docling",
        help="Conversion engine: docling (default, fast) or nougat (slow, better for math)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: current working directory)"
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=1.0,
        help="Image scale factor for extraction (1.0 ~= 72 DPI). Use >1 for higher resolution."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting when a folder with the same title exists"
    )
    
    args = parser.parse_args()
    
    # Handle URL or local path
    temp_pdf = None
    if is_url(args.pdf_source):
        pdf_path = download_pdf(args.pdf_source)
        temp_pdf = pdf_path.parent  # Remember temp dir for cleanup
    else:
        pdf_path = Path(args.pdf_source).resolve()
        if not pdf_path.exists():
            output_error(f"File not found: {pdf_path}")
        if not pdf_path.suffix.lower() == ".pdf":
            output_error(f"Not a PDF file: {pdf_path}")
    
    try:
        # Prepare assets directory (will be created during conversion if needed)
        output_root = get_output_root(args.output_dir)
        
        # Convert based on engine
        engine = args.engine
        if engine == "docling":
            # Create temp assets dir, will be moved later
            temp_assets = Path(tempfile.mkdtemp()) / "assets"
            temp_assets.mkdir(parents=True, exist_ok=True)
            markdown_content, detected_title = convert_with_docling(
                pdf_path, temp_assets, args.images_scale
            )
        else:
            markdown_content, detected_title = convert_with_nougat(pdf_path)
            temp_assets = None
        
        # Organize files
        paths = setup_paper_directory(
            pdf_path, markdown_content, engine, detected_title, args.output_dir, args.force
        )
        
        # Move assets to final location (for docling)
        if engine == "docling" and temp_assets and temp_assets.exists():
            final_assets = Path(paths["paper_dir"]) / "assets"
            if any(temp_assets.iterdir()):
                shutil.copytree(temp_assets, final_assets, dirs_exist_ok=True)
            shutil.rmtree(temp_assets.parent, ignore_errors=True)
        
        # Output success JSON
        output_json({
            "status": "success",
            "markdown_path": paths["markdown_path"],
            "engine_used": engine,
            "title": paths["title"],
            "date": paths["date"],
            "paper_dir": paths["paper_dir"],
        })
        
    finally:
        # Cleanup temp download
        if temp_pdf:
            shutil.rmtree(temp_pdf, ignore_errors=True)


if __name__ == "__main__":
    main()
