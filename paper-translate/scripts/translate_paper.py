#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "markdown-it-py>=3.0.0",
#     "openai>=1.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Paper Translation Tool - Translate research paper markdown to Chinese.

Preserves markdown structure including:
  - LaTeX formulas ($...$ and $$...$$)
  - Code blocks and inline code
  - Image references and links
  - YAML frontmatter
  - HTML tables and tags
  - Citation references [1], [17], etc.

Usage:
  uv run translate_paper.py <markdown_file> [--backend deepseek|tensorblock] [--target-lang <language>]
"""

import argparse
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from markdown_it import MarkdownIt


# ============================================================================
# Configuration
# ============================================================================

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env_file(path: Path) -> None:
    """Load environment variables from a .env file."""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and (value[0] == value[-1]) and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)
    except Exception:
        return


load_env_file(ENV_FILE)

# API Keys from environment
TENSORBLOCK_API_KEY = os.environ.get("TENSORBLOCK_API_KEY", "")
TENSORBLOCK_BASE_URL = os.environ.get(
    "TENSORBLOCK_BASE_URL",
    "https://api.forge.tensorblock.co/v1"
)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com/v1"
)

# Translation settings
DEFAULT_MAX_CHARS_PER_CHUNK = 3000  # DeepSeek default
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ============================================================================
# LLM Backend Abstraction
# ============================================================================


@dataclass
class TranslationResult:
    """Result of a translation request."""
    text: str
    success: bool
    error: str | None = None


class LLMBackend(ABC):
    """Abstract base class for LLM translation backends."""

    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> TranslationResult:
        """Translate text to target language."""
        pass


class OpenAICompatibleBackend(LLMBackend):
    """Backend for OpenAI-compatible APIs (TensorBlock, DeepSeek)."""

    def __init__(self, api_key: str, base_url: str, model: str, max_chars_per_chunk: int):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_chars_per_chunk = max_chars_per_chunk

    def translate(self, text: str, target_lang: str) -> TranslationResult:
        """Translate text using OpenAI-compatible chat API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        system_prompt = f"""You are a professional translator specializing in academic papers. 
Translate the following text to {target_lang}.

CRITICAL RULES:
1. Translate ONLY the text content - do NOT translate or modify:
   - LaTeX formulas (anything between $ or $$)
   - Markdown formatting symbols (*, **, #, -, etc.)
   - Image references ![...](...) 
   - Link URLs (only translate link text, keep URL intact)
   - Code or technical identifiers
   - Citation references like [1], [17]
   - Author names and affiliations
   - Mathematical variable names and symbols

2. Maintain the exact same formatting and structure
3. Preserve all newlines and paragraph breaks
4. Use accurate academic/technical terminology
5. Output ONLY the translated text, no explanations"""

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                )
                translated = response.choices[0].message.content
                return TranslationResult(text=translated, success=True)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return TranslationResult(
                    text="",
                    success=False,
                    error=f"Translation failed after {MAX_RETRIES} attempts: {e}"
                )


class TensorBlockBackend(OpenAICompatibleBackend):
    """TensorBlock Forge API backend."""

    def __init__(self):
        if not TENSORBLOCK_API_KEY:
            raise ValueError(
                "TENSORBLOCK_API_KEY is not set. "
                "Create paper-translate/.env or set the environment variable."
            )
        super().__init__(
            api_key=TENSORBLOCK_API_KEY,
            base_url=TENSORBLOCK_BASE_URL,
            model="tensorblock/gemini-3-flash-preview",  # Using DeepSeek-V3 on TensorBlock
            max_chars_per_chunk=8000,
        )


class DeepSeekBackend(OpenAICompatibleBackend):
    """DeepSeek API backend."""

    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Create paper-translate/.env or set the environment variable."
            )
        super().__init__(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model="deepseek-chat",
            max_chars_per_chunk=3000,
        )


def get_backend(name: str) -> LLMBackend:
    """Get translation backend by name."""
    backends = {
        "tensorblock": TensorBlockBackend,
        "deepseek": DeepSeekBackend,
    }
    if name not in backends:
        raise ValueError(f"Unknown backend: {name}. Available: {list(backends.keys())}")
    return backends[name]()


# ============================================================================
# Markdown Parser and Translator
# ============================================================================


class MarkdownTranslator:
    """Translates markdown while preserving structure."""

    # Patterns for content that should NOT be translated
    PROTECTED_PATTERNS = [
        # LaTeX display math
        (r'\$\$[\s\S]+?\$\$', 'LATEX_DISPLAY'),
        # LaTeX inline math
        (r'\$[^\$\n]+?\$', 'LATEX_INLINE'),
        # Code blocks with language
        (r'```[\s\S]*?```', 'CODE_BLOCK'),
        # Inline code
        (r'`[^`\n]+?`', 'INLINE_CODE'),
        # Image references
        (r'!\[[^\]]*\]\([^)]+\)', 'IMAGE_REF'),
        # Link URLs (but not link text)
        (r'\]\([^)]+\)', 'LINK_URL'),
        # HTML tags
        (r'<[^>]+>', 'HTML_TAG'),
        # Citation references [1], [17], etc.
        (r'\[\d+(?:,\s*\d+)*\]', 'CITATION'),
        # Horizontal rules
        (r'^---+$', 'HR'),
    ]

    def __init__(self, backend: LLMBackend, target_lang: str = "Chinese"):
        self.backend = backend
        self.target_lang = target_lang
        self.placeholder_map: dict[str, str] = {}
        self.placeholder_counter = 0
        self.max_chars_per_chunk = getattr(
            backend,
            "max_chars_per_chunk",
            DEFAULT_MAX_CHARS_PER_CHUNK,
        )

    def _create_placeholder(self, content: str, ptype: str) -> str:
        """Create a unique placeholder for protected content."""
        self.placeholder_counter += 1
        placeholder = f"⟦{ptype}_{self.placeholder_counter}⟧"
        self.placeholder_map[placeholder] = content
        return placeholder

    def _protect_content(self, text: str) -> str:
        """Replace protected content with placeholders."""
        result = text
        for pattern, ptype in self.PROTECTED_PATTERNS:
            def replacer(match, pt=ptype):
                return self._create_placeholder(match.group(0), pt)
            result = re.sub(pattern, replacer, result, flags=re.MULTILINE)
        return result

    def _restore_content(self, text: str) -> str:
        """Restore protected content from placeholders."""
        result = text
        for placeholder, original in self.placeholder_map.items():
            result = result.replace(placeholder, original)
        return result

    def _extract_frontmatter(self, content: str) -> tuple[str | None, str]:
        """Extract YAML frontmatter from markdown content."""
        if content.startswith('---'):
            # Find the closing ---
            end_match = re.search(r'\n---\n', content[3:])
            if end_match:
                frontmatter = content[:end_match.end() + 3]
                body = content[end_match.end() + 3:]
                return frontmatter, body
        return None, content

    def _add_language_to_frontmatter(self, frontmatter: str) -> str:
        """Add language tag to YAML frontmatter."""
        if not frontmatter:
            return frontmatter
        # Insert language before the closing ---
        lines = frontmatter.strip().split('\n')
        # Find where to insert (before last ---)
        insert_idx = len(lines) - 1
        lines.insert(insert_idx, f"language: {self.target_lang}")
        return '\n'.join(lines) + '\n'

    def _normalize_heading(self, heading_text: str) -> str:
        text = heading_text.strip().lower()
        text = re.sub(r'^\s*[\d\.\)\(]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[:：\s]+$', '', text)
        return text

    def _is_references_heading(self, heading_text: str) -> bool:
        normalized = self._normalize_heading(heading_text)
        patterns = [
            r'^references\b',
            r'^reference\b',
            r'^bibliography\b',
            r'^works cited\b',
            r'^literature cited\b',
            r'^参考文献$',
            r'^引用文献$',
            r'^参考资料$',
        ]
        return any(re.match(pattern, normalized) for pattern in patterns)

    def _strip_references_section(self, text: str) -> tuple[str, bool]:
        """Remove REFERENCES/BIBLIOGRAPHY section before translation."""
        md = MarkdownIt("commonmark")
        tokens = md.parse(text)
        lines = text.splitlines()
        headings: list[tuple[int, int, str]] = []

        for i, token in enumerate(tokens):
            if token.type != "heading_open" or not token.map:
                continue
            level = int(token.tag[1]) if token.tag and token.tag.startswith("h") else 6
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            heading_text = inline.content if inline and inline.type == "inline" else ""
            headings.append((token.map[0], level, heading_text))

        if not headings:
            return text, False

        ranges: list[tuple[int, int]] = []
        for idx, (start_line, level, heading_text) in enumerate(headings):
            if not self._is_references_heading(heading_text):
                continue
            end_line = len(lines)
            for next_start, next_level, _ in headings[idx + 1:]:
                if next_level <= level:
                    end_line = next_start
                    break
            if start_line < end_line:
                ranges.append((start_line, end_line))

        if not ranges:
            return text, False

        for start_line, end_line in sorted(ranges, reverse=True):
            start_line = max(0, start_line)
            end_line = min(len(lines), end_line)
            if start_line >= end_line:
                continue
            del lines[start_line:end_line]

        result = "\n".join(lines)
        if text.endswith("\n"):
            result += "\n"
        return result, True

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks for translation."""
        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\n+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # If adding this paragraph exceeds limit, save current chunk
            if current_length + para_length > self.max_chars_per_chunk and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_length + 2  # +2 for \n\n

        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def translate_markdown(self, content: str) -> tuple[str, bool, str | None]:
        """
        Translate markdown content while preserving structure.

        Returns: (translated_content, success, error_message)
        """
        # Extract and preserve frontmatter
        frontmatter, body = self._extract_frontmatter(content)

        # Reset placeholder state
        self.placeholder_map = {}
        self.placeholder_counter = 0

        # Remove references section before translation
        body, removed_refs = self._strip_references_section(body)
        if removed_refs:
            print("References section removed before translation.", file=sys.stderr)

        # Protect content that shouldn't be translated
        protected_body = self._protect_content(body)

        # Split into chunks for translation
        chunks = self._split_into_chunks(protected_body)

        print(f"Translating {len(chunks)} chunks...", file=sys.stderr)

        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...", file=sys.stderr)

            # Skip chunks that are mostly placeholders
            non_placeholder_text = re.sub(r'⟦[^⟧]+⟧', '', chunk)
            if len(non_placeholder_text.strip()) < 20:
                # Mostly protected content, keep as-is
                translated_chunks.append(chunk)
                continue

            result = self.backend.translate(chunk, self.target_lang)
            if not result.success:
                return "", False, result.error

            translated_chunks.append(result.text)
            time.sleep(0.5)  # Rate limiting

        # Join translated chunks
        translated_body = '\n\n'.join(translated_chunks)

        # Restore protected content
        translated_body = self._restore_content(translated_body)

        # Update frontmatter with language tag
        if frontmatter:
            frontmatter = self._add_language_to_frontmatter(frontmatter)
            result = frontmatter + '\n' + translated_body
        else:
            result = translated_body

        return result, True, None


# ============================================================================
# Main
# ============================================================================


def output_json(data: dict) -> None:
    """Print JSON to stdout."""
    print(json.dumps(data, ensure_ascii=False))


def output_error(message: str) -> None:
    """Output error JSON and exit."""
    output_json({"status": "error", "message": message})
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Translate research paper markdown to Chinese"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the markdown file to translate"
    )
    parser.add_argument(
        "--backend",
        choices=["deepseek", "tensorblock"],
        default="tensorblock",
        help="LLM backend to use (default: tensorblock)"
    )
    parser.add_argument(
        "--target-lang",
        default="Chinese",
        help="Target language (default: Chinese)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: input_file with _zh suffix)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        output_error(f"Input file not found: {args.input_file}")

    if not args.input_file.suffix == ".md":
        output_error("Input file must be a markdown file (.md)")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: add language suffix
        lang_suffix = args.target_lang.lower()[:2]  # e.g., "zh" for Chinese
        output_path = args.input_file.with_stem(f"{args.input_file.stem}_{lang_suffix}")

    print(f"Translating: {args.input_file}", file=sys.stderr)
    print(f"Backend: {args.backend}", file=sys.stderr)
    print(f"Target language: {args.target_lang}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    # Read input file
    try:
        content = args.input_file.read_text(encoding="utf-8")
    except Exception as e:
        output_error(f"Failed to read input file: {e}")

    # Initialize backend and translator
    try:
        backend = get_backend(args.backend)
    except Exception as e:
        output_error(f"Failed to initialize backend: {e}")

    translator = MarkdownTranslator(backend, args.target_lang)

    # Translate
    start_time = time.time()
    translated_content, success, error = translator.translate_markdown(content)
    elapsed = time.time() - start_time

    if not success:
        output_error(error or "Translation failed")

    # Write output
    try:
        output_path.write_text(translated_content, encoding="utf-8")
    except Exception as e:
        output_error(f"Failed to write output file: {e}")

    print(f"Translation complete in {elapsed:.1f}s", file=sys.stderr)

    # Output success JSON
    output_json({
        "status": "success",
        "output_path": str(output_path),
        "backend": args.backend,
        "target_lang": args.target_lang,
        "elapsed_seconds": round(elapsed, 1)
    })


if __name__ == "__main__":
    main()
