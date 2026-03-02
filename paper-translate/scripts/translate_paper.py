#!/usr/bin/env -S uv run
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
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from markdown_it import MarkdownIt


# ============================================================================
# Configuration
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
_ROOT_ENV = _SCRIPT_DIR.parent / ".env"
_LOCAL_ENV = _SCRIPT_DIR / ".env"


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


# Root .env takes priority (loaded first → setdefault locks the value),
# then local .env fills in any remaining keys.
load_env_file(_ROOT_ENV)
load_env_file(_LOCAL_ENV)

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
DEFAULT_MAX_CHARS_PER_CHUNK = 8000
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CHUNK_MISSING_RETRIES = 3
MAX_WORKERS_CAP = 24

# Pricing: (input_price_per_1K, output_price_per_1K) in USD
MODEL_PRICING = {
    "deepseek-chat": (0.00028, 0.00042),  # DeepSeek V3: $0.28/M in, $0.42/M out
    "tensorblock/gemini-3-flash-preview": (0.0005, 0.003),  # $0.50/M in, $3.00/M out
}


# ============================================================================
# Translation Cache
# ============================================================================

CACHE_VERSION = 1


def _compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TranslationCache:
    """Chunk-level translation cache with invalidation support."""

    version: int
    source_hash: str
    backend: str
    target_lang: str
    max_chars_per_chunk: int
    total_chunks: int
    chunks: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def create(
        cls,
        source_hash: str,
        backend: str,
        target_lang: str,
        max_chars: int,
        total_chunks: int,
    ) -> "TranslationCache":
        return cls(
            version=CACHE_VERSION,
            source_hash=source_hash,
            backend=backend,
            target_lang=target_lang,
            max_chars_per_chunk=max_chars,
            total_chunks=total_chunks,
        )

    @classmethod
    def load(cls, path: Path) -> "TranslationCache | None":
        """Load cache from file.  Returns None if missing or corrupt."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != CACHE_VERSION:
                return None
            return cls(
                version=data["version"],
                source_hash=data["source_hash"],
                backend=data["backend"],
                target_lang=data["target_lang"],
                max_chars_per_chunk=data["max_chars_per_chunk"],
                total_chunks=data["total_chunks"],
                chunks=data.get("chunks", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def is_valid(
        self,
        source_hash: str,
        backend: str,
        target_lang: str,
        max_chars: int,
        total_chunks: int,
    ) -> bool:
        """Check if cache is still valid for given parameters."""
        return (
            self.source_hash == source_hash
            and self.backend == backend
            and self.target_lang == target_lang
            and self.max_chars_per_chunk == max_chars
            and self.total_chunks == total_chunks
        )

    def get_chunk(self, index: int, expected_hash: str) -> str | None:
        """Get cached translation for a chunk, or None if miss / hash mismatch."""
        entry = self.chunks.get(str(index))
        if entry is None:
            return None
        if entry.get("source_hash") != expected_hash:
            return None
        return entry.get("translated")

    def set_chunk(self, index: int, source_hash: str, translated: str) -> None:
        """Store a chunk's translation result."""
        self.chunks[str(index)] = {
            "source_hash": source_hash,
            "translated": translated,
        }

    def set_chunk_and_save(
        self, index: int, source_hash: str, translated: str, path: Path,
    ) -> None:
        """Thread-safe set + atomic save."""
        with self._lock:
            self.set_chunk(index, source_hash, translated)
            self._save_unlocked(path)

    def save(self, path: Path) -> None:
        """Thread-safe atomic save."""
        with self._lock:
            self._save_unlocked(path)

    def _save_unlocked(self, path: Path) -> None:
        """Write cache to file atomically (caller must hold lock)."""
        tmp_path = path.with_suffix(".json.tmp")
        data = {
            "version": self.version,
            "source_hash": self.source_hash,
            "backend": self.backend,
            "target_lang": self.target_lang,
            "max_chars_per_chunk": self.max_chars_per_chunk,
            "total_chunks": self.total_chunks,
            "chunks": self.chunks,
        }
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        tmp_path.replace(path)

    def cached_count(self) -> int:
        return len(self.chunks)


# ============================================================================
# LLM Backend Abstraction
# ============================================================================


@dataclass
class UsageStats:
    """Accumulated API token usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0


@dataclass
class TimingStats:
    """Timing breakdown for translation pipeline."""
    preprocess_sec: float = 0.0
    api_wait_sec: float = 0.0
    postprocess_sec: float = 0.0
    total_sec: float = 0.0
    retry_count: int = 0
    chunk_count: int = 0
    cached_chunks: int = 0


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
    def translate(self, text: str, target_lang: str, context: str | None = None) -> TranslationResult:
        """Translate text to target language."""
        pass


class OpenAICompatibleBackend(LLMBackend):
    """Backend for OpenAI-compatible APIs (TensorBlock, DeepSeek)."""

    def __init__(self, api_key: str, base_url: str, model: str, max_chars_per_chunk: int):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_chars_per_chunk = max_chars_per_chunk
        self._usage_lock = threading.Lock()
        self._usage = UsageStats()

    def translate(self, text: str, target_lang: str, context: str | None = None) -> TranslationResult:
        """Translate text using OpenAI-compatible chat API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        system_prompt = f"""You are a professional translator specializing in academic papers.
Translate the following text to {target_lang}.

RULES:
1. Translate ONLY the natural language text
2. Do NOT translate or modify:
   - LaTeX formulas ($...$ and $$...$$)
   - Inline code (`...`)
   - Citation references like [1], [17]
   - Link URLs (translate link text only)
   - Author names, variable names, technical identifiers
3. Maintain the exact same paragraph structure
4. Use accurate academic/technical terminology
5. Output ONLY the translated text, no explanations"""

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Paper context (for reference only):\n{context}"
            })
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=8192,
                )
                translated = response.choices[0].message.content
                if response.usage:
                    with self._usage_lock:
                        self._usage.prompt_tokens += response.usage.prompt_tokens or 0
                        self._usage.completion_tokens += response.usage.completion_tokens or 0
                        self._usage.total_tokens += response.usage.total_tokens or 0
                        self._usage.api_calls += 1
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

    def get_usage(self) -> UsageStats:
        """Return accumulated usage stats (thread-safe snapshot)."""
        with self._usage_lock:
            return UsageStats(
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens,
                total_tokens=self._usage.total_tokens,
                api_calls=self._usage.api_calls,
            )

    def estimate_cost(self) -> float | None:
        """Estimate cost in USD based on model pricing."""
        pricing = MODEL_PRICING.get(self.model)
        if not pricing:
            return None
        input_price, output_price = pricing
        usage = self.get_usage()
        return round(
            (usage.prompt_tokens * input_price + usage.completion_tokens * output_price) / 1000,
            6,
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
# Markdown Parser and Translator (Structure-Aware Segmentation — B+)
# ============================================================================


@dataclass
class Block:
    """A block of content parsed from the markdown document."""
    content: str
    translatable: bool
    block_type: str  # 'text', 'latex_display', 'code_block', 'image', 'html', 'hr', 'heading'


class MarkdownTranslator:
    """Translates markdown while preserving structure.

    Uses structure-aware segmentation: the document is parsed into blocks
    (translatable text vs non-translatable formulas/code/images) and only
    translatable blocks are sent to the LLM.  Inline LaTeX, inline code,
    and citations remain in the text so the LLM can see surrounding context.
    """

    def __init__(
        self,
        backend: LLMBackend,
        target_lang: str = "Chinese",
        max_chars_per_chunk: int | None = None,
        chunk_missing_retries: int | None = None,
        backend_name: str | None = None,
    ):
        self.backend = backend
        self.backend_name = backend_name or backend.__class__.__name__
        self.target_lang = target_lang
        if max_chars_per_chunk is not None:
            self.max_chars_per_chunk = max_chars_per_chunk
        else:
            self.max_chars_per_chunk = getattr(
                backend,
                "max_chars_per_chunk",
                DEFAULT_MAX_CHARS_PER_CHUNK,
            )
        if chunk_missing_retries is not None:
            self.chunk_missing_retries = chunk_missing_retries
        else:
            self.chunk_missing_retries = CHUNK_MISSING_RETRIES

    # ------------------------------------------------------------------
    # Frontmatter helpers
    # ------------------------------------------------------------------

    def _extract_frontmatter(self, content: str) -> tuple[str | None, str]:
        """Extract YAML frontmatter from markdown content."""
        if content.startswith('---'):
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
        lines = frontmatter.strip().split('\n')
        insert_idx = len(lines) - 1
        lines.insert(insert_idx, f"language: {self.target_lang}")
        return '\n'.join(lines) + '\n'

    # ------------------------------------------------------------------
    # Heading / section helpers
    # ------------------------------------------------------------------

    def _normalize_heading(self, heading_text: str) -> str:
        text = heading_text.strip().lower()
        text = re.sub(r'^\s*[\d\.\)\(]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[:：\s]+$', '', text)
        return text

    def _is_abstract_heading(self, heading_text: str) -> bool:
        normalized = self._normalize_heading(heading_text)
        patterns = [
            r'^abstract\b',
            r'^summary\b',
            r'^概要$',
            r'^摘要$',
        ]
        return any(re.match(pattern, normalized) for pattern in patterns)

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

    def _collect_headings(self, text: str) -> list[tuple[int, int, int, str]]:
        """Collect markdown headings with line ranges."""
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

    def _strip_references_section(self, text: str) -> tuple[str, bool]:
        """Remove REFERENCES/BIBLIOGRAPHY section before translation."""
        lines = text.splitlines()
        headings = self._collect_headings(text)

        if not headings:
            return text, False

        ranges: list[tuple[int, int]] = []
        for idx, (start_line, _end_line, level, heading_text) in enumerate(headings):
            if not self._is_references_heading(heading_text):
                continue
            end_line = len(lines)
            for next_start, _next_end, next_level, _ in headings[idx + 1:]:
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

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def _extract_title(self, frontmatter: str | None, body: str) -> str | None:
        if frontmatter:
            for line in frontmatter.splitlines():
                match = re.match(r'^\s*title\s*:\s*(.+)\s*$', line, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip().strip('"').strip("'")

        headings = self._collect_headings(body)
        for _start_line, _end_line, _level, heading_text in headings:
            if not heading_text:
                continue
            if self._is_abstract_heading(heading_text) or self._is_references_heading(heading_text):
                continue
            return heading_text.strip()
        return None

    def _extract_abstract(self, body: str) -> str | None:
        lines = body.splitlines()
        headings = self._collect_headings(body)
        if not headings:
            return None

        for idx, (start_line, end_line, level, heading_text) in enumerate(headings):
            if not self._is_abstract_heading(heading_text):
                continue
            content_start = end_line
            content_end = len(lines)
            for next_start, _next_end, next_level, _ in headings[idx + 1:]:
                if next_level <= level:
                    content_end = next_start
                    break
            abstract_lines = lines[content_start:content_end]
            abstract_text = "\n".join(abstract_lines).strip()
            if abstract_text:
                return abstract_text
        return None

    def _build_context(self, frontmatter: str | None, body: str) -> str | None:
        title = self._extract_title(frontmatter, body)
        abstract = self._extract_abstract(body)

        if abstract and title:
            return f"Title: {title}\nAbstract:\n{abstract}"
        if title:
            return f"Title: {title}"
        return None

    # ------------------------------------------------------------------
    # Block parsing — structure-aware segmentation
    # ------------------------------------------------------------------

    def _parse_into_blocks(self, text: str) -> list[Block]:
        """Parse markdown text into an ordered list of translatable / non-translatable blocks.

        Non-translatable blocks are never sent to the LLM:
          - Display math ($$...$$)
          - Code blocks (```...```)
          - Standalone image references (![...](...)  on their own line)
          - Horizontal rules (---)
          - HTML table blocks (<table>...</table>)

        Translatable blocks (regular paragraphs, headings) are sent as-is,
        including any inline LaTeX, inline code, and citations they contain.
        """
        blocks: list[Block] = []
        lines = text.split('\n')
        i = 0
        pending_lines: list[str] = []

        def _flush_pending() -> None:
            """Flush accumulated pending lines as a translatable text block."""
            if not pending_lines:
                return
            raw = '\n'.join(pending_lines)
            stripped = raw.strip()
            if stripped:
                blocks.append(Block(raw, True, 'text'))
            pending_lines.clear()

        while i < len(lines):
            line = lines[i]

            # --- Display math: $$...$$ (may span multiple lines) ---
            if line.strip().startswith('$$'):
                if line.strip().endswith('$$') and len(line.strip()) > 4:
                    _flush_pending()
                    blocks.append(Block(line, False, 'latex_display'))
                    i += 1
                    continue
                _flush_pending()
                math_lines = [line]
                i += 1
                while i < len(lines):
                    math_lines.append(lines[i])
                    if lines[i].strip().endswith('$$'):
                        i += 1
                        break
                    i += 1
                blocks.append(Block('\n'.join(math_lines), False, 'latex_display'))
                continue

            # --- Code block: ```...``` ---
            if line.strip().startswith('```'):
                _flush_pending()
                code_lines = [line]
                i += 1
                while i < len(lines):
                    code_lines.append(lines[i])
                    if lines[i].strip().startswith('```') and len(code_lines) > 1:
                        i += 1
                        break
                    i += 1
                blocks.append(Block('\n'.join(code_lines), False, 'code_block'))
                continue

            # --- HTML table block ---
            if re.match(r'^\s*<table\b', line, re.IGNORECASE):
                _flush_pending()
                table_lines = [line]
                if not re.search(r'</table>', line, re.IGNORECASE):
                    i += 1
                    while i < len(lines):
                        table_lines.append(lines[i])
                        if re.search(r'</table>', lines[i], re.IGNORECASE):
                            i += 1
                            break
                        i += 1
                else:
                    i += 1
                blocks.append(Block('\n'.join(table_lines), False, 'html'))
                continue

            # --- Standalone image on its own line ---
            if re.match(r'^\s*!\[.*\]\(.*\)\s*$', line):
                _flush_pending()
                blocks.append(Block(line, False, 'image'))
                i += 1
                continue

            # --- Horizontal rule ---
            if re.match(r'^\s*---+\s*$', line):
                _flush_pending()
                blocks.append(Block(line, False, 'hr'))
                i += 1
                continue

            # --- Blank line: flush current pending block ---
            if line.strip() == '':
                if pending_lines:
                    _flush_pending()
                    blocks.append(Block('', False, 'hr'))
                else:
                    blocks.append(Block('', False, 'hr'))
                i += 1
                continue

            # --- Normal text line (paragraph, heading, list, etc.) ---
            pending_lines.append(line)
            i += 1

        _flush_pending()
        return blocks

    # ------------------------------------------------------------------
    # Chunking — group consecutive translatable blocks
    # ------------------------------------------------------------------

    def _group_translatable_chunks(
        self,
        blocks: list[Block],
    ) -> list[tuple[list[int], str]]:
        """Group consecutive translatable blocks into chunks for API calls.

        Returns a list of (block_indices, chunk_text) tuples.
        Only translatable blocks are included; non-translatable blocks are
        skipped entirely and will be reassembled from the original later.
        """
        chunks: list[tuple[list[int], str]] = []
        current_indices: list[int] = []
        current_parts: list[str] = []
        current_length = 0

        for idx, block in enumerate(blocks):
            if not block.translatable:
                continue

            block_length = len(block.content)

            # If adding this block exceeds limit, flush current chunk
            if current_length + block_length > self.max_chars_per_chunk and current_parts:
                chunks.append((list(current_indices), '\n\n'.join(current_parts)))
                current_indices = []
                current_parts = []
                current_length = 0

            current_indices.append(idx)
            current_parts.append(block.content)
            current_length += block_length + 2  # +2 for \n\n separator

        if current_parts:
            chunks.append((list(current_indices), '\n\n'.join(current_parts)))

        return chunks

    # ------------------------------------------------------------------
    # Post-translation validation (B+ Phase 7)
    # ------------------------------------------------------------------

    @staticmethod
    def _count_inline_dollars(text: str) -> int:
        """Count inline $ signs, excluding $$ display math delimiters."""
        # Remove $$ first, then count remaining $
        cleaned = re.sub(r'\$\$', '', text)
        return cleaned.count('$')

    @staticmethod
    def _count_headings(text: str) -> int:
        """Count markdown heading lines."""
        return len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))

    @staticmethod
    def _count_paragraphs(text: str) -> int:
        """Count non-empty paragraphs separated by blank lines."""
        parts = re.split(r'\n\n+', text.strip())
        return sum(1 for p in parts if p.strip())

    def _validate_chunk(
        self,
        original: str,
        translated: str,
        chunk_index: int,
    ) -> tuple[list[str], bool]:
        """Validate translated chunk against original.

        Returns (issues_list, has_hard_failure).
        Hard failures trigger retry; soft failures are logged as warnings.
        """
        issues: list[str] = []
        hard_fail = False

        # Check 1: Inline $ count (hard failure if mismatch > 2)
        orig_dollars = self._count_inline_dollars(original)
        trans_dollars = self._count_inline_dollars(translated)
        if abs(orig_dollars - trans_dollars) > 2:
            issues.append(
                f"inline $ mismatch: orig={orig_dollars} trans={trans_dollars}"
            )
            hard_fail = True

        # Check 2: Heading count (hard failure if any mismatch)
        orig_headings = self._count_headings(original)
        trans_headings = self._count_headings(translated)
        if orig_headings != trans_headings:
            issues.append(
                f"heading count mismatch: orig={orig_headings} trans={trans_headings}"
            )
            hard_fail = True

        # Check 3: Paragraph count ratio (soft failure)
        orig_paras = self._count_paragraphs(original)
        trans_paras = self._count_paragraphs(translated)
        if orig_paras > 0:
            ratio = trans_paras / orig_paras
            if ratio < 0.7 or ratio > 1.3:
                issues.append(
                    f"paragraph count drift: orig={orig_paras} trans={trans_paras} ratio={ratio:.2f}"
                )

        # Check 4: Length ratio (soft failure)
        # Chinese text is typically 0.4-1.0x the length of English
        if len(original) > 50:
            length_ratio = len(translated) / len(original)
            if length_ratio < 0.3 or length_ratio > 1.2:
                issues.append(
                    f"length ratio: {length_ratio:.2f} (expected 0.3-1.2)"
                )

        return issues, hard_fail

    # ------------------------------------------------------------------
    # Translation with validation
    # ------------------------------------------------------------------

    def _log_chunk(self, chunk_index: int, message: str) -> None:
        print(f"  Chunk {chunk_index + 1}: {message}", file=sys.stderr)

    def _translate_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        context: str | None,
    ) -> TranslationResult:
        """Translate a single chunk with validation and retry on hard failure."""
        result = self.backend.translate(chunk_text, self.target_lang, context)
        if not result.success:
            error_message = result.error or "unknown error"
            self._log_chunk(chunk_index, f"failed: {error_message}")
            return result

        # Validate translation
        issues, hard_fail = self._validate_chunk(
            chunk_text, result.text or "", chunk_index,
        )
        if issues:
            issue_str = "; ".join(issues)
            if hard_fail:
                self._log_chunk(chunk_index, f"validation hard fail: {issue_str}")
                # Retry with sub-chunks
                for retry in range(1, self.chunk_missing_retries + 1):
                    if hasattr(self, '_retry_count'):
                        self._retry_count += 1
                    # Split chunk and retry each half
                    paragraphs = re.split(r'\n\n+', chunk_text)
                    if len(paragraphs) <= 1:
                        self._log_chunk(chunk_index, f"retry {retry}: cannot split further, accepting")
                        break
                    mid = len(paragraphs) // 2
                    sub_a = '\n\n'.join(paragraphs[:mid])
                    sub_b = '\n\n'.join(paragraphs[mid:])
                    result_a = self.backend.translate(sub_a, self.target_lang, context)
                    result_b = self.backend.translate(sub_b, self.target_lang, context)
                    if result_a.success and result_b.success:
                        merged = (result_a.text or "") + "\n\n" + (result_b.text or "")
                        retry_issues, retry_hard = self._validate_chunk(
                            chunk_text, merged, chunk_index,
                        )
                        if not retry_hard:
                            self._log_chunk(chunk_index, f"ok after retry {retry}")
                            return TranslationResult(text=merged, success=True)
                        self._log_chunk(chunk_index, f"retry {retry} still failing: {'; '.join(retry_issues)}")
                    else:
                        err = result_a.error or result_b.error or "sub-chunk failed"
                        self._log_chunk(chunk_index, f"retry {retry} sub-chunk API error: {err}")
                # Accept best-effort after retries exhausted
                self._log_chunk(chunk_index, "accepting best-effort after retries")
            else:
                self._log_chunk(chunk_index, f"validation warning: {issue_str}")
        else:
            self._log_chunk(chunk_index, "ok")
        return result

    # ------------------------------------------------------------------
    # Reassembly with heading-anchored alignment (B+ Phase 8)
    # ------------------------------------------------------------------

    @staticmethod
    def _realign_chunk_to_blocks(
        chunk_translated: str,
        block_indices: list[int],
        blocks: list[Block],
    ) -> dict[int, str]:
        """Map translated chunk text back to individual block indices.

        Uses heading-anchored alignment when available, with proportional
        length fallback instead of the naive "everything to first block".
        """
        if len(block_indices) == 1:
            return {block_indices[0]: chunk_translated}

        # Split the translated chunk back into per-block pieces
        parts = chunk_translated.split('\n\n')
        if len(parts) == len(block_indices):
            return dict(zip(block_indices, parts))

        # -- Heading-anchored alignment --
        # Find heading positions in the original blocks
        heading_positions: list[tuple[int, str]] = []
        for pos, bi in enumerate(block_indices):
            block_content = blocks[bi].content
            if re.match(r'^#{1,6}\s', block_content.strip()):
                heading_positions.append((pos, block_content.strip().split('\n')[0]))

        if heading_positions:
            # Try to find matching headings in translated parts
            heading_indices_in_parts: list[tuple[int, int]] = []  # (part_idx, block_pos)
            for block_pos, heading_line in heading_positions:
                # Match by heading marker pattern (# count)
                heading_level = len(re.match(r'^(#+)', heading_line).group(1))
                for p_idx, part in enumerate(parts):
                    part_stripped = part.strip()
                    if part_stripped.startswith('#' * heading_level + ' '):
                        heading_indices_in_parts.append((p_idx, block_pos))
                        break

            if heading_indices_in_parts:
                result: dict[int, str] = {}
                # Build segments between heading anchors
                anchors = [(0, 0)] + heading_indices_in_parts
                for a_idx in range(len(anchors)):
                    part_start = anchors[a_idx][0]
                    block_start = anchors[a_idx][1]
                    if a_idx + 1 < len(anchors):
                        part_end = anchors[a_idx + 1][0]
                        block_end = anchors[a_idx + 1][1]
                    else:
                        part_end = len(parts)
                        block_end = len(block_indices)

                    segment_parts = parts[part_start:part_end]
                    segment_blocks = block_indices[block_start:block_end]
                    if len(segment_parts) == len(segment_blocks):
                        for bi, p in zip(segment_blocks, segment_parts):
                            result[bi] = p
                    else:
                        # Proportional assignment within segment
                        merged = '\n\n'.join(segment_parts)
                        if len(segment_blocks) == 1:
                            result[segment_blocks[0]] = merged
                        else:
                            total_orig_len = sum(len(blocks[bi].content) for bi in segment_blocks)
                            if total_orig_len == 0:
                                for bi in segment_blocks:
                                    result[bi] = ''
                                if segment_blocks:
                                    result[segment_blocks[0]] = merged
                            else:
                                offset = 0
                                for j, bi in enumerate(segment_blocks):
                                    proportion = len(blocks[bi].content) / total_orig_len
                                    if j == len(segment_blocks) - 1:
                                        result[bi] = merged[offset:]
                                    else:
                                        end = offset + int(len(merged) * proportion)
                                        # Snap to paragraph boundary
                                        snap = merged.find('\n\n', max(offset, end - 50))
                                        if snap != -1 and snap < end + 100:
                                            end = snap + 2
                                        result[bi] = merged[offset:end]
                                        offset = end

                if len(result) == len(block_indices):
                    return result

        # -- Proportional length fallback --
        merged = '\n\n'.join(parts) if len(parts) != len(block_indices) else chunk_translated
        total_orig_len = sum(len(blocks[bi].content) for bi in block_indices)
        result = {}
        if total_orig_len == 0:
            result[block_indices[0]] = merged
            for bi in block_indices[1:]:
                result[bi] = ''
        else:
            offset = 0
            for j, bi in enumerate(block_indices):
                proportion = len(blocks[bi].content) / total_orig_len
                if j == len(block_indices) - 1:
                    result[bi] = merged[offset:]
                else:
                    end = offset + int(len(merged) * proportion)
                    # Snap to nearest paragraph boundary
                    snap = merged.find('\n\n', max(offset, end - 50))
                    if snap != -1 and snap < end + 100:
                        end = snap + 2
                    result[bi] = merged[offset:end]
                    offset = end
        return result

    @staticmethod
    def _reassemble_blocks(
        blocks: list[Block],
        translated_map: dict[int, str],
    ) -> str:
        """Reassemble the document from original blocks and translated text.

        For non-translatable blocks the original content is used.
        For translatable blocks the translated text is used.
        """
        parts: list[str] = []
        for idx, block in enumerate(blocks):
            if idx in translated_map:
                parts.append(translated_map[idx])
            else:
                parts.append(block.content)
        return '\n'.join(parts)

    # ------------------------------------------------------------------
    # Main translation pipeline
    # ------------------------------------------------------------------

    def translate_markdown(
        self,
        content: str,
        cache_path: Path | None = None,
    ) -> tuple[str, bool, str | None, int, TimingStats]:
        """
        Translate markdown content while preserving structure.

        Returns: (translated_content, success, error_message, cached_chunks_used, timing_stats)
        """
        timing = TimingStats()
        total_start = time.time()
        preprocess_start = time.time()

        # 1. Extract and preserve frontmatter
        frontmatter, body = self._extract_frontmatter(content)

        # 2. Remove references section before translation
        body, removed_refs = self._strip_references_section(body)
        if removed_refs:
            print("References section removed before translation.", file=sys.stderr)

        # 3. Build context (title + abstract)
        context = self._build_context(frontmatter, body)
        if context:
            print("Context extracted for chunk translation.", file=sys.stderr)

        # 4. Parse body into blocks (translatable vs non-translatable)
        blocks = self._parse_into_blocks(body)
        translatable_count = sum(1 for b in blocks if b.translatable)
        non_translatable_count = sum(1 for b in blocks if not b.translatable)
        print(
            f"Parsed {len(blocks)} blocks: {translatable_count} translatable, "
            f"{non_translatable_count} non-translatable.",
            file=sys.stderr,
        )

        # 5. Group translatable blocks into chunks
        chunk_groups = self._group_translatable_chunks(blocks)
        chunk_texts = [text for _, text in chunk_groups]

        timing.preprocess_sec = time.time() - preprocess_start

        # Compute hashes for cache validation
        body_hash = _compute_hash(body)
        chunk_hashes = [_compute_hash(ct) for ct in chunk_texts]

        # Load and validate cache
        cache: TranslationCache | None = None
        cached_count = 0
        if cache_path:
            cache = TranslationCache.load(cache_path)
            if cache and not cache.is_valid(
                source_hash=body_hash,
                backend=self.backend_name,
                target_lang=self.target_lang,
                max_chars=self.max_chars_per_chunk,
                total_chunks=len(chunk_texts),
            ):
                print("Cache invalidated (parameters changed), starting fresh.", file=sys.stderr)
                cache = None

        # Pre-populate translated_chunks from cache
        translated_chunks: list[str | None] = [None] * len(chunk_texts)
        if cache:
            for i in range(len(chunk_texts)):
                cached_text = cache.get_chunk(i, chunk_hashes[i])
                if cached_text is not None:
                    translated_chunks[i] = cached_text
                    cached_count += 1
            if cached_count > 0:
                print(f"Resumed {cached_count}/{len(chunk_texts)} chunks from cache.", file=sys.stderr)

        # Initialize active cache for incremental saving
        active_cache: TranslationCache | None = None
        if cache_path:
            active_cache = TranslationCache.create(
                source_hash=body_hash,
                backend=self.backend_name,
                target_lang=self.target_lang,
                max_chars=self.max_chars_per_chunk,
                total_chunks=len(chunk_texts),
            )
            if cache:
                for i in range(len(chunk_texts)):
                    if translated_chunks[i] is not None:
                        active_cache.set_chunk(i, chunk_hashes[i], translated_chunks[i])

        timing.chunk_count = len(chunk_texts)
        timing.cached_chunks = cached_count
        print(f"Translating {len(chunk_texts)} chunks ({cached_count} cached)...", file=sys.stderr)

        # 6. Translate chunks in parallel
        api_start = time.time()
        self._retry_count = 0

        futures: dict = {}
        pending_count = sum(1 for c in translated_chunks if c is None)
        num_workers = min(pending_count, MAX_WORKERS_CAP)
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            for i, chunk_text in enumerate(chunk_texts):
                if translated_chunks[i] is not None:
                    print(f"  Chunk {i+1}/{len(chunk_texts)}: cached", file=sys.stderr)
                    continue

                # Skip chunks with negligible translatable content
                if len(chunk_text.strip()) < 20:
                    translated_chunks[i] = chunk_text
                    continue

                print(f"  Chunk {i+1}/{len(chunk_texts)}...", file=sys.stderr)
                future = executor.submit(
                    self._translate_chunk,
                    chunk_text,
                    i,
                    context,
                )
                futures[future] = i

            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                if not result.success:
                    # Save partial progress before failing
                    if active_cache and cache_path:
                        active_cache.save(cache_path)
                        print(
                            f"Partial progress saved to {cache_path} "
                            f"({active_cache.cached_count()}/{len(chunk_texts)} chunks cached)",
                            file=sys.stderr,
                        )
                    for pending in futures:
                        pending.cancel()
                    timing.api_wait_sec = time.time() - api_start
                    timing.total_sec = time.time() - total_start
                    return "", False, result.error, cached_count, timing
                translated_chunks[index] = result.text
                # Incrementally save to cache
                if active_cache and cache_path:
                    active_cache.set_chunk_and_save(
                        index, chunk_hashes[index], result.text, cache_path,
                    )

        timing.api_wait_sec = time.time() - api_start

        if any(chunk is None for chunk in translated_chunks):
            if active_cache and cache_path:
                active_cache.save(cache_path)
            timing.total_sec = time.time() - total_start
            return "", False, "Translation failed: missing chunk results.", cached_count, timing

        # 7. Reassemble all blocks in original order
        postprocess_start = time.time()

        translated_map: dict[int, str] = {}
        for chunk_idx, (block_indices, _) in enumerate(chunk_groups):
            chunk_translated = translated_chunks[chunk_idx]
            if chunk_translated is None:
                continue
            aligned = self._realign_chunk_to_blocks(
                chunk_translated, block_indices, blocks,
            )
            translated_map.update(aligned)

        translated_body = self._reassemble_blocks(blocks, translated_map)

        # Update frontmatter with language tag
        if frontmatter:
            frontmatter = self._add_language_to_frontmatter(frontmatter)
            final_result = frontmatter + '\n' + translated_body
        else:
            final_result = translated_body

        timing.postprocess_sec = time.time() - postprocess_start

        # Success: delete cache
        if cache_path and cache_path.exists():
            cache_path.unlink()
            print("Cache file removed (translation complete).", file=sys.stderr)

        timing.retry_count = getattr(self, '_retry_count', 0)
        timing.total_sec = time.time() - total_start
        return final_result, True, None, cached_count, timing


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
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=None,
        help="Override max chars per chunk (default: backend setting)"
    )
    parser.add_argument(
        "--chunk-retries",
        type=int,
        default=None,
        help="Retries for missing paragraph markers (default: config)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume translation from cache if available",
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

    translator = MarkdownTranslator(
        backend,
        args.target_lang,
        max_chars_per_chunk=args.max_chars_per_chunk,
        chunk_missing_retries=args.chunk_retries,
        backend_name=args.backend,
    )

    print(
        f"Max chars per chunk: {translator.max_chars_per_chunk}",
        file=sys.stderr,
    )
    print(
        f"Chunk missing retries: {translator.chunk_missing_retries}",
        file=sys.stderr,
    )

    # Determine cache path
    cache_path = None
    if args.resume:
        cache_path = args.input_file.parent / ".translate_cache.json"
        print(f"Resume mode: cache at {cache_path}", file=sys.stderr)

    # Translate
    start_time = time.time()
    translated_content, success, error, cached_used, timing = translator.translate_markdown(
        content, cache_path=cache_path,
    )
    elapsed = time.time() - start_time

    if not success:
        output_error(error or "Translation failed")

    # Write output
    try:
        output_path.write_text(translated_content, encoding="utf-8")
    except Exception as e:
        output_error(f"Failed to write output file: {e}")

    print(f"Translation complete in {elapsed:.1f}s", file=sys.stderr)

    # Token usage summary
    usage = backend.get_usage()
    cost = backend.estimate_cost()

    print(f"\n{'='*50}", file=sys.stderr)
    print("Token Usage Summary", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"  API Calls:         {usage.api_calls}", file=sys.stderr)
    print(f"  Prompt Tokens:     {usage.prompt_tokens:,}", file=sys.stderr)
    print(f"  Completion Tokens: {usage.completion_tokens:,}", file=sys.stderr)
    print(f"  Total Tokens:      {usage.total_tokens:,}", file=sys.stderr)
    if cost is not None:
        print(f"  Estimated Cost:    ${cost:.4f} USD", file=sys.stderr)
    else:
        print(f"  Estimated Cost:    (pricing not available for {backend.model})", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    # Timing breakdown
    print(f"\n{'='*50}", file=sys.stderr)
    print("Timing Breakdown", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"  Preprocess:   {timing.preprocess_sec:.2f}s", file=sys.stderr)
    print(f"  API Wait:     {timing.api_wait_sec:.2f}s", file=sys.stderr)
    print(f"  Postprocess:  {timing.postprocess_sec:.2f}s", file=sys.stderr)
    print(f"  Total:        {timing.total_sec:.2f}s", file=sys.stderr)
    print(f"  Chunks:       {timing.chunk_count} ({timing.cached_chunks} cached)", file=sys.stderr)
    print(f"  Retries:      {timing.retry_count}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    # Output success JSON
    result_data = {
        "status": "success",
        "output_path": str(output_path),
        "backend": args.backend,
        "target_lang": args.target_lang,
        "elapsed_seconds": round(elapsed, 1),
        "cached_chunks": cached_used,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "api_calls": usage.api_calls,
        },
        "timing": {
            "preprocess_sec": round(timing.preprocess_sec, 3),
            "api_wait_sec": round(timing.api_wait_sec, 3),
            "postprocess_sec": round(timing.postprocess_sec, 3),
            "total_sec": round(timing.total_sec, 3),
            "chunk_count": timing.chunk_count,
            "cached_chunks": timing.cached_chunks,
            "retry_count": timing.retry_count,
        },
    }
    if cost is not None:
        result_data["estimated_cost_usd"] = cost
    output_json(result_data)


if __name__ == "__main__":
    main()
