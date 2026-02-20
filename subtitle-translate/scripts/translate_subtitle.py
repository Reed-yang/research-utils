#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Subtitle Translation Tool - Translate SRT subtitle files using LLM.

Preserves SRT structure including:
  - Sequence numbers
  - Timestamps (start --> end)
  - Multi-line subtitle text
  - Blank line separators

Usage:
  uv run translate_subtitle.py <srt_file> [--backend deepseek|tensorblock] [--target-lang <language>]
"""

import argparse
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
DEFAULT_ENTRIES_PER_CHUNK = 40       # SRT entries per chunk
DEFAULT_MAX_CHARS_PER_CHUNK = 6000   # Fallback char limit per chunk
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CHUNK_MISSING_RETRIES = 3
MAX_WORKERS = 8

# Pricing: (input_price_per_1K, output_price_per_1K) in USD
MODEL_PRICING = {
    "deepseek-chat": (0.00028, 0.00042),  # DeepSeek V3: $0.28/M in, $0.42/M out
    "tensorblock/gemini-3-flash-preview": (0.0005, 0.003),  # $0.50/M in, $3.00/M out
}


# ============================================================================
# SRT Data Structures
# ============================================================================


@dataclass
class SRTEntry:
    """A single SRT subtitle entry."""
    index: int
    start_time: str
    end_time: str
    text: str  # May contain newlines for multi-line subtitles


@dataclass
class UsageStats:
    """Accumulated API token usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0


@dataclass
class TranslationResult:
    """Result of a translation request."""
    text: str
    success: bool
    error: str | None = None


# ============================================================================
# SRT Parser / Writer
# ============================================================================


def parse_srt(content: str) -> list[SRTEntry]:
    """Parse SRT content into a list of SRTEntry objects."""
    entries: list[SRTEntry] = []
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    # Remove BOM if present
    if content.startswith('\ufeff'):
        content = content[1:]

    # Split into blocks by blank lines
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split('\n')
        if len(lines) < 2:
            continue

        # First line: sequence number
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Second line: timestamp
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue

        start_time = time_match.group(1)
        end_time = time_match.group(2)

        # Remaining lines: subtitle text
        text = '\n'.join(lines[2:]).strip()

        entries.append(SRTEntry(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text,
        ))

    return entries


def write_srt(entries: list[SRTEntry]) -> str:
    """Convert SRTEntry list back to SRT format string."""
    parts: list[str] = []
    for entry in entries:
        parts.append(
            f"{entry.index}\n"
            f"{entry.start_time} --> {entry.end_time}\n"
            f"{entry.text}\n"
        )
    return '\n'.join(parts) + '\n'


# ============================================================================
# LLM Backend Abstraction
# ============================================================================


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

        system_prompt = f"""You are a professional subtitle translator.
Translate the following subtitle text to {target_lang}.

CRITICAL RULES:
1. You will receive subtitle text in a structured format with sequence markers like [SEQ_001].
   Each marker is followed by the subtitle text for that entry.
2. You MUST preserve ALL sequence markers [SEQ_XXX] exactly as-is in your output.
3. Translate ONLY the text content after each marker.
4. Keep translations concise and natural for subtitles — they must be readable at spoken pace.
5. Preserve speaker changes and dialogue structure.
6. Do NOT merge, split, reorder, or omit any subtitle entries.
7. Do NOT add any explanations, notes, or extra text.
8. Maintain the same number of lines per entry as the original when possible.
9. For technical terms, proper nouns, and brand names, keep them in original language or use well-known translations.
10. Output ONLY the translated subtitle text with markers, nothing else."""

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Subtitle context (for reference only):\n{context}"
            })
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4096,
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
                "Create subtitle-translate/.env or set the environment variable."
            )
        super().__init__(
            api_key=TENSORBLOCK_API_KEY,
            base_url=TENSORBLOCK_BASE_URL,
            model="tensorblock/gemini-3-flash-preview",
            max_chars_per_chunk=6000,
        )


class DeepSeekBackend(OpenAICompatibleBackend):
    """DeepSeek API backend."""

    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Create subtitle-translate/.env or set the environment variable."
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
# Subtitle Translator
# ============================================================================


class SubtitleTranslator:
    """Translates SRT subtitle entries while preserving structure."""

    def __init__(
        self,
        backend: LLMBackend,
        target_lang: str = "Chinese",
        entries_per_chunk: int | None = None,
        chunk_missing_retries: int | None = None,
    ):
        self.backend = backend
        self.target_lang = target_lang
        self.entries_per_chunk = entries_per_chunk or DEFAULT_ENTRIES_PER_CHUNK
        self.chunk_missing_retries = chunk_missing_retries or CHUNK_MISSING_RETRIES
        self.max_chars_per_chunk = getattr(
            backend,
            "max_chars_per_chunk",
            DEFAULT_MAX_CHARS_PER_CHUNK,
        )

    # ------------------------------------------------------------------
    # Chunk building: group SRT entries into translation batches
    # ------------------------------------------------------------------

    def _build_chunk_text(self, entries: list[SRTEntry]) -> str:
        """Build the text to send to LLM for a group of entries.

        Format:
            [SEQ_001] Subtitle text line 1
            line 2
            [SEQ_002] Next subtitle text
        """
        parts: list[str] = []
        for entry in entries:
            marker = f"[SEQ_{entry.index:03d}]"
            parts.append(f"{marker} {entry.text}")
        return '\n'.join(parts)

    def _split_into_chunks(
        self,
        entries: list[SRTEntry],
    ) -> list[list[SRTEntry]]:
        """Split entries into chunks respecting both entry count and char limit."""
        chunks: list[list[SRTEntry]] = []
        current_chunk: list[SRTEntry] = []
        current_chars = 0

        for entry in entries:
            entry_chars = len(entry.text) + 12  # overhead for marker
            should_split = (
                (len(current_chunk) >= self.entries_per_chunk and current_chunk)
                or (current_chars + entry_chars > self.max_chars_per_chunk and current_chunk)
            )
            if should_split:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0

            current_chunk.append(entry)
            current_chars += entry_chars

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # ------------------------------------------------------------------
    # Integrity checking: verify all sequence markers present
    # ------------------------------------------------------------------

    def _expected_markers(self, entries: list[SRTEntry]) -> list[str]:
        """Get expected sequence markers for a chunk."""
        return [f"[SEQ_{e.index:03d}]" for e in entries]

    def _check_markers(self, text: str, expected: list[str]) -> list[str]:
        """Return list of missing markers."""
        return [m for m in expected if m not in text]

    def _parse_translated_chunk(
        self,
        translated_text: str,
        entries: list[SRTEntry],
    ) -> list[SRTEntry] | None:
        """Parse LLM output back into SRTEntry objects.

        Returns None if parsing fails or entries are missing.
        """
        result: list[SRTEntry] = []
        # Build a map of expected entries for easy lookup
        entry_map = {e.index: e for e in entries}

        # Split by sequence markers
        # Pattern: [SEQ_XXX] followed by text until next marker or end
        pattern = r'\[SEQ_(\d+)\]\s*([\s\S]*?)(?=\[SEQ_\d+\]|$)'
        matches = re.findall(pattern, translated_text)

        if not matches:
            return None

        for idx_str, text in matches:
            idx = int(idx_str)
            if idx not in entry_map:
                continue
            original = entry_map[idx]
            translated_entry = SRTEntry(
                index=original.index,
                start_time=original.start_time,
                end_time=original.end_time,
                text=text.strip(),
            )
            result.append(translated_entry)

        # Check we got all entries
        if len(result) != len(entries):
            return None

        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_chunk(self, chunk_index: int, message: str) -> None:
        print(f"  Chunk {chunk_index + 1}: {message}", file=sys.stderr)

    def _log_subchunk(self, chunk_index: int, sub_index: int, sub_total: int, message: str) -> None:
        print(
            f"    Chunk {chunk_index + 1} sub {sub_index}/{sub_total}: {message}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Translation with retry
    # ------------------------------------------------------------------

    def _translate_entries_chunk(
        self,
        entries: list[SRTEntry],
        chunk_index: int,
        context: str | None,
    ) -> tuple[list[SRTEntry] | None, str | None]:
        """Translate a chunk of SRT entries. Returns (translated_entries, error)."""
        chunk_text = self._build_chunk_text(entries)
        expected = self._expected_markers(entries)

        result = self.backend.translate(chunk_text, self.target_lang, context)
        if not result.success:
            self._log_chunk(chunk_index, f"failed: {result.error}")
            return None, result.error

        # Check marker integrity
        missing = self._check_markers(result.text, expected)
        if not missing:
            parsed = self._parse_translated_chunk(result.text, entries)
            if parsed:
                self._log_chunk(chunk_index, "ok")
                return parsed, None
            else:
                self._log_chunk(chunk_index, "parse failed, will retry with smaller chunks")

        if missing:
            self._log_chunk(chunk_index, f"missing {len(missing)} markers: {', '.join(missing[:5])}...")

        # Retry with smaller sub-chunks
        for retry in range(1, self.chunk_missing_retries + 1):
            sub_size = max(3, len(entries) // (2 ** retry))
            sub_chunks = [
                entries[i:i + sub_size]
                for i in range(0, len(entries), sub_size)
            ]
            if len(sub_chunks) <= 1:
                continue

            self._log_chunk(
                chunk_index,
                f"retry {retry}/{self.chunk_missing_retries}: split into "
                f"{len(sub_chunks)} subchunks ({sub_size} entries each)",
            )

            all_translated: list[SRTEntry] = []
            sub_failed = False
            for si, sub_entries in enumerate(sub_chunks, start=1):
                sub_text = self._build_chunk_text(sub_entries)
                sub_expected = self._expected_markers(sub_entries)
                sub_result = self.backend.translate(sub_text, self.target_lang, context)

                if not sub_result.success:
                    self._log_subchunk(si, si, len(sub_chunks), f"failed: {sub_result.error}")
                    sub_failed = True
                    break

                sub_missing = self._check_markers(sub_result.text, sub_expected)
                if sub_missing:
                    self._log_subchunk(
                        chunk_index, si, len(sub_chunks),
                        f"missing markers: {', '.join(sub_missing[:3])}",
                    )
                    sub_failed = True
                    break

                parsed_sub = self._parse_translated_chunk(sub_result.text, sub_entries)
                if not parsed_sub:
                    self._log_subchunk(chunk_index, si, len(sub_chunks), "parse failed")
                    sub_failed = True
                    break

                self._log_subchunk(chunk_index, si, len(sub_chunks), "ok")
                all_translated.extend(parsed_sub)

            if not sub_failed:
                self._log_chunk(chunk_index, f"ok after retry {retry}")
                return all_translated, None

        return None, f"Chunk {chunk_index + 1}: failed after {self.chunk_missing_retries} retries"

    # ------------------------------------------------------------------
    # Main translation method
    # ------------------------------------------------------------------

    def _build_context(self, entries: list[SRTEntry]) -> str | None:
        """Build context from first few subtitle entries."""
        if not entries:
            return None
        preview = entries[:min(10, len(entries))]
        text = ' '.join(e.text.replace('\n', ' ') for e in preview)
        if len(text) > 500:
            text = text[:500] + "..."
        return f"The following subtitles are from a video. Initial content:\n{text}"

    def translate_srt(
        self,
        entries: list[SRTEntry],
    ) -> tuple[list[SRTEntry], bool, str | None]:
        """
        Translate SRT entries while preserving structure.

        Returns: (translated_entries, success, error_message)
        """
        if not entries:
            return [], True, None

        context = self._build_context(entries)
        if context:
            print("Context extracted from initial subtitles.", file=sys.stderr)

        # Split into chunks
        chunks = self._split_into_chunks(entries)
        print(f"Translating {len(chunks)} chunks ({len(entries)} entries total)...", file=sys.stderr)

        translated_all: list[list[SRTEntry] | None] = [None] * len(chunks)
        futures: dict = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for i, chunk_entries in enumerate(chunks):
                print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk_entries)} entries)...", file=sys.stderr)
                future = executor.submit(
                    self._translate_entries_chunk,
                    chunk_entries,
                    i,
                    context,
                )
                futures[future] = i

            for future in as_completed(futures):
                index = futures[future]
                translated_entries, error = future.result()
                if error:
                    for pending in futures:
                        pending.cancel()
                    return [], False, error
                translated_all[index] = translated_entries

        # Flatten results
        result: list[SRTEntry] = []
        for chunk_result in translated_all:
            if chunk_result is None:
                return [], False, "Translation failed: missing chunk results."
            result.extend(chunk_result)

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
        description="Translate SRT subtitle files using LLM"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the SRT file to translate"
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
        "--entries-per-chunk",
        type=int,
        default=None,
        help="Max SRT entries per chunk (default: 40)"
    )
    parser.add_argument(
        "--chunk-retries",
        type=int,
        default=None,
        help="Retries for missing markers (default: 3)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        output_error(f"Input file not found: {args.input_file}")

    if args.input_file.suffix.lower() not in (".srt",):
        output_error("Input file must be an SRT file (.srt)")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        lang_suffix = args.target_lang.lower()[:2]
        output_path = args.input_file.with_stem(f"{args.input_file.stem}_{lang_suffix}")

    print(f"Translating: {args.input_file}", file=sys.stderr)
    print(f"Backend: {args.backend}", file=sys.stderr)
    print(f"Target language: {args.target_lang}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    # Read and parse input file
    try:
        content = args.input_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = args.input_file.read_text(encoding="utf-8-sig")
        except Exception as e:
            output_error(f"Failed to read input file: {e}")
    except Exception as e:
        output_error(f"Failed to read input file: {e}")

    entries = parse_srt(content)
    if not entries:
        output_error("No valid SRT entries found in input file")

    print(f"Parsed {len(entries)} subtitle entries.", file=sys.stderr)

    # Initialize backend and translator
    try:
        backend = get_backend(args.backend)
    except Exception as e:
        output_error(f"Failed to initialize backend: {e}")

    translator = SubtitleTranslator(
        backend,
        args.target_lang,
        entries_per_chunk=args.entries_per_chunk,
        chunk_missing_retries=args.chunk_retries,
    )

    print(
        f"Entries per chunk: {translator.entries_per_chunk}",
        file=sys.stderr,
    )

    # Translate
    start_time = time.time()
    translated_entries, success, error = translator.translate_srt(entries)
    elapsed = time.time() - start_time

    if not success:
        output_error(error or "Translation failed")

    # Validate: all entries accounted for
    if len(translated_entries) != len(entries):
        output_error(
            f"Entry count mismatch: expected {len(entries)}, got {len(translated_entries)}"
        )

    # Write output
    output_text = write_srt(translated_entries)
    try:
        output_path.write_text(output_text, encoding="utf-8")
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

    # Output success JSON
    result_data = {
        "status": "success",
        "output_path": str(output_path),
        "backend": args.backend,
        "target_lang": args.target_lang,
        "entries_translated": len(translated_entries),
        "elapsed_seconds": round(elapsed, 1),
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "api_calls": usage.api_calls,
        },
    }
    if cost is not None:
        result_data["estimated_cost_usd"] = cost
    output_json(result_data)


if __name__ == "__main__":
    main()
