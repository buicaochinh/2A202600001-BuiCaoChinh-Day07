from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text:
            return []

        # Split on sentence boundaries: . ! ? followed by space or newline, or standalone newlines
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # Fallback: split by character if no separators left
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Special case: empty separator means split by character
        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        parts = current_text.split(separator)
        final_chunks: list[str] = []
        current_buffer = ""

        for part in parts:
            # If buffer is empty, start with part. Else, try adding separator + part.
            proposed = (current_buffer + separator + part) if current_buffer else part

            if len(proposed) <= self.chunk_size:
                current_buffer = proposed
            else:
                # Flush the current buffer
                if current_buffer:
                    final_chunks.append(current_buffer)
                    current_buffer = ""

                # If the part itself is too long, recurse on it with the NEXT separator
                if len(part) > self.chunk_size:
                    final_chunks.extend(self._split(part, next_separators))
                else:
                    current_buffer = part

        if current_buffer:
            final_chunks.append(current_buffer)

        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    dot_prod = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_prod / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        results = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            if not chunks:
                results[name] = {"count": 0, "avg_length": 0.0, "chunks": []}
                continue

            avg_len = sum(len(c) for c in chunks) / len(chunks)
            results[name] = {
                "count": len(chunks),
                "avg_length": avg_len,
                "chunks": chunks,
            }
        return results
