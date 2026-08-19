"""Tests for the document ingestion pipeline."""
from __future__ import annotations

import pytest

from app.ingestion.pipeline import chunk_text, clean_text


class TestCleanText:
    def test_collapses_whitespace(self) -> None:
        raw = "Hello    world\t\there"
        assert clean_text(raw) == "Hello world here"

    def test_collapses_excess_newlines(self) -> None:
        raw = "Para 1\n\n\n\n\nPara 2"
        assert clean_text(raw) == "Para 1\n\nPara 2"

    def test_strips_trailing_whitespace(self) -> None:
        raw = "  hello  \n  "
        assert clean_text(raw) == "hello"


class TestChunkText:
    def test_single_chunk_when_text_fits(self) -> None:
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0][0] == text

    def test_multiple_chunks_on_long_text(self) -> None:
        text = "word " * 500   # 2500 chars
        chunks = chunk_text(text, chunk_size=256, chunk_overlap=32)
        assert len(chunks) > 1

    def test_overlap_positions(self) -> None:
        text = "A" * 600
        chunks = chunk_text(text, chunk_size=256, chunk_overlap=64)
        # Each chunk starts before the previous ended - overlap
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1][2]
            curr_start = chunks[i][1]
            assert curr_start < prev_end, "Chunks should overlap"

    def test_char_positions_are_correct(self) -> None:
        text = "Hello world, this is a test of chunking."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        for chunk_text_, start, end in chunks:
            # The chunk text should be a substring of the original
            assert text[start:end].strip().startswith(chunk_text_[:10].strip())

    def test_empty_text(self) -> None:
        chunks = chunk_text("", chunk_size=256, chunk_overlap=64)
        assert chunks == []

    def test_chunk_size_respected(self) -> None:
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
        for chunk_text_, start, end in chunks:
            assert len(chunk_text_) <= 300   # some slack for sentence boundary extension
