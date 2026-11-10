"""
Tests for Text Chunking Module

This module contains tests for the TextChunker class, including:
- Chunk size and overlap configuration
- Chunk ordering and indexing
- Metadata preservation (chunk_id, document_id, page numbers)
- Page traceability across chunk boundaries
- Empty input handling
- Short documents
- Long documents
- Boundary cases
"""

import pytest

from chunking.text_chunker import (
    TextChunker,
    ChunkingError,
    EmptyInputError
)


class TestTextChunker:
    """Test cases for TextChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_initialization_valid(self):
        """Test successful initialization with valid parameters."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_initialization_invalid_chunk_size(self):
        """Test initialization with invalid chunk size."""
        with pytest.raises(ValueError) as exc_info:
            TextChunker(chunk_size=0, chunk_overlap=10)
        assert "chunk_size must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            TextChunker(chunk_size=-100, chunk_overlap=10)
        assert "chunk_size must be positive" in str(exc_info.value)

    def test_initialization_invalid_overlap(self):
        """Test initialization with invalid overlap."""
        with pytest.raises(ValueError) as exc_info:
            TextChunker(chunk_size=100, chunk_overlap=-10)
        assert "chunk_overlap must be non-negative" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            TextChunker(chunk_size=100, chunk_overlap=100)
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            TextChunker(chunk_size=100, chunk_overlap=150)
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

    def test_chunk_simple_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "This is a short text"
        result = self.chunker.chunk_simple_text(text, document_id="test_doc")

        assert result['success'] is True
        assert result['total_chunks'] == 1
        assert len(result['chunks']) == 1
        assert result['chunks'][0]['text'] == text
        assert result['chunks'][0]['chunk_index'] == 0
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == len(text)
        assert 'chunking_time_seconds' in result
        assert result['chunking_time_seconds'] >= 0

    def test_chunk_simple_text_exact_size(self):
        """Test chunking text exactly at chunk size."""
        text = "a" * 100  # Exactly chunk_size
        result = self.chunker.chunk_simple_text(text)

        assert result['total_chunks'] == 1
        assert result['chunks'][0]['chunk_size'] == 100

    def test_chunk_simple_text_multiple_chunks(self):
        """Test chunking text requiring multiple chunks."""
        text = "a" * 250  # 3 chunks: 0-99, 80-179, 160-249
        result = self.chunker.chunk_simple_text(text)

        assert result['total_chunks'] == 3
        assert result['chunks'][0]['chunk_index'] == 0
        assert result['chunks'][1]['chunk_index'] == 1
        assert result['chunks'][2]['chunk_index'] == 2

        # Verify overlap
        # Chunk 0: 0-99
        # Chunk 1: 80-179 (overlap of 20 chars from positions 80-99)
        # Chunk 2: 160-249 (overlap of 20 chars from positions 160-179)
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == 100
        assert result['chunks'][1]['start_char'] == 80
        assert result['chunks'][1]['end_char'] == 180
        assert result['chunks'][2]['start_char'] == 160
        assert result['chunks'][2]['end_char'] == 250

    def test_chunk_simple_text_metadata(self):
        """Test that chunk metadata is correctly assigned."""
        text = "This is a test text for metadata verification"
        result = self.chunker.chunk_simple_text(text, document_id="custom_doc_id")

        assert result['document_id'] == "custom_doc_id"
        assert result['chunk_size'] == 100
        assert result['chunk_overlap'] == 20

        chunk = result['chunks'][0]
        assert chunk['chunk_id'].startswith("custom_doc_id_chunk_")
        assert chunk['document_id'] == "custom_doc_id"
        assert chunk['chunk_index'] == 0
        assert 'start_char' in chunk
        assert 'end_char' in chunk
        assert 'chunk_size' in chunk

    def test_chunk_simple_text_auto_document_id(self):
        """Test that document ID is auto-generated when not provided."""
        text = "Test text"
        result1 = self.chunker.chunk_simple_text(text)
        result2 = self.chunker.chunk_simple_text(text)

        # Different document IDs should be generated
        assert result1['document_id'] != result2['document_id']
        assert result1['chunks'][0]['document_id'] != result2['chunks'][0]['document_id']

    def test_chunk_simple_text_empty(self):
        """Test chunking empty text."""
        with pytest.raises(EmptyInputError) as exc_info:
            self.chunker.chunk_simple_text("")
        assert "empty or contains only whitespace" in str(exc_info.value)

    def test_chunk_simple_text_whitespace_only(self):
        """Test chunking text with only whitespace."""
        with pytest.raises(EmptyInputError) as exc_info:
            self.chunker.chunk_simple_text("   \n\t  ")
        assert "empty or contains only whitespace" in str(exc_info.value)

    def test_chunk_extraction_result_single_page(self):
        """Test chunking extraction result with single page."""
        extraction_result = {
            'pages': [
                {'page_number': 1, 'text': 'a' * 150}
            ],
            'total_pages': 1,
            'pages_with_text': 1,
            'pages_without_text': [],
            'total_characters': 150,
            'success': True,
            'extraction_time_seconds': 0.1
        }

        result = self.chunker.chunk_extraction_result(extraction_result, document_id="doc1")

        assert result['success'] is True
        assert result['document_id'] == "doc1"
        assert result['total_chunks'] == 2  # 150 chars with size 100 and overlap 20
        assert 'chunking_time_seconds' in result
        assert result['chunking_time_seconds'] >= 0

        # Verify page metadata
        assert result['chunks'][0]['start_page'] == 1
        assert result['chunks'][0]['end_page'] == 1
        assert result['chunks'][1]['start_page'] == 1
        assert result['chunks'][1]['end_page'] == 1

    def test_chunk_extraction_result_multiple_pages(self):
        """Test chunking extraction result with multiple pages."""
        extraction_result = {
            'pages': [
                {'page_number': 1, 'text': 'a' * 80},
                {'page_number': 2, 'text': 'b' * 80},
                {'page_number': 3, 'text': 'c' * 80}
            ],
            'total_pages': 3,
            'pages_with_text': 3,
            'pages_without_text': [],
            'total_characters': 240,
            'success': True,
            'extraction_time_seconds': 0.1
        }

        result = self.chunker.chunk_extraction_result(extraction_result)

        assert result['success'] is True
        assert result['total_chunks'] > 0
        assert 'chunking_time_seconds' in result
        assert result['chunking_time_seconds'] >= 0

        # Verify at least one chunk spans multiple pages
        spans_multiple_pages = any(
            chunk['start_page'] != chunk['end_page']
            for chunk in result['chunks']
        )
        assert spans_multiple_pages, "Expected at least one chunk to span multiple pages"

    def test_chunk_extraction_result_page_traceability(self):
        """Test that page numbers are accurately tracked."""
        extraction_result = {
            'pages': [
                {'page_number': 1, 'text': 'Page 1 content '},
                {'page_number': 2, 'text': 'Page 2 content '},
                {'page_number': 3, 'text': 'Page 3 content'}
            ],
            'total_pages': 3,
            'pages_with_text': 3,
            'pages_without_text': [],
            'total_characters': 54,
            'success': True,
            'extraction_time_seconds': 0.1
        }

        result = self.chunker.chunk_extraction_result(extraction_result, document_id="trace_test")

        # Verify all chunks have page metadata
        for chunk in result['chunks']:
            assert 'start_page' in chunk
            assert 'end_page' in chunk
            assert isinstance(chunk['start_page'], int)
            assert isinstance(chunk['end_page'], int)
            assert chunk['start_page'] >= 1
            assert chunk['end_page'] >= chunk['start_page']
        assert 'chunking_time_seconds' in result
        assert result['chunking_time_seconds'] >= 0

    def test_chunk_extraction_result_empty_input(self):
        """Test chunking extraction result with no pages."""
        extraction_result = {
            'pages': [],
            'total_pages': 0,
            'pages_with_text': 0,
            'pages_without_text': [],
            'total_characters': 0,
            'success': True
        }

        with pytest.raises(EmptyInputError) as exc_info:
            self.chunker.chunk_extraction_result(extraction_result)
        assert "contains no pages with text" in str(exc_info.value)

    def test_chunk_extraction_result_whitespace_only(self):
        """Test chunking extraction result with only whitespace."""
        extraction_result = {
            'pages': [
                {'page_number': 1, 'text': '   \n\t  '}
            ],
            'total_pages': 1,
            'pages_with_text': 1,
            'pages_without_text': [],
            'total_characters': 6,
            'success': True
        }

        with pytest.raises(EmptyInputError) as exc_info:
            self.chunker.chunk_extraction_result(extraction_result)
        assert "empty or contains only whitespace" in str(exc_info.value)

    def test_chunk_result_structure(self):
        """Test that chunking result has the expected structure."""
        text = "Test text for structure verification"
        result = self.chunker.chunk_simple_text(text)

        # Check all expected keys are present
        expected_keys = ['chunks', 'total_chunks', 'document_id', 'chunk_size', 'chunk_overlap', 'success', 'chunking_time_seconds']
        for key in expected_keys:
            assert key in result

        # Check types
        assert isinstance(result['chunks'], list)
        assert isinstance(result['total_chunks'], int)
        assert isinstance(result['document_id'], str)
        assert isinstance(result['chunk_size'], int)
        assert isinstance(result['chunk_overlap'], int)
        assert isinstance(result['success'], bool)

        # Check chunk structure
        if result['chunks']:
            chunk = result['chunks'][0]
            chunk_keys = ['chunk_id', 'document_id', 'chunk_index', 'text', 'start_char', 'end_char', 'chunk_size']
            for key in chunk_keys:
                assert key in chunk

    def test_chunk_overlap_configuration(self):
        """Test that different overlap configurations work correctly."""
        text = "a" * 300

        # Zero overlap
        chunker_no_overlap = TextChunker(chunk_size=100, chunk_overlap=0)
        result = chunker_no_overlap.chunk_simple_text(text)
        assert result['total_chunks'] == 3
        assert result['chunks'][0]['end_char'] == 100
        assert result['chunks'][1]['start_char'] == 100  # No overlap

        # Large overlap
        chunker_large_overlap = TextChunker(chunk_size=100, chunk_overlap=50)
        result = chunker_large_overlap.chunk_simple_text(text)
        assert result['total_chunks'] == 5  # More chunks due to larger overlap
        assert result['chunks'][1]['start_char'] == 50  # 50 character overlap

    def test_very_long_text(self):
        """Test chunking of very long text."""
        text = "a" * 10000  # 10,000 characters
        result = self.chunker.chunk_simple_text(text)

        assert result['success'] is True
        assert result['total_chunks'] > 10  # Should create many chunks
        assert 'chunking_time_seconds' in result
        assert result['chunking_time_seconds'] >= 0

        # Verify chunk ordering
        for i in range(len(result['chunks']) - 1):
            assert result['chunks'][i]['chunk_index'] < result['chunks'][i + 1]['chunk_index']
            assert result['chunks'][i]['start_char'] < result['chunks'][i + 1]['start_char']

    def test_preserve_whitespace(self):
        """Test that whitespace is preserved in chunks."""
        text = "Word1  \n  Word2\t\tWord3"
        result = self.chunker.chunk_simple_text(text)

        # Should be one chunk since text is short
        assert result['total_chunks'] == 1
        assert result['chunks'][0]['text'] == text

    def test_chunk_simple_text_no_page_metadata(self):
        """Test that chunk_simple_text does not include page metadata."""
        text = "Test text without page information"
        result = self.chunker.chunk_simple_text(text)

        chunk = result['chunks'][0]
        # Should NOT have page metadata
        assert 'start_page' not in chunk
        assert 'end_page' not in chunk
        # Should have other metadata
        assert 'chunk_id' in chunk
        assert 'document_id' in chunk
        assert 'chunk_index' in chunk
        assert 'start_char' in chunk
        assert 'end_char' in chunk

    def test_character_offsets_inclusive_exclusive(self):
        """Test that start_char is inclusive and end_char is exclusive."""
        text = "0123456789"  # 10 characters
        chunker = TextChunker(chunk_size=5, chunk_overlap=0)
        result = chunker.chunk_simple_text(text)

        # Chunk 0: characters 0-4 (start_char=0, end_char=5)
        chunk0 = result['chunks'][0]
        assert chunk0['start_char'] == 0
        assert chunk0['end_char'] == 5
        assert chunk0['text'] == text[0:5]  # "01234"

        # Chunk 1: characters 5-9 (start_char=5, end_char=10)
        chunk1 = result['chunks'][1]
        assert chunk1['start_char'] == 5
        assert chunk1['end_char'] == 10
        assert chunk1['text'] == text[5:10]  # "56789"

        # Verify reconstruction
        reconstructed = text[chunk0['start_char']:chunk0['end_char']]
        assert reconstructed == chunk0['text']

    def test_page_traceability_across_boundaries(self):
        """Test that chunks spanning page boundaries have accurate page numbers."""
        # Create extraction result with known text on each page
        extraction_result = {
            'pages': [
                {'page_number': 1, 'text': 'A' * 40},  # Page 1: 40 chars
                {'page_number': 2, 'text': 'B' * 40},  # Page 2: 40 chars
                {'page_number': 3, 'text': 'C' * 40},  # Page 3: 40 chars
            ],
            'total_pages': 3,
            'pages_with_text': 3,
            'pages_without_text': [],
            'total_characters': 120,
            'success': True
        }

        # Use chunk_size=60, overlap=20
        # Expected chunks:
        # Chunk 0: chars 0-59 (pages 1-2)
        # Chunk 1: chars 40-99 (pages 2-3)
        # Chunk 2: chars 80-119 (page 3)
        chunker = TextChunker(chunk_size=60, chunk_overlap=20)
        result = chunker.chunk_extraction_result(extraction_result, document_id="boundary_test")

        # Chunk 0 should span pages 1-2
        chunk0 = result['chunks'][0]
        assert chunk0['start_page'] == 1
        assert chunk0['end_page'] == 2
        assert chunk0['start_char'] == 0
        assert chunk0['end_char'] == 60

        # Chunk 1 should span pages 2-3
        chunk1 = result['chunks'][1]
        assert chunk1['start_page'] == 2
        assert chunk1['end_page'] == 3
        assert chunk1['start_char'] == 40
        assert chunk1['end_char'] == 100

        # Chunk 2 should be on page 3 only
        chunk2 = result['chunks'][2]
        assert chunk2['start_page'] == 3
        assert chunk2['end_page'] == 3
        assert chunk2['start_char'] == 80
        assert chunk2['end_char'] == 120

    def test_overlap_zero(self):
        """Test chunking with zero overlap."""
        text = "a" * 300
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        result = chunker.chunk_simple_text(text)

        assert result['total_chunks'] == 3
        # No overlap means chunks are back-to-back
        assert result['chunks'][0]['end_char'] == 100
        assert result['chunks'][1]['start_char'] == 100
        assert result['chunks'][1]['end_char'] == 200
        assert result['chunks'][2]['start_char'] == 200

    def test_overlap_at_boundary(self):
        """Test chunking when text length is exactly at chunk boundary."""
        text = "a" * 100  # Exactly chunk_size
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        result = chunker.chunk_simple_text(text)

        # Should create exactly 1 chunk
        assert result['total_chunks'] == 1
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == 100

    def test_overlap_just_above_boundary(self):
        """Test chunking when text length is just above chunk boundary."""
        text = "a" * 101  # Just over chunk_size
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        result = chunker.chunk_simple_text(text)

        # Should create 2 chunks
        assert result['total_chunks'] == 2
        assert result['chunks'][0]['end_char'] == 100
        assert result['chunks'][1]['start_char'] == 80  # 100 - 20 overlap

    def test_chunk_ordering_and_ids(self):
        """Test that chunks are ordered correctly and have consistent IDs."""
        text = "a" * 500
        result = self.chunker.chunk_simple_text(text, document_id="order_test")

        # Verify ordering
        for i in range(len(result['chunks']) - 1):
            assert result['chunks'][i]['chunk_index'] < result['chunks'][i + 1]['chunk_index']
            assert result['chunks'][i]['start_char'] < result['chunks'][i + 1]['start_char']
            assert result['chunks'][i]['end_char'] <= result['chunks'][i + 1]['end_char']

        # Verify chunk IDs
        for i, chunk in enumerate(result['chunks']):
            expected_id = f"order_test_chunk_{i}"
            assert chunk['chunk_id'] == expected_id

        # Verify document_id consistency
        doc_ids = [chunk['document_id'] for chunk in result['chunks']]
        assert all(doc_id == "order_test" for doc_id in doc_ids)

    def test_overlap_correctness(self):
        """Test that overlap between consecutive chunks is correct."""
        text = "abcdefghij" * 20  # 200 characters
        chunker = TextChunker(chunk_size=100, chunk_overlap=30)
        result = chunker.chunk_simple_text(text)

        # Check overlap between consecutive chunks
        for i in range(len(result['chunks']) - 1):
            chunk1 = result['chunks'][i]
            chunk2 = result['chunks'][i + 1]

            # Overlap should be chunk2.start_char - chunk1.end_char (negative)
            overlap_size = chunk2['start_char'] - chunk1['end_char']
            expected_overlap = -chunker.chunk_overlap
            assert overlap_size == expected_overlap

            # Verify overlapping text matches
            overlap_text_1 = chunk1['text'][-chunker.chunk_overlap:]
            overlap_text_2 = chunk2['text'][:chunker.chunk_overlap]
            assert overlap_text_1 == overlap_text_2

    def test_no_character_loss(self):
        """Test that no characters are lost during chunking."""
        original_text = "This is a test string to verify no character loss during chunking process."
        result = self.chunker.chunk_simple_text(original_text)

        # Reconstruct text from chunks
        reconstructed = ""
        for chunk in result['chunks']:
            reconstructed += chunk['text']

        assert reconstructed == original_text

    def test_input_length_equals_chunk_size(self):
        """Test that input length exactly equal to chunk_size produces one complete chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "a" * 100  # Exactly chunk_size
        result = chunker.chunk_simple_text(text)

        # Should produce exactly one chunk
        assert result['total_chunks'] == 1
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == 100
        assert result['chunks'][0]['chunk_size'] == 100
        assert result['chunks'][0]['text'] == text  # No missing or extra characters

    def test_input_length_below_chunk_size(self):
        """Test that input length just below chunk_size produces one complete chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "a" * 99  # Just below chunk_size
        result = chunker.chunk_simple_text(text)

        # Should produce exactly one chunk
        assert result['total_chunks'] == 1
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == 99
        assert result['chunks'][0]['chunk_size'] == 99
        assert result['chunks'][0]['text'] == text  # No missing or extra characters

    def test_input_length_above_chunk_size_with_overlap(self):
        """Test that input length just above chunk_size with overlap behaves correctly."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "a" * 101  # Just above chunk_size
        result = chunker.chunk_simple_text(text)

        # Should produce 2 chunks
        assert result['total_chunks'] == 2

        # First chunk: 0-99
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == 100
        assert result['chunks'][0]['chunk_size'] == 100

        # Second chunk: 80-100 (20-character overlap)
        assert result['chunks'][1]['start_char'] == 80
        assert result['chunks'][1]['end_char'] == 101
        assert result['chunks'][1]['chunk_size'] == 21

        # Verify characters are correctly present (not duplicated)
        # First chunk should be chars 0-99
        assert result['chunks'][0]['text'] == text[0:100]
        # Second chunk should be chars 80-100
        assert result['chunks'][1]['text'] == text[80:101]

    def test_input_length_below_chunk_size_with_overlap(self):
        """Test that input length below chunk_size with overlap produces one chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=50)
        text = "a" * 60  # Below chunk_size
        result = chunker.chunk_simple_text(text)

        # Should produce exactly one chunk regardless of overlap
        assert result['total_chunks'] == 1
        assert result['chunks'][0]['start_char'] == 0
        assert result['chunks'][0]['end_char'] == 60
        assert result['chunks'][0]['chunk_size'] == 60
        assert result['chunks'][0]['text'] == text
