"""
Text Chunking Module

This module provides functionality to split extracted PDF text into manageable chunks
while preserving traceability through metadata.
"""

import time
from typing import Dict, List, Optional
import uuid


class ChunkingError(Exception):
    """Base exception for chunking errors."""
    pass


class EmptyInputError(ChunkingError):
    """Raised when input text is empty."""
    pass


class TextChunker:
    """
    Splits extracted PDF text into manageable chunks with metadata.

    This class provides methods to:
    - Split text into chunks with configurable size and overlap
    - Preserve page traceability for each chunk
    - Track chunk metadata (chunk_id, chunk_index, page numbers)
    - Handle empty input, short documents, and long text gracefully
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the TextChunker.

        Args:
            chunk_size: Maximum number of characters per chunk (default: 500)
            chunk_overlap: Number of characters to overlap between chunks (default: 50)

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap must be less than chunk_size, got {chunk_overlap} >= {chunk_size}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_extraction_result(
        self,
        extraction_result: Dict[str, any],
        document_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Chunk the text from a PDF extraction result.

        Args:
            extraction_result: Result dictionary from TextExtractor.extract_from_file()
            document_id: Optional identifier for the document (auto-generated if None)

        Returns:
            Dictionary containing:
                - 'chunks': List of chunk dictionaries with metadata
                - 'total_chunks': Total number of chunks created
                - 'document_id': Document identifier
                - 'chunk_size': Chunk size used
                - 'chunk_overlap': Overlap used
                - 'success': Boolean indicating if chunking succeeded

        Raises:
            EmptyInputError: If extraction result contains no text
        """
        # Validate input
        if not extraction_result.get('pages') or len(extraction_result['pages']) == 0:
            raise EmptyInputError("Extraction result contains no pages with text")

        started_at = time.perf_counter()

        # Generate document ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Concatenate all page text with page boundary markers
        # We use a special marker to track where page boundaries are
        full_text = ""
        page_boundaries = []  # List of (character_position, page_number)
        current_pos = 0

        for page_data in extraction_result['pages']:
            page_text = page_data['text']
            page_number = page_data['page_number']

            # Record page boundary before adding text
            page_boundaries.append((current_pos, page_number))

            # Add page text (preserve original whitespace)
            full_text += page_text
            current_pos += len(page_text)

        # Check if we have any text to chunk
        if not full_text.strip():
            raise EmptyInputError("Extracted text is empty or contains only whitespace")

        # Create chunks
        chunks = []
        chunk_index = 0
        pos = 0

        while pos < len(full_text):
            # Calculate chunk end position
            end_pos = min(pos + self.chunk_size, len(full_text))

            # Extract chunk text
            chunk_text = full_text[pos:end_pos]

            # Determine which pages this chunk spans
            start_page = self._get_page_at_position(pos, page_boundaries)
            end_page = self._get_page_at_position(end_pos - 1, page_boundaries)

            # Create chunk metadata
            chunk = {
                'chunk_id': f"{document_id}_chunk_{chunk_index}",
                'document_id': document_id,
                'chunk_index': chunk_index,
                'text': chunk_text,
                'start_char': pos,
                'end_char': end_pos,
                'start_page': start_page,
                'end_page': end_page,
                'chunk_size': len(chunk_text)
            }

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk (with overlap)
            if end_pos >= len(full_text):
                break
            pos = end_pos - self.chunk_overlap

        elapsed = time.perf_counter() - started_at
        return {
            'chunks': chunks,
            'total_chunks': len(chunks),
            'document_id': document_id,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'success': True,
            'chunking_time_seconds': elapsed
        }

    def _get_page_at_position(self, position: int, page_boundaries: List[tuple]) -> int:
        """
        Get the page number at a given character position.

        Args:
            position: Character position in the concatenated text
            page_boundaries: List of (character_position, page_number) tuples

        Returns:
            Page number (1-based)
        """
        # Find the last page boundary that is <= position
        page_number = 1
        for boundary_pos, page_num in page_boundaries:
            if boundary_pos <= position:
                page_number = page_num
            else:
                break
        return page_number

    def chunk_simple_text(
        self,
        text: str,
        document_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Chunk a simple text string (not from PDF extraction).

        This is a convenience method for chunking arbitrary text without page information.

        Args:
            text: Text to chunk
            document_id: Optional identifier for the document (auto-generated if None)

        Returns:
            Dictionary containing chunks (similar to chunk_extraction_result, but without page metadata)
        """
        if not text or not text.strip():
            raise EmptyInputError("Input text is empty or contains only whitespace")

        started_at = time.perf_counter()

        # Generate document ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Create chunks
        chunks = []
        chunk_index = 0
        pos = 0

        while pos < len(text):
            end_pos = min(pos + self.chunk_size, len(text))
            chunk_text = text[pos:end_pos]

            chunk = {
                'chunk_id': f"{document_id}_chunk_{chunk_index}",
                'document_id': document_id,
                'chunk_index': chunk_index,
                'text': chunk_text,
                'start_char': pos,
                'end_char': end_pos,
                'chunk_size': len(chunk_text)
            }

            chunks.append(chunk)
            chunk_index += 1

            if end_pos >= len(text):
                break
            pos = end_pos - self.chunk_overlap

        elapsed = time.perf_counter() - started_at
        return {
            'chunks': chunks,
            'total_chunks': len(chunks),
            'document_id': document_id,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'success': True,
            'chunking_time_seconds': elapsed
        }
