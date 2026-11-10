"""
Chunking Module

This module provides text chunking functionality with metadata preservation.
"""

from chunking.text_chunker import (
    TextChunker,
    ChunkingError,
    EmptyInputError
)

__all__ = [
    'TextChunker',
    'ChunkingError',
    'EmptyInputError'
]
