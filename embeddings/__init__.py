"""
Embeddings Module

This module provides text embedding functionality using sentence-transformers.
"""

from embeddings.embedding_generator import (
    EmbeddingGenerator,
    EmbeddingError,
    InvalidInputError,
    ModelLoadError
)

__all__ = [
    'EmbeddingGenerator',
    'EmbeddingError',
    'InvalidInputError',
    'ModelLoadError'
]
