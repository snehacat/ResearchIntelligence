"""
Retrieval Module

This module provides semantic retrieval functionality using FAISS.
"""

from retrieval.semantic_retriever import (
    SemanticRetriever,
    RetrievalError,
    InvalidInputError,
    IndexBuildError,
    QueryError
)

__all__ = [
    'SemanticRetriever',
    'RetrievalError',
    'InvalidInputError',
    'IndexBuildError',
    'QueryError'
]
