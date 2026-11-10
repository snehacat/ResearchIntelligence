"""
Semantic Retrieval Module

This module provides functionality to build a FAISS index for semantic search
and retrieve relevant chunks based on natural language queries.
"""

import time
from typing import List, Dict, Optional, Union
import numpy as np
import faiss


class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass


class InvalidInputError(RetrievalError):
    """Raised when input is invalid."""
    pass


class IndexBuildError(RetrievalError):
    """Raised when index construction fails."""
    pass


class QueryError(RetrievalError):
    """Raised when query processing fails."""
    pass


class SemanticRetriever:
    """
    Builds a FAISS index for semantic search and retrieves relevant chunks.

    This class provides methods to:
    - Build an in-memory FAISS index from chunk embeddings
    - Search for top-k most similar chunks to a query
    - Preserve all chunk metadata in results
    - Use normalized vectors with inner-product for cosine similarity
    - Handle empty collections, invalid queries, and edge cases
    """

    def __init__(self, embedding_generator, embedding_dimension: int = 384):
        """
        Initialize the SemanticRetriever.

        Args:
            embedding_generator: EmbeddingGenerator instance for generating embeddings
            embedding_dimension: Dimension of embedding vectors (default: 384 for MiniLM-L6-v2)

        Raises:
            ValueError: If embedding_dimension is not positive
        """
        if embedding_dimension <= 0:
            raise ValueError(f"embedding_dimension must be positive, got {embedding_dimension}")

        self.embedding_generator = embedding_generator
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.chunks = []  # Stores chunk records in index order
        self.is_built = False

    def build_index(self, chunks: List[Dict[str, any]], rebuild: bool = False):
        """
        Build a FAISS index from chunk embeddings.

        Args:
            chunks: List of chunk dictionaries from TextChunker
            rebuild: If True, rebuild index even if already built

        Raises:
            InvalidInputError: If chunks is empty or invalid
            IndexBuildError: If index construction fails
        """
        # Validate input
        if not isinstance(chunks, list):
            raise InvalidInputError(f"chunks must be a list, got {type(chunks).__name__}")

        if len(chunks) == 0:
            raise InvalidInputError("chunks list cannot be empty")

        # Check if index already exists
        if self.is_built and not rebuild:
            raise IndexBuildError("Index already built. Set rebuild=True to rebuild.")

        # Validate chunk structure
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise InvalidInputError(f"Chunk {i} must be a dictionary, got {type(chunk).__name__}")
            if 'text' not in chunk:
                raise InvalidInputError(f"Chunk {i} missing required field 'text'")

        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        started_at = time.perf_counter()
        try:
            embeddings = self.embedding_generator.encode_texts(chunk_texts)
        except Exception as e:
            raise IndexBuildError(f"Failed to generate embeddings: {str(e)}")
        embed_time = time.perf_counter() - started_at

        # Validate embeddings
        if embeddings.shape[1] != self.embedding_dimension:
            raise IndexBuildError(
                f"Expected embedding dimension {self.embedding_dimension}, "
                f"got {embeddings.shape[1]}"
            )

        # Check for non-finite values
        if not np.all(np.isfinite(embeddings)):
            raise IndexBuildError("Embeddings contain non-finite values (NaN or Inf)")

        # Normalize embeddings for inner-product = cosine similarity
        embeddings = self._normalize_vectors(embeddings)

        # Build FAISS index (FlatL2 for exact search, but we use normalized vectors + inner product)
        # Using Inner Product index for cosine similarity with normalized vectors
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(embeddings.astype(np.float32))

        # Store chunks in index order
        self.chunks = chunks.copy()
        self.is_built = True
        self._last_build_time_seconds = embed_time

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for top-k most similar chunks to a query.

        Args:
            query: Query text string
            top_k: Number of results to return

        Returns:
            List of dictionaries containing:
                - chunk: Original chunk dictionary with all metadata
                - similarity: Similarity score (cosine similarity)
                - rank: Result rank (1-based)

        Raises:
            InvalidInputError: If index not built, query invalid, or top_k invalid
            QueryError: If query processing fails
        """
        # Validate index is built
        if not self.is_built:
            raise InvalidInputError("Index not built. Call build_index() first.")

        # Validate query
        if not isinstance(query, str):
            raise InvalidInputError(f"query must be a string, got {type(query).__name__}")

        if not query or not query.strip():
            raise InvalidInputError("query cannot be empty or whitespace-only")

        # Validate top_k
        if not isinstance(top_k, int):
            raise InvalidInputError(f"top_k must be an integer, got {type(top_k).__name__}")

        if top_k <= 0:
            raise InvalidInputError(f"top_k must be positive, got {top_k}")

        # Adjust top_k if larger than number of chunks
        actual_top_k = min(top_k, len(self.chunks))

        # Generate query embedding
        started_at = time.perf_counter()
        try:
            query_embedding = self.embedding_generator.encode_text(query)
        except Exception as e:
            raise QueryError(f"Failed to generate query embedding: {str(e)}")
        query_embed_time = time.perf_counter() - started_at

        # Validate query embedding dimension
        if query_embedding.shape[0] != self.embedding_dimension:
            raise QueryError(
                f"Expected query embedding dimension {self.embedding_dimension}, "
                f"got {query_embedding.shape[0]}"
            )

        # Check for non-finite values
        if not np.all(np.isfinite(query_embedding)):
            raise QueryError("Query embedding contains non-finite values (NaN or Inf)")

        # Normalize query embedding
        query_embedding = self._normalize_vectors(query_embedding.reshape(1, -1))

        # Search index
        started_at = time.perf_counter()
        try:
            similarities, indices = self.index.search(query_embedding.astype(np.float32), actual_top_k)
        except Exception as e:
            raise QueryError(f"Failed to search index: {str(e)}")
        search_time = time.perf_counter() - started_at

        # Build results with metadata
        results = []
        for rank in range(actual_top_k):
            idx = indices[0][rank]
            similarity = float(similarities[0][rank])
            chunk = self.chunks[idx]

            result = {
                'chunk': chunk,
                'similarity': similarity,
                'rank': rank + 1  # 1-based ranking
            }
            results.append(result)

        # Store search timing
        self._last_search_time_seconds = search_time
        return results

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.

        Args:
            vectors: numpy array of shape (n, d) or (d,)

        Returns:
            Normalized vectors of same shape
        """
        # Handle single vector case
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # Compute L2 norms
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Avoid division by zero
        norms[norms == 0] = 1.0

        # Normalize
        normalized = vectors / norms

        return normalized

    def get_index_size(self) -> int:
        """
        Get the number of chunks in the index.

        Returns:
            Number of indexed chunks
        """
        return len(self.chunks) if self.is_built else 0

    def is_index_built(self) -> bool:
        """
        Check if the index is built.

        Returns:
            True if index is built, False otherwise
        """
        return self.is_built
