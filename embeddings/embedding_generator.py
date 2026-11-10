"""
Embedding Generation Module

This module provides functionality to generate text embeddings using sentence-transformers,
specifically the all-MiniLM-L6-v2 model for semantic similarity tasks.
"""

import time
from typing import List, Union, Optional
import numpy as np


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class InvalidInputError(EmbeddingError):
    """Raised when input text is invalid."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when the model fails to load."""
    pass


class EmbeddingGenerator:
    """
    Generates text embeddings using sentence-transformers.

    This class provides methods to:
    - Generate embeddings for single text strings
    - Generate embeddings for lists of text strings
    - Load the model lazily (only when first needed)
    - Validate inputs and handle errors gracefully
    - Return embeddings in consistent numpy array format
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingGenerator.

        Args:
            model_name: Name of the sentence-transformers model to use

        Raises:
            ValueError: If model_name is empty
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name
        self._model = None
        self._embedding_dimension = 384  # Known dimension for all-MiniLM-L6-v2

    def _load_model(self):
        """
        Load the sentence-transformers model lazily.

        Raises:
            ModelLoadError: If the model fails to load
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise ModelLoadError(f"Failed to load model '{self.model_name}': {str(e)}")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (384,) containing the embedding vector

        Raises:
            InvalidInputError: If text is empty or not a string
            ModelLoadError: If the model fails to load
        """
        # Validate input
        if not isinstance(text, str):
            raise InvalidInputError(f"Input must be a string, got {type(text).__name__}")

        if not text or not text.strip():
            raise InvalidInputError("Input text cannot be empty or whitespace-only")

        # Load model if not already loaded
        self._load_model()

        # Generate embedding
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

        # Validate embedding dimension
        if embedding.shape != (self._embedding_dimension,):
            raise EmbeddingError(
                f"Expected embedding dimension {self._embedding_dimension}, "
                f"got {embedding.shape}"
            )

        return embedding

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (n, 384) containing the embedding vectors

        Raises:
            InvalidInputError: If texts is empty, not a list, or contains invalid entries
            ModelLoadError: If the model fails to load
        """
        # Validate input
        if not isinstance(texts, list):
            raise InvalidInputError(f"Input must be a list, got {type(texts).__name__}")

        if len(texts) == 0:
            raise InvalidInputError("Input list cannot be empty")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise InvalidInputError(
                    f"Element {i} must be a string, got {type(text).__name__}"
                )
            if not text or not text.strip():
                raise InvalidInputError(
                    f"Element {i} cannot be empty or whitespace-only"
                )

        # Load model if not already loaded
        self._load_model()

        # Generate embeddings
        started_at = time.perf_counter()
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
        elapsed = time.perf_counter() - started_at

        # Validate embedding dimensions
        if len(embeddings.shape) != 2 or embeddings.shape[1] != self._embedding_dimension:
            raise EmbeddingError(
                f"Expected embeddings shape (n, {self._embedding_dimension}), "
                f"got {embeddings.shape}"
            )

        # Add timing as attribute for access if needed
        self._last_encode_time_seconds = elapsed
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return self._embedding_dimension

    def is_model_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None
