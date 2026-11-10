"""
Tests for Embedding Generation Module

This module contains tests for the EmbeddingGenerator class, including:
- Model loading
- Single text embedding
- Batch text embedding
- Input validation
- Error handling
- Embedding dimension validation
"""

import pytest
import numpy as np
from unittest.mock import Mock

from embeddings.embedding_generator import (
    EmbeddingGenerator,
    EmbeddingError,
    InvalidInputError,
    ModelLoadError
)


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = EmbeddingGenerator()

    def test_initialization_valid(self):
        """Test successful initialization with valid model name."""
        generator = EmbeddingGenerator(model_name="test-model")
        assert generator.model_name == "test-model"
        assert generator._model is None  # Model not loaded yet
        assert generator.get_embedding_dimension() == 384

    def test_initialization_empty_model_name(self):
        """Test initialization with empty model name."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingGenerator(model_name="")
        assert "model_name cannot be empty" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            EmbeddingGenerator(model_name="   ")
        assert "model_name cannot be empty" in str(exc_info.value)

    def test_model_not_loaded_initially(self):
        """Test that model is not loaded on initialization."""
        assert not self.generator.is_model_loaded()

    def test_encode_text_with_injected_mock(self):
        """Test encoding single text with injected mock model."""
        # Create mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)

        # Inject mock model
        self.generator._model = mock_model

        # Encode text
        text = "This is a test text"
        embedding = self.generator.encode_text(text)

        # Verify embedding
        assert embedding.shape == (384,)
        assert np.allclose(embedding, np.array([0.1] * 384))
        assert self.generator.is_model_loaded()

    def test_encode_texts_with_injected_mock(self):
        """Test encoding multiple texts with injected mock model."""
        # Create mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ])

        # Inject mock model
        self.generator._model = mock_model

        # Encode texts
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = self.generator.encode_texts(texts)

        # Verify embeddings
        assert embeddings.shape == (3, 384)
        assert np.allclose(embeddings[0], np.array([0.1] * 384))
        assert np.allclose(embeddings[1], np.array([0.2] * 384))
        assert np.allclose(embeddings[2], np.array([0.3] * 384))

    def test_encode_text_empty_string(self):
        """Test encoding empty string."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_text("")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_encode_text_whitespace_only(self):
        """Test encoding whitespace-only string."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_text("   \n\t  ")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_encode_text_invalid_type(self):
        """Test encoding non-string input."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_text(123)
        assert "Input must be a string" in str(exc_info.value)

        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_text(["text"])
        assert "Input must be a string" in str(exc_info.value)

    def test_encode_texts_empty_list(self):
        """Test encoding empty list."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_texts([])
        assert "Input list cannot be empty" in str(exc_info.value)

    def test_encode_texts_invalid_type(self):
        """Test encoding with invalid list type."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_texts("not a list")
        assert "Input must be a list" in str(exc_info.value)

    def test_encode_texts_invalid_element_type(self):
        """Test encoding list with non-string element."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_texts(["valid", 123, "also valid"])
        assert "Element 1 must be a string" in str(exc_info.value)

    def test_encode_texts_empty_element(self):
        """Test encoding list with empty element."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_texts(["valid", "", "also valid"])
        assert "Element 1 cannot be empty or whitespace-only" in str(exc_info.value)

    def test_encode_texts_whitespace_element(self):
        """Test encoding list with whitespace-only element."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.generator.encode_texts(["valid", "   ", "also valid"])
        assert "Element 1 cannot be empty or whitespace-only" in str(exc_info.value)

    def test_encode_text_wrong_dimension(self):
        """Test handling of wrong embedding dimension."""
        # Create mock model with wrong dimension
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 512)  # Wrong dimension
        self.generator._model = mock_model

        with pytest.raises(EmbeddingError) as exc_info:
            self.generator.encode_text("test text")
        assert "Expected embedding dimension 384" in str(exc_info.value)

    def test_encode_texts_wrong_dimension(self):
        """Test handling of wrong embedding dimension for batch."""
        # Create mock model with wrong dimension
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1] * 512,  # Wrong dimension
            [0.2] * 512
        ])
        self.generator._model = mock_model

        with pytest.raises(EmbeddingError) as exc_info:
            self.generator.encode_texts(["text1", "text2"])
        assert "Expected embeddings shape (n, 384)" in str(exc_info.value)

    def test_encode_failure(self):
        """Test handling of encoding failure."""
        # Create mock model that raises exception during encode
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        self.generator._model = mock_model

        with pytest.raises(EmbeddingError) as exc_info:
            self.generator.encode_text("test")
        assert "Failed to generate embedding" in str(exc_info.value)

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        dimension = self.generator.get_embedding_dimension()
        assert dimension == 384

    def test_result_format_single_text(self):
        """Test that single text embedding returns correct format."""
        # Create mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        self.generator._model = mock_model

        embedding = self.generator.encode_text("test")

        # Verify format
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64
        assert embedding.shape == (384,)

    def test_result_format_batch(self):
        """Test that batch embedding returns correct format."""
        # Create mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384
        ])
        self.generator._model = mock_model

        embeddings = self.generator.encode_texts(["text1", "text2"])

        # Verify format
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
        assert embeddings.shape == (2, 384)

    def test_model_reuse(self):
        """Test that model is reused when already loaded."""
        # Create mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        self.generator._model = mock_model

        # Model is already loaded
        assert self.generator.is_model_loaded()

        # First call
        self.generator.encode_text("text1")

        # Second call should reuse model
        self.generator.encode_text("text2")
        assert mock_model.encode.call_count == 2  # Called twice (once per encode)

    def test_consistent_embedding_same_text(self):
        """Test that the same text produces the same embedding."""
        # Create mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        self.generator._model = mock_model

        text = "Consistent test text"
        embedding1 = self.generator.encode_text(text)
        embedding2 = self.generator.encode_text(text)

        # Verify embeddings are identical
        assert np.allclose(embedding1, embedding2)
