"""
Tests for Semantic Retrieval Module

This module contains tests for the SemanticRetriever class, including:
- Index construction
- Empty index behavior
- Query embedding and similarity ranking
- Top-k results and ordering
- Chunk metadata preservation
- Error handling
- Cosine similarity behavior
"""

import pytest
import numpy as np
from unittest.mock import Mock

from retrieval.semantic_retriever import (
    SemanticRetriever,
    RetrievalError,
    InvalidInputError,
    IndexBuildError,
    QueryError
)


class TestSemanticRetriever:
    """Test cases for SemanticRetriever class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock embedding generator
        self.mock_generator = Mock()
        self.retriever = SemanticRetriever(self.mock_generator, embedding_dimension=384)

    def test_initialization_valid(self):
        """Test successful initialization."""
        retriever = SemanticRetriever(self.mock_generator, embedding_dimension=384)
        assert retriever.embedding_dimension == 384
        assert retriever.is_index_built() is False
        assert retriever.get_index_size() == 0

    def test_initialization_invalid_dimension(self):
        """Test initialization with invalid dimension."""
        with pytest.raises(ValueError) as exc_info:
            SemanticRetriever(self.mock_generator, embedding_dimension=0)
        assert "embedding_dimension must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            SemanticRetriever(self.mock_generator, embedding_dimension=-100)
        assert "embedding_dimension must be positive" in str(exc_info.value)

    def test_build_index_with_chunks(self):
        """Test building index with valid chunks."""
        # Setup mock embeddings
        chunks = [
            {'text': 'First chunk', 'chunk_id': 'c1', 'document_id': 'doc1'},
            {'text': 'Second chunk', 'chunk_id': 'c2', 'document_id': 'doc1'},
            {'text': 'Third chunk', 'chunk_id': 'c3', 'document_id': 'doc1'}
        ]
        embeddings = np.random.randn(3, 384)
        self.mock_generator.encode_texts.return_value = embeddings

        # Build index
        self.retriever.build_index(chunks)

        # Verify
        assert self.retriever.is_index_built() is True
        assert self.retriever.get_index_size() == 3
        assert len(self.retriever.chunks) == 3
        assert self.mock_generator.encode_texts.called

    def test_build_index_empty_chunks(self):
        """Test building index with empty chunk list."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.build_index([])
        assert "chunks list cannot be empty" in str(exc_info.value)

    def test_build_index_invalid_type(self):
        """Test building index with non-list input."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.build_index("not a list")
        assert "chunks must be a list" in str(exc_info.value)

    def test_build_index_invalid_chunk_type(self):
        """Test building index with non-dict chunk."""
        chunks = ['not a dict', 'also not a dict']
        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.build_index(chunks)
        assert "Chunk 0 must be a dictionary" in str(exc_info.value)

    def test_build_index_missing_text_field(self):
        """Test building index with chunk missing text field."""
        chunks = [{'chunk_id': 'c1'}, {'text': 'valid', 'chunk_id': 'c2'}]
        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.build_index(chunks)
        assert "Chunk 0 missing required field 'text'" in str(exc_info.value)

    def test_build_index_wrong_dimension(self):
        """Test building index with wrong embedding dimension."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 512)  # Wrong dimension
        self.mock_generator.encode_texts.return_value = embeddings

        with pytest.raises(IndexBuildError) as exc_info:
            self.retriever.build_index(chunks)
        assert "Expected embedding dimension 384" in str(exc_info.value)

    def test_build_index_non_finite_embeddings(self):
        """Test building index with non-finite embeddings."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        embeddings[0, 0] = np.nan  # Non-finite value
        self.mock_generator.encode_texts.return_value = embeddings

        with pytest.raises(IndexBuildError) as exc_info:
            self.retriever.build_index(chunks)
        assert "Embeddings contain non-finite values" in str(exc_info.value)

    def test_build_index_already_built(self):
        """Test that rebuild is required to rebuild index."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings

        # Build first time
        self.retriever.build_index(chunks)
        assert self.retriever.is_index_built() is True

        # Try to build again without rebuild=True
        with pytest.raises(IndexBuildError) as exc_info:
            self.retriever.build_index(chunks)
        assert "Index already built" in str(exc_info.value)

    def test_build_index_rebuild(self):
        """Test rebuilding index with rebuild=True."""
        chunks1 = [{'text': 'test1', 'chunk_id': 'c1'}]
        chunks2 = [{'text': 'test2', 'chunk_id': 'c2'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings

        # Build first time
        self.retriever.build_index(chunks1)
        assert self.retriever.get_index_size() == 1

        # Rebuild
        self.retriever.build_index(chunks2, rebuild=True)
        assert self.retriever.get_index_size() == 1
        assert self.retriever.chunks[0]['chunk_id'] == 'c2'

    def test_search_query(self):
        """Test searching with a query."""
        # Build index
        chunks = [
            {'text': 'First chunk', 'chunk_id': 'c1', 'document_id': 'doc1', 'start_page': 1, 'end_page': 1},
            {'text': 'Second chunk', 'chunk_id': 'c2', 'document_id': 'doc1', 'start_page': 2, 'end_page': 2},
            {'text': 'Third chunk', 'chunk_id': 'c3', 'document_id': 'doc1', 'start_page': 3, 'end_page': 3}
        ]
        embeddings = np.random.randn(3, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        # Setup query embedding
        query_embedding = np.random.randn(384)
        self.mock_generator.encode_text.return_value = query_embedding

        # Search
        results = self.retriever.search("test query", top_k=2)

        # Verify
        assert len(results) == 2
        assert results[0]['rank'] == 1
        assert results[1]['rank'] == 2
        assert 'chunk' in results[0]
        assert 'similarity' in results[0]
        assert results[0]['chunk']['chunk_id'] in ['c1', 'c2', 'c3']

    def test_search_index_not_built(self):
        """Test searching when index not built."""
        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search("test query")
        assert "Index not built" in str(exc_info.value)

    def test_search_empty_query(self):
        """Test searching with empty query."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search("")
        assert "query cannot be empty" in str(exc_info.value)

    def test_search_whitespace_query(self):
        """Test searching with whitespace-only query."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search("   ")
        assert "query cannot be empty" in str(exc_info.value)

    def test_search_invalid_query_type(self):
        """Test searching with non-string query."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search(123)
        assert "query must be a string" in str(exc_info.value)

    def test_search_invalid_top_k_type(self):
        """Test searching with invalid top_k type."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search("test", top_k="5")
        assert "top_k must be an integer" in str(exc_info.value)

    def test_search_invalid_top_k_value(self):
        """Test searching with invalid top_k value."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search("test", top_k=0)
        assert "top_k must be positive" in str(exc_info.value)

        with pytest.raises(InvalidInputError) as exc_info:
            self.retriever.search("test", top_k=-5)
        assert "top_k must be positive" in str(exc_info.value)

    def test_search_top_k_larger_than_index(self):
        """Test searching with top_k larger than indexed chunks."""
        chunks = [
            {'text': 'chunk1', 'chunk_id': 'c1'},
            {'text': 'chunk2', 'chunk_id': 'c2'}
        ]
        embeddings = np.random.randn(2, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        query_embedding = np.random.randn(384)
        self.mock_generator.encode_text.return_value = query_embedding

        # Request more results than available
        results = self.retriever.search("test", top_k=10)

        # Should return only available chunks
        assert len(results) == 2

    def test_search_query_embedding_dimension_mismatch(self):
        """Test search with wrong query embedding dimension."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        query_embedding = np.random.randn(512)  # Wrong dimension
        self.mock_generator.encode_text.return_value = query_embedding

        with pytest.raises(QueryError) as exc_info:
            self.retriever.search("test")
        assert "Expected query embedding dimension 384" in str(exc_info.value)

    def test_search_query_non_finite(self):
        """Test search with non-finite query embedding."""
        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        query_embedding = np.random.randn(384)
        query_embedding[0] = np.nan
        self.mock_generator.encode_text.return_value = query_embedding

        with pytest.raises(QueryError) as exc_info:
            self.retriever.search("test")
        assert "Query embedding contains non-finite values" in str(exc_info.value)

    def test_chunk_metadata_preservation(self):
        """Test that chunk metadata is preserved in results."""
        chunks = [
            {
                'text': 'Test chunk',
                'chunk_id': 'c1',
                'document_id': 'doc1',
                'chunk_index': 0,
                'start_char': 0,
                'end_char': 100,
                'start_page': 1,
                'end_page': 1,
                'chunk_size': 100
            }
        ]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        query_embedding = np.random.randn(384)
        self.mock_generator.encode_text.return_value = query_embedding

        results = self.retriever.search("test", top_k=1)

        # Verify all metadata preserved
        chunk = results[0]['chunk']
        assert chunk['chunk_id'] == 'c1'
        assert chunk['document_id'] == 'doc1'
        assert chunk['chunk_index'] == 0
        assert chunk['start_char'] == 0
        assert chunk['end_char'] == 100
        assert chunk['start_page'] == 1
        assert chunk['end_page'] == 1
        assert chunk['chunk_size'] == 100
        assert chunk['text'] == 'Test chunk'

    def test_similarity_ranking(self):
        """Test that results are ranked by similarity."""
        chunks = [
            {'text': 'chunk1', 'chunk_id': 'c1'},
            {'text': 'chunk2', 'chunk_id': 'c2'},
            {'text': 'chunk3', 'chunk_id': 'c3'}
        ]
        # Create embeddings with known similarity pattern
        embeddings = np.array([
            [1.0] * 384,  # chunk1
            [0.5] * 384,  # chunk2
            [0.1] * 384   # chunk3
        ])
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        # Query similar to chunk1
        query_embedding = np.array([0.9] * 384)
        self.mock_generator.encode_text.return_value = query_embedding

        results = self.retriever.search("test", top_k=3)

        # Verify ranking (highest similarity first)
        assert results[0]['similarity'] >= results[1]['similarity']
        assert results[1]['similarity'] >= results[2]['similarity']
        assert results[0]['rank'] == 1
        assert results[1]['rank'] == 2
        assert results[2]['rank'] == 3

    def test_faiiss_index_position_mapping(self):
        """Test that FAISS index positions map correctly to chunk records."""
        chunks = [
            {'text': 'chunk0', 'chunk_id': 'c0'},
            {'text': 'chunk1', 'chunk_id': 'c1'},
            {'text': 'chunk2', 'chunk_id': 'c2'}
        ]
        embeddings = np.random.randn(3, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        query_embedding = np.random.randn(384)
        self.mock_generator.encode_text.return_value = query_embedding

        results = self.retriever.search("test", top_k=3)

        # Verify that result chunks match original chunks
        retrieved_ids = [r['chunk']['chunk_id'] for r in results]
        original_ids = [c['chunk_id'] for c in chunks]
        assert set(retrieved_ids) == set(original_ids)

    def test_get_index_size(self):
        """Test getting index size."""
        assert self.retriever.get_index_size() == 0

        chunks = [
            {'text': 'chunk1', 'chunk_id': 'c1'},
            {'text': 'chunk2', 'chunk_id': 'c2'}
        ]
        embeddings = np.random.randn(2, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        assert self.retriever.get_index_size() == 2

    def test_is_index_built(self):
        """Test checking if index is built."""
        assert self.retriever.is_index_built() is False

        chunks = [{'text': 'test', 'chunk_id': 'c1'}]
        embeddings = np.random.randn(1, 384)
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        assert self.retriever.is_index_built() is True

    def test_normalization_cosine_similarity(self):
        """Test that normalized vectors produce cosine similarity."""
        # Create vectors with known angle
        vec1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        vec2 = np.array([0.707, 0.707, 0.0] + [0.0] * 381)  # 45 degrees

        chunks = [
            {'text': 'chunk1', 'chunk_id': 'c1'},
            {'text': 'chunk2', 'chunk_id': 'c2'}
        ]
        embeddings = np.array([vec1, vec2])
        self.mock_generator.encode_texts.return_value = embeddings
        self.retriever.build_index(chunks)

        # Query parallel to vec1
        query_embedding = vec1.copy()
        self.mock_generator.encode_text.return_value = query_embedding

        results = self.retriever.search("test", top_k=2)

        # vec1 should have higher similarity (closer to query)
        assert results[0]['similarity'] > results[1]['similarity']
