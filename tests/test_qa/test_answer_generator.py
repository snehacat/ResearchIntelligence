"""Deterministic tests for the configured AnswerGenerator backend."""

import pytest
from unittest.mock import patch
import numpy as np
import json

from qa.answer_generator import (
    AnswerGenerator,
    AnswerGenerationError,
    InvalidAnswerError
)
from analysis.groq_backend import GroqBackend
from analysis.llm_backend import LLMGenerationResult
from retrieval.semantic_retriever import SemanticRetriever, InvalidInputError


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""

    def __init__(self):
        self.dimension = 384

    def encode_text(self, text):
        return np.random.randn(self.dimension).astype(np.float32)

    def encode_texts(self, texts):
        return np.random.randn(len(texts), self.dimension).astype(np.float32)



@pytest.fixture
def mock_embedding_generator():
    """Create a mock embedding generator."""
    return MockEmbeddingGenerator()


@pytest.fixture
def mock_retriever(mock_embedding_generator):
    """Create a mock semantic retriever with test data."""
    retriever = SemanticRetriever(mock_embedding_generator)
    test_chunks = [
        {
            'chunk_id': 'doc1_chunk_0',
            'document_id': 'doc1',
            'chunk_index': 0,
            'text': 'The study used a sample size of 500 participants.',
            'start_char': 0,
            'end_char': 50,
            'start_page': 1,
            'end_page': 1,
            'chunk_size': 50
        },
        {
            'chunk_id': 'doc1_chunk_1',
            'document_id': 'doc1',
            'chunk_index': 1,
            'text': 'Participants were aged 18-65.',
            'start_char': 50,
            'end_char': 80,
            'start_page': 1,
            'end_page': 1,
            'chunk_size': 30
        },
        {
            'chunk_id': 'doc1_chunk_2',
            'document_id': 'doc1',
            'chunk_index': 2,
            'text': 'The experiment was conducted over 6 months.',
            'start_char': 80,
            'end_char': 120,
            'start_page': 2,
            'end_page': 2,
            'chunk_size': 40
        }
    ]
    retriever.build_index(test_chunks)
    return retriever


@pytest.fixture
def answer_generator(mock_retriever):
    """Create an AnswerGenerator instance for testing."""
    return AnswerGenerator(mock_retriever, top_k=3)


class TestAnswerGeneratorInitialization:
    """Tests for AnswerGenerator initialization."""

    def test_initialization_valid(self, mock_retriever):
        """Test valid initialization."""
        generator = AnswerGenerator(mock_retriever, top_k=5)
        assert generator.model_name == GroqBackend.DEFAULT_MODEL
        assert generator.top_k == 5
        assert generator.semantic_retriever == mock_retriever

    def test_initialization_invalid_top_k(self, mock_retriever):
        """Test initialization with invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            AnswerGenerator(mock_retriever, top_k=0)

    def test_injected_backend_generates_answers_without_provider_imports(
        self, mock_retriever
    ):
        class FakeBackend:
            name = "fake"

            def generate_json(self, prompt, *, timeout_seconds, options):
                assert "Question:" in prompt
                assert timeout_seconds == 12
                assert options == {"num_predict": 768}
                return LLMGenerationResult('{"status":"ok"}')

        generator = AnswerGenerator(
            mock_retriever,
            llm_backend=FakeBackend(),
            llm_timeout_seconds=12,
        )

        assert generator._call_llm("Question: test") == '{"status":"ok"}'

    def test_missing_groq_key_is_reported_without_network(
        self, mock_retriever, monkeypatch
    ):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        generator = AnswerGenerator(mock_retriever)

        with pytest.raises(
            AnswerGenerationError, match="GROQ_API_KEY"
        ):
            generator._call_llm("test prompt")

    def test_backend_timeout_is_mapped_to_answer_error(self, mock_retriever):
        class TimeoutBackend:
            def generate_json(self, *_args, **_kwargs):
                raise TimeoutError("provider timeout")

        generator = AnswerGenerator(
            mock_retriever,
            llm_backend=TimeoutBackend(),
        )

        with pytest.raises(AnswerGenerationError, match="timed out"):
            generator._call_llm("test prompt")


class TestEvidenceRetrieval:
    """Tests for evidence retrieval."""

    def test_retrieve_evidence(self, answer_generator):
        """Test retrieving evidence for a question."""
        results = answer_generator._retrieve_evidence("What was the sample size?")
        assert len(results) == 3
        assert 'chunk' in results[0]
        assert 'similarity' in results[0]
        assert 'rank' in results[0]

    def test_retrieve_evidence_empty_question(self, answer_generator):
        """Test retrieving evidence with empty question raises error."""
        with pytest.raises(InvalidInputError, match="query cannot be empty"):
            answer_generator._retrieve_evidence("")


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_prompt(self, answer_generator):
        """Test building a prompt with evidence."""
        evidence_chunks = [
            {
                'chunk': {
                    'chunk_id': 'doc1_chunk_0',
                    'text': 'Test text',
                    'start_page': 1,
                    'end_page': 1
                },
                'similarity': 0.9
            }
        ]
        prompt = answer_generator._build_prompt("Test question", evidence_chunks)

        assert "Test question" in prompt
        assert "Test text" in prompt
        assert "JSON format" in prompt


class TestLLMResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_response(self, answer_generator):
        """Test parsing a valid LLM response."""
        response_text = json.dumps({
            "answer_text": "The sample size was 500.",
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.9,
            "evidence_indices": [1]
        })

        parsed = answer_generator._parse_llm_response(response_text)
        assert parsed['answer_text'] == "The sample size was 500."
        assert parsed['evidence_type'] == "explicit_fact"
        assert parsed['evidence_sufficiency'] == "sufficient"
        assert parsed['confidence'] == 0.9
        assert parsed['evidence_indices'] == [1]

    def test_parse_response_with_extra_text(self, answer_generator):
        """Test parsing response with text before/after JSON."""
        response_text = "Here is the answer: " + json.dumps({
            "answer_text": "Test answer",
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.8,
            "evidence_indices": [1]
        }) + " End of response."

        parsed = answer_generator._parse_llm_response(response_text)
        assert parsed['answer_text'] == "Test answer"

    def test_parse_missing_required_field(self, answer_generator):
        """Test parsing response with missing required field raises error."""
        response_text = json.dumps({
            "answer_text": "Test answer",
            "evidence_type": "explicit_fact"
            # Missing evidence_sufficiency
        })

        with pytest.raises(InvalidAnswerError, match="Missing required field"):
            answer_generator._parse_llm_response(response_text)

    def test_parse_invalid_evidence_type(self, answer_generator):
        """Test parsing response with invalid evidence_type raises error."""
        response_text = json.dumps({
            "answer_text": "Test answer",
            "evidence_type": "invalid_type",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.8,
            "evidence_indices": [1]
        })

        with pytest.raises(InvalidAnswerError, match="Invalid evidence_type"):
            answer_generator._parse_llm_response(response_text)

    def test_parse_invalid_sufficiency(self, answer_generator):
        """Test parsing response with invalid evidence_sufficiency raises error."""
        response_text = json.dumps({
            "answer_text": "Test answer",
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "invalid",
            "confidence": 0.8,
            "evidence_indices": [1]
        })

        with pytest.raises(InvalidAnswerError, match="Invalid evidence_sufficiency"):
            answer_generator._parse_llm_response(response_text)

    def test_parse_invalid_confidence(self, answer_generator):
        """Test parsing response with invalid confidence raises error."""
        response_text = json.dumps({
            "answer_text": "Test answer",
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",
            "confidence": 1.5,  # Invalid: > 1.0
            "evidence_indices": [1]
        })

        with pytest.raises(InvalidAnswerError, match="Invalid confidence"):
            answer_generator._parse_llm_response(response_text)

    def test_parse_no_json(self, answer_generator):
        """Test parsing response without JSON raises error."""
        response_text = "This is not JSON"

        with pytest.raises(InvalidAnswerError, match="does not contain valid JSON"):
            answer_generator._parse_llm_response(response_text)


class TestCitationValidation:
    """Tests for citation validation."""

    def test_validate_valid_citations(self, answer_generator):
        """Test validating valid citation indices."""
        evidence_chunks = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1', 'start_page': 1, 'end_page': 1}},
            {'chunk': {'chunk_id': 'chunk1', 'document_id': 'doc1', 'start_page': 2, 'end_page': 2}}
        ]

        citations = answer_generator._validate_citations([1, 2], evidence_chunks)
        assert len(citations) == 2
        assert citations[0]['chunk_id'] == 'chunk0'
        assert citations[1]['chunk_id'] == 'chunk1'
        assert citations[0]['start_page'] == 1
        assert citations[1]['start_page'] == 2

    def test_validate_citation_missing_page_metadata(self, answer_generator):
        """Test that citations without page metadata don't fabricate page numbers."""
        evidence_chunks = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1'}}  # No page metadata
        ]

        citations = answer_generator._validate_citations([1], evidence_chunks)
        assert len(citations) == 1
        assert citations[0]['chunk_id'] == 'chunk0'
        assert 'start_page' not in citations[0]
        assert 'end_page' not in citations[0]

    def test_validate_invalid_citation_index(self, answer_generator):
        """Test that invalid citation index raises error."""
        evidence_chunks = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1'}}
        ]

        with pytest.raises(InvalidAnswerError, match="Invalid evidence index"):
            answer_generator._validate_citations([5], evidence_chunks)  # Index out of bounds

    def test_validate_citation_index_zero(self, answer_generator):
        """Test that citation index of 0 raises error."""
        evidence_chunks = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1'}}
        ]

        with pytest.raises(InvalidAnswerError, match="Invalid evidence index"):
            answer_generator._validate_citations([0], evidence_chunks)  # Invalid 0-based index


class TestEvidenceSufficiencyAssessment:
    """Tests for evidence sufficiency assessment."""

    def test_sufficiency_no_evidence(self, answer_generator):
        """Test sufficiency assessment with no evidence."""
        sufficiency = answer_generator._assess_evidence_sufficiency("Test question", [], "sufficient")
        assert sufficiency == "insufficient"

    def test_sufficiency_low_similarity(self, answer_generator):
        """Test sufficiency assessment with low similarity chunks."""
        evidence_chunks = [
            {'similarity': 0.2},
            {'similarity': 0.25}
        ]
        sufficiency = answer_generator._assess_evidence_sufficiency("Test question", evidence_chunks, "sufficient")
        assert sufficiency == "insufficient"

    def test_sufficiency_high_similarity(self, answer_generator):
        """High similarity and an LLM label alone cannot establish support."""
        evidence_chunks = [
            {
                'chunk': {'text': 'A relevant passage.'},
                'similarity': 0.99,
            }
        ]
        sufficiency = answer_generator._assess_evidence_sufficiency(
            "Test question",
            evidence_chunks,
            "sufficient",
            "A different claim.",
        )
        assert sufficiency == "partial"

    def test_sufficiency_verbatim_answer_is_required_for_sufficient(
        self, answer_generator
    ):
        passage = "The trial included 120 participants."
        evidence_chunks = [
            {'chunk': {'text': passage}, 'similarity': 0.1}
        ]
        sufficiency = answer_generator._assess_evidence_sufficiency(
            "How many participants?",
            evidence_chunks,
            "sufficient",
            passage,
        )
        assert sufficiency == "sufficient"

    def test_sufficiency_high_similarity_partial(self, answer_generator):
        """Test sufficiency assessment with high similarity but LLM says partial."""
        evidence_chunks = [
            {'chunk': {'text': 'Relevant evidence.'}, 'similarity': 0.8}
        ]
        sufficiency = answer_generator._assess_evidence_sufficiency(
            "Test question",
            evidence_chunks,
            "partial",
            "A paraphrased answer.",
        )
        assert sufficiency == "partial"

    def test_sufficiency_default_partial(self, answer_generator):
        """Test default to partial when not clearly sufficient or insufficient."""
        evidence_chunks = [
            {'chunk': {'text': 'Relevant evidence.'}, 'similarity': 0.5}
        ]
        sufficiency = answer_generator._assess_evidence_sufficiency(
            "Test question",
            evidence_chunks,
            "sufficient",
            "A paraphrased answer.",
        )
        assert sufficiency == "partial"

    def test_sufficiency_high_similarity_but_llm_says_insufficient(self, answer_generator):
        """High similarity never overrides the model's insufficient label."""
        evidence_chunks = [
            {'chunk': {'text': 'Relevant evidence.'}, 'similarity': 0.99}
        ]
        sufficiency = answer_generator._assess_evidence_sufficiency(
            "Test question",
            evidence_chunks,
            "insufficient",
            "Any answer.",
        )
        assert sufficiency == "insufficient"


class TestGenerateAnswer:
    """End-to-end orchestration tests with mocked LLM output."""

    def test_generate_answer_fully_supported(self, answer_generator):
        mock_response = {
            "answer_text": "The study used a sample size of 500 participants.",
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.9,
            "evidence_indices": [1]
        }

        high_sim_results = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1', 'text': 'The study used a sample size of 500 participants.', 'start_page': 1, 'end_page': 1}, 'similarity': 0.8}
        ]

        with patch.object(
            answer_generator, '_retrieve_evidence', return_value=high_sim_results
        ), patch.object(
            answer_generator, '_call_llm', return_value=json.dumps(mock_response)
        ):
            response = answer_generator.generate_answer("What was the sample size?")

        assert response['answer_text'] == "The study used a sample size of 500 participants."
        assert response['evidence_type'] == "explicit_fact"
        assert response['evidence_sufficiency'] == "sufficient"
        assert response['confidence'] is None
        assert response['factual_confidence'] is None
        assert response['retrieval_similarity'] == 0.8
        assert response['support_status'] == 'verbatim_supported'
        assert len(response['evidence_chunks']) == 1
        assert len(response['citations']) == 1

    def test_generate_answer_supported_claim_control(self, answer_generator):
        passage = "The study used a sample size of 500 participants."
        evidence = [{
            'chunk': {
                'chunk_id': 'doc1_chunk_0',
                'document_id': 'doc1',
                'text': passage,
                'start_page': 1,
                'end_page': 1,
            },
            'similarity': 0.42,
        }]
        response_data = {
            "answer_text": passage,
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.99,
            "evidence_indices": [1],
        }

        with patch.object(
            answer_generator, '_retrieve_evidence', return_value=evidence
        ), patch.object(
            answer_generator,
            '_call_llm',
            return_value=json.dumps(response_data),
        ):
            response = answer_generator.generate_answer(
                "What was the sample size?"
            )

        assert response["evidence_sufficiency"] == "sufficient"
        assert response["support_status"] == "verbatim_supported"
        assert response["retrieval_similarity"] == 0.42
        assert response["factual_confidence"] is None
        assert response["confidence"] is None

    def test_generate_answer_no_evidence(self, answer_generator):
        """Test generating an answer when no evidence is available."""
        # Mock retrieval to return empty results
        with patch.object(answer_generator, '_retrieve_evidence', return_value=[]):
            response = answer_generator.generate_answer("What is the meaning of life?")

            assert response['evidence_sufficiency'] == "insufficient"
            assert response['requires_human_review'] is True
            assert "do not contain information" in response['answer_text']
            assert len(response['evidence_chunks']) == 0

    def test_generate_answer_invalid_question(self, answer_generator):
        """Test generating an answer with invalid question raises error."""
        with pytest.raises(ValueError, match="question cannot be empty"):
            answer_generator.generate_answer("")

    def test_generate_answer_provider_failure(self, answer_generator):
        with patch.object(
            answer_generator,
            '_call_llm',
            side_effect=AnswerGenerationError(
                "Groq authentication failed; check GROQ_API_KEY."
            ),
        ):
            with pytest.raises(
                AnswerGenerationError, match="check GROQ_API_KEY"
            ):
                answer_generator.generate_answer("Test question")

    def test_generate_answer_inference_failure_is_surfaced(self, answer_generator):
        with patch.object(
            answer_generator,
            '_call_llm',
            side_effect=AnswerGenerationError("LLM inference failed."),
        ):
            with pytest.raises(AnswerGenerationError, match="LLM inference failed"):
                answer_generator.generate_answer("Test question")

    def test_generate_answer_malformed_response(self, answer_generator):
        """Test handling when LLM returns malformed response."""
        with patch.object(
            answer_generator, '_call_llm', return_value="This is not JSON"
        ):
            response = answer_generator.generate_answer("Test question")
        assert response['evidence_sufficiency'] == 'insufficient'
        assert 'LLM response malformed' in response['answer_text']
        assert response['confidence'] is None

    def test_generate_answer_multiple_evidence_chunks(self, answer_generator):
        """Test generating an answer using multiple evidence chunks."""
        mock_response = {
            "answer_text": "The study used 500 participants aged 18-65 over 6 months.",
            "evidence_type": "synthesis",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.85,
            "evidence_indices": [1, 2, 3]
        }

        # Mock retrieval with high similarity
        high_sim_results = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1', 'text': 'Text 1', 'start_page': 1, 'end_page': 1}, 'similarity': 0.8},
            {'chunk': {'chunk_id': 'chunk1', 'document_id': 'doc1', 'text': 'Text 2', 'start_page': 2, 'end_page': 2}, 'similarity': 0.75},
            {'chunk': {'chunk_id': 'chunk2', 'document_id': 'doc1', 'text': 'Text 3', 'start_page': 3, 'end_page': 3}, 'similarity': 0.7}
        ]

        with patch.object(
            answer_generator, '_retrieve_evidence', return_value=high_sim_results
        ), patch.object(
            answer_generator, '_call_llm', return_value=json.dumps(mock_response)
        ):
            response = answer_generator.generate_answer(
                "Describe the study participants and duration?"
            )
        assert response['evidence_type'] == "synthesis"
        assert len(response['citations']) == 3
        assert response['evidence_sufficiency'] == "partial"
        assert response['requires_human_review'] is True

    def test_generate_answer_override_insufficient(self, answer_generator):
        """Test that independent assessment overrides LLM's sufficient claim."""
        # Mock retrieval with low similarity
        low_sim_results = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1', 'text': 'Low similarity text'}, 'similarity': 0.2}
        ]

        mock_response = {
            "answer_text": "Some answer",
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",  # LLM claims sufficient
            "confidence": 0.9,
            "evidence_indices": [1]
        }

        with patch.object(
            answer_generator, '_retrieve_evidence', return_value=low_sim_results
        ), patch.object(
            answer_generator, '_call_llm', return_value=json.dumps(mock_response)
        ):
            response = answer_generator.generate_answer("Test question")
        assert response['evidence_sufficiency'] == "partial"
        assert response['support_status'] == "unverified"
        assert response['requires_human_review'] is True

    def test_generate_answer_high_similarity_unsupported_claim(self, answer_generator):
        """A supported clause does not support an additional unsupported claim."""
        high_sim_results = [
            {'chunk': {'chunk_id': 'chunk0', 'document_id': 'doc1', 'text': 'The study used 500 participants', 'start_page': 1, 'end_page': 1}, 'similarity': 0.8}
        ]

        mock_response = {
            "answer_text": "The study used 500 participants and found no side effects.",  # "no side effects" is NOT in evidence
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": "sufficient",
            "confidence": 0.9,
            "evidence_indices": [1]
        }

        with patch.object(
            answer_generator, '_retrieve_evidence', return_value=high_sim_results
        ), patch.object(
            answer_generator, '_call_llm', return_value=json.dumps(mock_response)
        ):
            response = answer_generator.generate_answer("What were the results?")
        assert response['answer_text'] == "The study used 500 participants and found no side effects."
        assert response['evidence_sufficiency'] == "partial"
        assert response['support_status'] == "unverified"
        assert response['requires_human_review'] is True
        assert response['retrieval_similarity'] == 0.8
        assert response['factual_confidence'] is None
        assert response['confidence'] is None
        assert len(response['citations']) == 1
