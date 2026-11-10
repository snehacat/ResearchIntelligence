"""Evidence-based question answering through the configured hosted LLM."""

from typing import Dict, List, Optional
import json
import re

from analysis.backend_factory import (
    configured_backend_name,
    configured_model_name,
    create_llm_backend,
)
from analysis.groq_backend import GroqBackendError
from analysis.llm_backend import LLMBackend, LLMGenerationResult


class AnswerGenerationError(Exception):
    """Base exception for answer generation errors."""
    pass


class InvalidAnswerError(AnswerGenerationError):
    """Raised when the LLM output is malformed or missing required fields."""
    pass


class AnswerGenerator:
    """
    Generates evidence-based answers using the configured LLM backend.

    This class provides methods to:
    - Retrieve relevant chunks using semantic search
    - Generate answers grounded in retrieved evidence
    - Validate citations against actual chunk metadata
    - Assess evidence sufficiency with explicit rules
    - Surface provider failures and reject malformed output
    """

    def __init__(
        self,
        semantic_retriever,
        model_name: Optional[str] = None,
        top_k: int = 5,
        llm_backend: Optional[LLMBackend] = None,
        backend_name: Optional[str] = None,
        llm_timeout_seconds: float = 90.0,
    ):
        """
        Initialize the AnswerGenerator.

        Args:
            semantic_retriever: SemanticRetriever instance for retrieving relevant chunks
            model_name: Optional model override; otherwise uses configuration
            top_k: Number of chunks to retrieve as evidence (default: 5)
            llm_backend: Optional injectable structured-generation backend
            backend_name: Optional provider override
            llm_timeout_seconds: Maximum model-request duration

        Raises:
            ValueError: If top_k or the timeout is not positive
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if (
            not isinstance(llm_timeout_seconds, (int, float))
            or isinstance(llm_timeout_seconds, bool)
            or llm_timeout_seconds <= 0
        ):
            raise ValueError("llm_timeout_seconds must be positive")

        self.semantic_retriever = semantic_retriever
        self.backend_name = backend_name or configured_backend_name()
        selected_model = model_name or configured_model_name(self.backend_name)
        if not selected_model:
            raise ValueError(
                f"Set RESEARCHINTELLIGENCE_LLM_MODEL when using "
                f"{self.backend_name}."
            )
        self.model_name = selected_model
        self.top_k = top_k
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.llm_backend = llm_backend or create_llm_backend(
            self.backend_name, selected_model
        )

    def _retrieve_evidence(self, question: str) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks for a question using semantic search.

        Args:
            question: User's question

        Returns:
            List of retrieved chunks with similarity scores

        Raises:
            InvalidInputError: If question is invalid or index not built
        """
        # Use the semantic retriever to search
        results = self.semantic_retriever.search(question, top_k=self.top_k)
        return results

    def _build_prompt(self, question: str, evidence_chunks: List[Dict[str, any]]) -> str:
        """
        Build a prompt for the LLM with the question and evidence.

        Args:
            question: User's question
            evidence_chunks: List of retrieved chunks with metadata

        Returns:
            Prompt string for the LLM
        """
        # Format evidence chunks for the prompt
        evidence_text = ""
        for i, result in enumerate(evidence_chunks, 1):
            chunk = result['chunk']
            similarity = result['similarity']
            evidence_text += f"\n[Evidence {i} - Similarity: {similarity:.3f}]\n"
            evidence_text += f"Chunk ID: {chunk.get('chunk_id', 'unknown')}\n"
            if 'start_page' in chunk and 'end_page' in chunk:
                evidence_text += f"Pages: {chunk['start_page']}-{chunk['end_page']}\n"
            evidence_text += f"Text: {chunk['text']}\n"

        prompt = f"""You are a research assistant answering questions based ONLY on the provided evidence. Follow these rules strictly:

1. Use ONLY the evidence provided below to answer the question.
2. If the evidence does not contain the answer, state that the information is not available.
3. Distinguish between:
   - EXPLICIT FACT: Information directly stated in the evidence
   - INTERPRETATION: A reasonable inference from stated facts
   - SYNTHESIS: Information combined from multiple evidence chunks
4. Provide your answer in the following JSON format:
   {{
     "answer_text": "your answer here",
     "evidence_type": "explicit_fact|interpretation|synthesis",
     "evidence_sufficiency": "sufficient|partial|insufficient",
     "evidence_indices": [list of evidence chunk numbers used, e.g., [1, 2]]
   }}

Question: {question}

{evidence_text}

Answer in JSON format:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the configured backend and return its generated JSON text."""
        try:
            result = self.llm_backend.generate_json(
                prompt,
                timeout_seconds=self.llm_timeout_seconds,
                options={"num_predict": 768},
            )
        except TimeoutError:
            raise AnswerGenerationError("LLM request timed out.") from None
        except GroqBackendError as error:
            raise AnswerGenerationError(str(error)) from None
        except Exception:
            raise AnswerGenerationError("LLM inference failed.") from None
        if not isinstance(result, LLMGenerationResult):
            raise AnswerGenerationError(
                "LLM backend returned an invalid generation result."
            )
        return result.text

    def _parse_llm_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse the LLM response into a structured dictionary.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed response dictionary

        Raises:
            InvalidAnswerError: If response is malformed or missing required fields
        """
        try:
            # Try to extract JSON from the response
            # The LLM might include text before/after the JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise InvalidAnswerError("Response does not contain valid JSON")

            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)

            # Validate required fields
            required_fields = ['answer_text', 'evidence_type', 'evidence_sufficiency']
            for field in required_fields:
                if field not in parsed:
                    raise InvalidAnswerError(f"Missing required field: {field}")
            if not isinstance(parsed["answer_text"], str):
                raise InvalidAnswerError("answer_text must be a string")

            # Validate field values
            if parsed['evidence_type'] not in ['explicit_fact', 'interpretation', 'synthesis']:
                raise InvalidAnswerError(
                    f"Invalid evidence_type: {parsed['evidence_type']}. "
                    "Must be one of: explicit_fact, interpretation, synthesis"
                )

            if parsed['evidence_sufficiency'] not in ['sufficient', 'partial', 'insufficient']:
                raise InvalidAnswerError(
                    f"Invalid evidence_sufficiency: {parsed['evidence_sufficiency']}. "
                    "Must be one of: sufficient, partial, insufficient"
                )

            # Validate confidence if present
            if 'confidence' in parsed:
                confidence = parsed['confidence']
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    raise InvalidAnswerError(
                        f"Invalid confidence: {confidence}. Must be between 0.0 and 1.0"
                    )

            return parsed

        except json.JSONDecodeError as e:
            raise InvalidAnswerError(f"Failed to parse JSON response: {str(e)}")

    def _validate_citations(
        self,
        evidence_indices: List[int],
        evidence_chunks: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Validate citation references against actual retrieved chunks.

        Args:
            evidence_indices: List of 1-based indices from LLM response
            evidence_chunks: List of retrieved chunks with metadata

        Returns:
            List of validated citation dictionaries

        Raises:
            InvalidAnswerError: If citation indices are invalid
        """
        citations = []

        for idx in evidence_indices:
            # Convert to 0-based index
            chunk_idx = idx - 1

            # Validate index is within bounds
            if chunk_idx < 0 or chunk_idx >= len(evidence_chunks):
                raise InvalidAnswerError(
                    f"Invalid evidence index: {idx}. "
                    f"Must be between 1 and {len(evidence_chunks)}"
                )

            chunk = evidence_chunks[chunk_idx]['chunk']

            # Build citation from actual chunk metadata
            citation = {
                'chunk_id': chunk.get('chunk_id'),
                'document_id': chunk.get('document_id')
            }

            # Include page numbers only if available
            if 'start_page' in chunk and 'end_page' in chunk:
                citation['start_page'] = chunk['start_page']
                citation['end_page'] = chunk['end_page']

            citations.append(citation)

        return citations

    def _assess_evidence_sufficiency(
        self,
        question: str,
        evidence_chunks: List[Dict[str, any]],
        llm_sufficiency: str,
        answer_text: Optional[str] = None,
    ) -> str:
        """
        Assess whether the complete answer is directly present in its evidence.

        Args:
            question: User's question
            evidence_chunks: List of retrieved chunks
            llm_sufficiency: LLM's self-reported sufficiency, used only to
                lower the result
            answer_text: Answer text returned by the model

        Returns:
            Evidence sufficiency: "sufficient", "partial", or "insufficient"
        """
        if not evidence_chunks or llm_sufficiency == "insufficient":
            return "insufficient"

        passages = [
            result.get("chunk", {}).get("text", "")
            for result in evidence_chunks
            if isinstance(result.get("chunk"), dict)
            and isinstance(result["chunk"].get("text"), str)
            and result["chunk"]["text"].strip()
        ]
        if not passages:
            return "insufficient"

        if answer_text and self._answer_is_verbatim_in_evidence(
            answer_text, passages
        ):
            return (
                "sufficient"
                if llm_sufficiency == "sufficient"
                else "partial"
            )
        return "partial"

    @staticmethod
    def _answer_is_verbatim_in_evidence(
        answer_text: str,
        passages: List[str],
    ) -> bool:
        """Return true only when the complete answer occurs in the evidence."""
        normalized_answer = " ".join(
            re.findall(r"\w+", answer_text.casefold())
        )
        if not normalized_answer:
            return False
        return any(
            normalized_answer
            in " ".join(re.findall(r"\w+", passage.casefold()))
            for passage in passages
        )

    @staticmethod
    def _retrieval_similarity(
        evidence_results: List[Dict[str, any]]
    ) -> Optional[float]:
        scores = [
            result["similarity"]
            for result in evidence_results
            if isinstance(result.get("similarity"), (int, float))
            and not isinstance(result.get("similarity"), bool)
        ]
        return sum(scores) / len(scores) if scores else None

    def generate_answer(self, question: str) -> Dict[str, any]:
        """
        Generate an evidence-based answer to a question.

        Args:
            question: User's question

        Returns:
            Dictionary containing:
                - answer_text: The answer text
                - evidence_chunks: List of evidence chunks with metadata
                - citations: List of validated citation dictionaries
                - evidence_type: One of "explicit_fact", "interpretation", "synthesis"
                - evidence_sufficiency: One of "sufficient", "partial", "insufficient"
                - retrieval_similarity: average retrieval score, not correctness
                - factual_confidence: None (not calibrated by this system)

        Raises:
            InvalidInputError: If question is invalid
            AnswerGenerationError: If answer generation fails
        """
        # Validate question
        if not isinstance(question, str):
            raise ValueError(f"question must be a string, got {type(question).__name__}")

        if not question or not question.strip():
            raise ValueError("question cannot be empty or whitespace-only")

        # Retrieve evidence chunks
        evidence_results = self._retrieve_evidence(question)
        evidence_chunks = [result['chunk'] for result in evidence_results]

        # If no evidence chunks, return insufficient response without calling LLM
        if len(evidence_results) == 0:
            return {
                'answer_text': "The provided documents do not contain information to answer this question.",
                'evidence_chunks': [],
                'citations': [],
                'evidence_type': 'explicit_fact',
                'evidence_sufficiency': 'insufficient',
                'retrieval_similarity': None,
                'factual_confidence': None,
                'confidence': None,
                'support_status': 'insufficient',
                'requires_human_review': True,
            }

        # Build prompt
        prompt = self._build_prompt(question, evidence_results)

        llm_response = self._call_llm(prompt)

        # Parse LLM response
        try:
            parsed_response = self._parse_llm_response(llm_response)
        except InvalidAnswerError:
            # If LLM response is malformed, we can't extract citations
            # Return insufficient response with empty citations
            return {
                'answer_text': "The provided documents do not contain sufficient information to answer this question (LLM response malformed).",
                'evidence_chunks': evidence_chunks,
                'citations': [],
                'evidence_type': 'explicit_fact',
                'evidence_sufficiency': 'insufficient',
                'retrieval_similarity': self._retrieval_similarity(
                    evidence_results
                ),
                'factual_confidence': None,
                'confidence': None,
                'support_status': 'unverified',
                'requires_human_review': True,
            }

        # Validate citations
        evidence_indices = parsed_response.get('evidence_indices', list(range(1, len(evidence_chunks) + 1)))
        citations = self._validate_citations(evidence_indices, evidence_results)
        cited_results = [
            evidence_results[index - 1] for index in evidence_indices
        ]

        # Independent evidence sufficiency assessment
        final_sufficiency = self._assess_evidence_sufficiency(
            question,
            cited_results,
            parsed_response['evidence_sufficiency'],
            parsed_response['answer_text'],
        )

        # If independent assessment says insufficient, override the answer
        if final_sufficiency == "insufficient":
            return {
                'answer_text': "The provided documents do not contain sufficient information to answer this question.",
                'evidence_chunks': evidence_chunks,
                'citations': citations,
                'evidence_type': 'explicit_fact',
                'evidence_sufficiency': 'insufficient',
                'retrieval_similarity': self._retrieval_similarity(
                    cited_results
                ),
                'factual_confidence': None,
                'confidence': None,
                'support_status': 'insufficient',
                'requires_human_review': True,
            }

        # Build final response
        response = {
            'answer_text': parsed_response['answer_text'],
            'evidence_chunks': evidence_chunks,
            'citations': citations,
            'evidence_type': parsed_response['evidence_type'],
            'evidence_sufficiency': final_sufficiency,
            'retrieval_similarity': self._retrieval_similarity(cited_results),
            'factual_confidence': None,
            # Deprecated compatibility field; never use retrieval similarity
            # or the model's self-rating as factual confidence.
            'confidence': None,
            'support_status': (
                'verbatim_supported'
                if final_sufficiency == 'sufficient'
                else 'unverified'
            ),
            'requires_human_review': final_sufficiency != 'sufficient',
        }

        return response
