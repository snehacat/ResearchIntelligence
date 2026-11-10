"""Diagnose Groq extraction on one PDF after explicit data/cost confirmation."""

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from analysis.backend_factory import configured_model_name
from analysis.groq_backend import GroqBackend, GroqBackendError
from analysis.paper_analyzer import PaperAnalyzer
from chunking.text_chunker import TextChunker
from pdf_processing.text_extractor import TextExtractor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF to analyze")
    parser.add_argument(
        "--confirm-public-paper",
        action="store_true",
        help="Confirm this PDF excerpt may be sent to Groq.",
    )
    parser.add_argument(
        "--confirm-costs",
        action="store_true",
        help="Confirm the request may incur Groq charges.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_public_paper:
        parser.error("--confirm-public-paper is required.")
    if not args.confirm_costs:
        parser.error("--confirm-costs is required.")
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")

    extracted = TextExtractor().extract_from_file(str(args.pdf))
    chunker = TextChunker()
    chunks = chunker.chunk_extraction_result(
        extracted, document_id=args.pdf.stem
    )["chunks"]
    analyzer = PaperAnalyzer(
        chunker=chunker,
        retriever=None,
        answer_generator=None,
        model_name=configured_model_name("groq"),
        llm_backend_name="groq",
    )
    prompt = analyzer._build_extraction_prompt(chunks)
    backend = GroqBackend(model_name=analyzer.model_name)
    print(
        f"Pages: {extracted['total_pages']}; characters: "
        f"{extracted['total_characters']}; prompt characters: {len(prompt)}"
    )

    started = time.perf_counter()
    try:
        result = backend.generate_json(
            prompt,
            timeout_seconds=analyzer.llm_timeout_seconds,
            options={"num_predict": analyzer.MAX_GENERATED_TOKENS},
        )
    except GroqBackendError as error:
        print(f"Groq extraction request failed: {error}")
        return 1

    duration = time.perf_counter() - started
    output_dir = Path(__file__).parent / "raw_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{args.pdf.stem}_groq_response.json"
    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError:
        parsed = None
    output_path.write_text(result.text, encoding="utf-8")
    print(f"Model: {analyzer.model_name}")
    print(f"Valid JSON: {isinstance(parsed, dict)}")
    print(f"Call duration seconds: {duration:.2f}")
    print(f"Prompt tokens: {result.metadata.get('prompt_eval_count')}")
    print(f"Generated tokens: {result.metadata.get('eval_count')}")
    print(f"Response saved to: {output_path}")
    return 0 if isinstance(parsed, dict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
