"""Generate a gold-set annotation template for a given paper.

The generated file is placed at evaluation/gold_set/<document_id>.json with
every value set to "TODO" so the schema validator catches unfilled entries.

Usage
-----
    python evaluation/generate_template.py --paper attention_is_all_you_need
    python evaluation/generate_template.py --paper my_paper --papers-dir /path/to/pdfs
    python evaluation/generate_template.py --list-papers
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# Project root on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.gold_set_schema import ALL_FIELDS, LIST_FIELDS, STRING_FIELDS


def _default_papers_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "validation" / "step5" / "papers"


def _gold_set_dir() -> Path:
    return Path(__file__).resolve().parent / "gold_set"


def build_template(document_id: str, source_pdf: str) -> dict:
    """Return a template annotation dict with every value set to 'TODO'."""
    fields: dict = {}
    for field_name in ALL_FIELDS:
        if field_name in LIST_FIELDS:
            fields[field_name] = {
                "value": ["TODO"],
                "evidence_pages": [],
            }
        else:
            fields[field_name] = {
                "value": "TODO",
                "evidence_pages": [],
            }

    return {
        "document_id": document_id,
        "source_pdf": source_pdf,
        "annotator": "TODO",
        "annotation_date": date.today().isoformat(),
        "notes": None,
        "fields": fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a gold-set annotation template."
    )
    parser.add_argument(
        "--paper",
        help="Paper filename without .pdf extension (e.g. attention_is_all_you_need)",
    )
    parser.add_argument(
        "--papers-dir",
        default=None,
        help="Directory containing PDF papers (default: validation/step5/papers)",
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="List available papers in the papers directory and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the template (default: evaluation/gold_set/)",
    )
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir) if args.papers_dir else _default_papers_dir()
    output_dir = Path(args.output_dir) if args.output_dir else _gold_set_dir()

    if args.list_papers:
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in {papers_dir}")
        else:
            print(f"Papers available in {papers_dir}:")
            for pdf in pdfs:
                already = (output_dir / f"{pdf.stem}.json").exists()
                status = " [template exists]" if already else ""
                print(f"  {pdf.stem}{status}")
        return

    if not args.paper:
        parser.error("--paper is required unless --list-papers is used.")

    document_id = args.paper
    pdf_path = papers_dir / f"{document_id}.pdf"

    if not pdf_path.exists():
        # Allow template generation even without the PDF, but warn
        print(
            f"Warning: PDF not found at {pdf_path}. "
            "Template will be created with the given document_id; "
            "update 'source_pdf' manually if the path differs."
        )
        source_pdf_rel = f"validation/step5/papers/{document_id}.pdf"
    else:
        # Make path relative to project root
        project_root = Path(__file__).resolve().parent.parent
        try:
            source_pdf_rel = str(pdf_path.relative_to(project_root)).replace("\\", "/")
        except ValueError:
            source_pdf_rel = str(pdf_path).replace("\\", "/")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document_id}.json"

    if output_path.exists():
        print(f"Template already exists at {output_path}.")
        print("Delete it first if you want to regenerate.")
        return

    template = build_template(document_id, source_pdf_rel)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"Template created: {output_path}")
    print()
    print("Fill in each field:")
    print("  - Set 'annotator' to your name.")
    print("  - For each field under 'fields':")
    print("    - Replace the 'value' with the correct answer from the PDF.")
    print("    - Set 'evidence_pages' to [page_number, ...] where the info appears.")
    print("    - Use null (not 'TODO') if the information is genuinely absent.")
    print()
    print("Then run the harness:")
    print(
        f"  python evaluation/run_harness.py "
        f"--gold-dir evaluation/gold_set "
        f"--papers-dir {papers_dir}"
    )


if __name__ == "__main__":
    main()
