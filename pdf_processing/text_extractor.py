"""
PDF Text Extraction Module

This module provides functionality to extract text from PDF files page by page,
preserving page numbers and handling various error conditions gracefully.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import pymupdf  # PyMuPDF


class PDFExtractionError(Exception):
    """Base exception for PDF extraction errors."""
    pass


class FileNotFoundError(PDFExtractionError):
    """Raised when the PDF file does not exist."""
    pass


class InvalidPDFError(PDFExtractionError):
    """Raised when the file is not a valid PDF or is corrupt."""
    pass


class EmptyPDFError(PDFExtractionError):
    """Raised when the PDF has no pages."""
    pass


class NoExtractableTextError(PDFExtractionError):
    """Raised when no text can be extracted from the PDF."""
    pass


class PDFLimitExceededError(PDFExtractionError):
    """Raised when a configured PDF page or extracted-text limit is exceeded."""
    pass


class TextExtractor:
    """
    Extracts text from PDF files page by page.

    This class provides methods to:
    - Extract text from PDF files
    - Preserve page numbers for traceability
    - Handle missing, invalid, empty, and non-extractable PDFs
    - Support partial extraction when some pages have text
    """

    def __init__(self):
        """Initialize the TextExtractor."""
        pass

    def extract_from_file(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing:
                - 'pages': List of dictionaries with 'page_number' and 'text'
                - 'total_pages': Total number of pages in PDF
                - 'pages_with_text': Number of pages that contained extractable text
                - 'pages_without_text': List of page numbers with no extractable text
                - 'total_characters': Total number of characters extracted
                - 'success': Boolean indicating if extraction succeeded
                - 'extraction_time_seconds': Elapsed time for extraction

        Raises:
            FileNotFoundError: If the PDF file does not exist
            InvalidPDFError: If the file is not a valid PDF or is corrupt
            EmptyPDFError: If the PDF has no pages
        """
        started_at = time.perf_counter()
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check if file is a file (not a directory)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"Path is not a file: {pdf_path}")

        # Try to open the PDF
        doc = None
        try:
            doc = pymupdf.Document(pdf_path)
        except Exception as e:
            raise InvalidPDFError(f"Cannot open PDF file: {pdf_path}. Error: {str(e)}")

        try:
            # Check if PDF is empty (no pages)
            page_count = doc.page_count
            if page_count == 0:
                raise EmptyPDFError(f"PDF file has no pages: {pdf_path}")

            # Extract text page by page
            pages = []
            pages_without_text = []
            total_characters = 0

            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()

                # Strip whitespace to check if page has meaningful content
                stripped_text = text.strip()

                if stripped_text:
                    pages.append({
                        'page_number': page_num + 1,  # 1-based page numbering
                        'text': text
                    })
                    total_characters += len(text)
                else:
                    pages_without_text.append(page_num + 1)

            # Check if any text was extracted
            if not pages:
                raise NoExtractableTextError(
                    f"No extractable text found in PDF: {pdf_path}. "
                    f"This may be a scanned document, image-based PDF, or otherwise non-extractable format."
                )

            elapsed = time.perf_counter() - started_at
            return {
                'pages': pages,
                'total_pages': page_count,
                'pages_with_text': len(pages),
                'pages_without_text': pages_without_text,
                'total_characters': total_characters,
                'success': True,
                'extraction_time_seconds': elapsed
            }
        finally:
            # Ensure document is closed even if an exception occurs
            if doc is not None:
                doc.close()

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        *,
        max_pages: Optional[int] = None,
        max_characters: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            Dictionary containing extraction results (same format as extract_from_file)

        Raises:
            InvalidPDFError: If the bytes do not represent a valid PDF
            EmptyPDFError: If the PDF has no pages
            NoExtractableTextError: If no text can be extracted
            PDFLimitExceededError: If configured page or character limit is exceeded
        """
        if not isinstance(pdf_bytes, bytes):
            raise InvalidPDFError("PDF content must be bytes")
        if max_pages is not None and (
            not isinstance(max_pages, int)
            or isinstance(max_pages, bool)
            or max_pages <= 0
        ):
            raise ValueError("max_pages must be a positive integer")
        if max_characters is not None and (
            not isinstance(max_characters, int)
            or isinstance(max_characters, bool)
            or max_characters <= 0
        ):
            raise ValueError("max_characters must be a positive integer")

        started_at = time.perf_counter()

        # Try to open the PDF from bytes
        doc = None
        try:
            doc = pymupdf.Document(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise InvalidPDFError(f"Cannot open PDF from bytes. Error: {str(e)}")

        try:
            # Check if PDF is empty (no pages)
            page_count = doc.page_count
            if page_count == 0:
                raise EmptyPDFError("PDF has no pages")
            if max_pages is not None and page_count > max_pages:
                raise PDFLimitExceededError(
                    f"PDF has {page_count} pages; limit is {max_pages}"
                )

            # Extract text page by page
            pages = []
            pages_without_text = []
            total_characters = 0

            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()

                # Strip whitespace to check if page has meaningful content
                stripped_text = text.strip()

                if stripped_text:
                    if (
                        max_characters is not None
                        and total_characters + len(text) > max_characters
                    ):
                        raise PDFLimitExceededError(
                            "PDF extracted text exceeds the "
                            f"{max_characters}-character limit"
                        )
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text
                    })
                    total_characters += len(text)
                else:
                    pages_without_text.append(page_num + 1)

            # Check if any text was extracted
            if not pages:
                raise NoExtractableTextError(
                    "No extractable text found in PDF. "
                    "This may be a scanned document, image-based PDF, or otherwise non-extractable format."
                )

            elapsed = time.perf_counter() - started_at
            return {
                'pages': pages,
                'total_pages': page_count,
                'pages_with_text': len(pages),
                'pages_without_text': pages_without_text,
                'total_characters': total_characters,
                'success': True,
                'extraction_time_seconds': elapsed
            }
        finally:
            # Ensure document is closed even if an exception occurs
            if doc is not None:
                doc.close()

    def get_text_as_string(self, extraction_result: Dict[str, any]) -> str:
        """
        Convert extraction result to a single string with page markers.

        Args:
            extraction_result: Result dictionary from extract_from_file or extract_from_bytes

        Returns:
            String with all text concatenated, with page markers like "[Page 1]"
        """
        text_parts = []
        for page_data in extraction_result['pages']:
            text_parts.append(f"[Page {page_data['page_number']}]\n{page_data['text']}")
        return "\n\n".join(text_parts)

    def get_text_by_page(self, extraction_result: Dict[str, any], page_number: int) -> Optional[str]:
        """
        Get text for a specific page number.

        Args:
            extraction_result: Result dictionary from extract_from_file or extract_from_bytes
            page_number: Page number (1-based)

        Returns:
            Text for the specified page, or None if page has no text
        """
        for page_data in extraction_result['pages']:
            if page_data['page_number'] == page_number:
                return page_data['text']
        return None
