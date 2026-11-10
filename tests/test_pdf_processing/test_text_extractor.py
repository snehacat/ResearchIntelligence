"""
Tests for PDF Text Extraction Module

This module contains tests for the TextExtractor class, including:
- Successful text extraction
- Error handling for missing files, invalid PDFs, empty PDFs
- Handling of PDFs with no extractable text
- Partial extraction when some pages have text
"""

import pytest
import os
import tempfile
from pathlib import Path
import time

from pdf_processing.text_extractor import (
    TextExtractor,
    FileNotFoundError,
    InvalidPDFError,
    EmptyPDFError,
    NoExtractableTextError,
    PDFLimitExceededError
)


class TestTextExtractor:
    """Test cases for TextExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in Path(self.temp_dir).glob('*'):
            try:
                file.unlink()
            except PermissionError:
                # File might still be held by another process, try once more after a short delay
                time.sleep(0.1)
                try:
                    file.unlink()
                except PermissionError:
                    # If still fails, just continue (temp dir will be cleaned by OS eventually)
                    pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            # Directory might not be empty due to locked files
            pass

    def create_test_pdf(self, filename: str, text_content: str = "Test page content") -> str:
        """
        Create a simple test PDF with one page of text.

        Args:
            filename: Name of the PDF file to create
            text_content: Text content to put in the PDF

        Returns:
            Full path to the created PDF
        """
        import pymupdf

        doc = pymupdf.Document()
        page = doc.new_page()
        page.insert_text((50, 72), text_content, fontsize=12)
        pdf_path = os.path.join(self.temp_dir, filename)
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def create_multi_page_pdf(self, filename: str, num_pages: int = 3) -> str:
        """
        Create a test PDF with multiple pages.

        Args:
            filename: Name of the PDF file to create
            num_pages: Number of pages to create

        Returns:
            Full path to the created PDF
        """
        import pymupdf

        doc = pymupdf.Document()
        for i in range(num_pages):
            page = doc.new_page()
            page.insert_text((50, 72), f"Content for page {i + 1}", fontsize=12)
        pdf_path = os.path.join(self.temp_dir, filename)
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def create_empty_pdf(self, filename: str) -> str:
        """
        Create a PDF with no pages.

        Note: PyMuPDF does not allow saving documents with zero pages.
        This method is kept for documentation purposes but cannot create
        an actual empty PDF. For testing empty PDF behavior, a real
        empty PDF file would need to be provided.

        Args:
            filename: Name of the PDF file to create

        Returns:
            Full path to the created PDF (will raise NotImplementedError)
        """
        raise NotImplementedError("PyMuPDF does not allow saving documents with zero pages")

    def create_mixed_content_pdf(self, filename: str) -> str:
        """
        Create a PDF with some pages containing text and some empty pages.

        Args:
            filename: Name of the PDF file to create

        Returns:
            Full path to the created PDF
        """
        import pymupdf

        doc = pymupdf.Document()
        # Page 1: with text
        page1 = doc.new_page()
        page1.insert_text((50, 72), "Text content on page 1", fontsize=12)
        # Page 2: empty (no text)
        doc.new_page()
        # Page 3: with text
        page3 = doc.new_page()
        page3.insert_text((50, 72), "Text content on page 3", fontsize=12)
        # Page 4: empty
        doc.new_page()

        pdf_path = os.path.join(self.temp_dir, filename)
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def create_no_text_pdf(self, filename: str) -> str:
        """
        Create a PDF with pages but no extractable text (blank pages).

        Args:
            filename: Name of the PDF file to create

        Returns:
            Full path to the created PDF
        """
        import pymupdf

        doc = pymupdf.Document()
        # Create multiple blank pages
        for _ in range(3):
            doc.new_page()

        pdf_path = os.path.join(self.temp_dir, filename)
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def test_successful_extraction_single_page(self):
        """Test successful extraction from a single-page PDF."""
        pdf_path = self.create_test_pdf("test_single.pdf", "This is test content")

        result = self.extractor.extract_from_file(pdf_path)

        assert result['success'] is True
        assert result['total_pages'] == 1
        assert result['pages_with_text'] == 1
        assert result['total_characters'] > 0
        assert len(result['pages']) == 1
        assert result['pages'][0]['page_number'] == 1
        assert "test content" in result['pages'][0]['text']
        assert 'extraction_time_seconds' in result
        assert result['extraction_time_seconds'] >= 0

    def test_successful_extraction_multi_page(self):
        """Test successful extraction from a multi-page PDF."""
        pdf_path = self.create_multi_page_pdf("test_multi.pdf", 3)

        result = self.extractor.extract_from_file(pdf_path)

        assert result['success'] is True
        assert result['total_pages'] == 3
        assert result['pages_with_text'] == 3
        assert len(result['pages']) == 3
        assert result['pages'][0]['page_number'] == 1
        assert result['pages'][1]['page_number'] == 2
        assert result['pages'][2]['page_number'] == 3
        assert 'extraction_time_seconds' in result
        assert result['extraction_time_seconds'] >= 0

    def test_extract_from_bytes_enforces_page_and_text_limits(self):
        pdf_path = self.create_multi_page_pdf("limited.pdf", 2)
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        with pytest.raises(PDFLimitExceededError, match="pages; limit"):
            self.extractor.extract_from_bytes(pdf_bytes, max_pages=1)
        with pytest.raises(PDFLimitExceededError, match="character limit"):
            self.extractor.extract_from_bytes(pdf_bytes, max_characters=1)

        result = self.extractor.extract_from_bytes(
            pdf_bytes,
            max_pages=2,
            max_characters=1000,
        )
        assert result["total_pages"] == 2
        assert result["pages_with_text"] == 2

    @pytest.mark.parametrize(
        ("max_pages", "max_characters"),
        [(0, None), (True, None), (None, 0), (None, False)],
    )
    def test_extract_from_bytes_validates_limits(self, max_pages, max_characters):
        with pytest.raises(ValueError):
            self.extractor.extract_from_bytes(
                b"%PDF-1.7",
                max_pages=max_pages,
                max_characters=max_characters,
            )

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            self.extractor.extract_from_file("nonexistent.pdf")

        assert "PDF file not found" in str(exc_info.value)

    def test_invalid_pdf(self):
        """Test handling of invalid PDF file."""
        # PyMuPDF can open some non-PDF files, so we test with a truly invalid PDF
        # Create a file with invalid PDF magic bytes
        invalid_path = os.path.join(self.temp_dir, "invalid.pdf")
        with open(invalid_path, 'wb') as f:
            f.write(b"This is not a PDF file at all")

        with pytest.raises(InvalidPDFError) as exc_info:
            self.extractor.extract_from_file(invalid_path)

        assert "Cannot open PDF file" in str(exc_info.value)

    def test_empty_pdf(self):
        """
        Test handling of PDF with no pages.

        Note: PyMuPDF does not allow saving documents with zero pages,
        so we cannot programmatically create an empty PDF for testing.
        This test documents the limitation. For actual empty PDF testing,
        a real empty PDF file would need to be provided in the test fixtures.
        """
        pytest.skip("Cannot programmatically create empty PDF with PyMuPDF (requires pre-made test file)")

    def test_directory_path(self):
        """Test handling when path is a directory, not a file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            self.extractor.extract_from_file(self.temp_dir)

        assert "Path is not a file" in str(exc_info.value)

    def test_get_text_as_string(self):
        """Test converting extraction result to string with page markers."""
        pdf_path = self.create_multi_page_pdf("test_string.pdf", 2)
        result = self.extractor.extract_from_file(pdf_path)

        text_string = self.extractor.get_text_as_string(result)

        assert "[Page 1]" in text_string
        assert "[Page 2]" in text_string
        assert "Content for page 1" in text_string
        assert "Content for page 2" in text_string

    def test_get_text_by_page(self):
        """Test getting text for a specific page."""
        pdf_path = self.create_multi_page_pdf("test_page_specific.pdf", 3)
        result = self.extractor.extract_from_file(pdf_path)

        page_1_text = self.extractor.get_text_by_page(result, 1)
        page_2_text = self.extractor.get_text_by_page(result, 2)
        page_3_text = self.extractor.get_text_by_page(result, 3)

        assert "page 1" in page_1_text
        assert "page 2" in page_2_text
        assert "page 3" in page_3_text

    def test_get_text_by_page_invalid(self):
        """Test getting text for a non-existent page."""
        pdf_path = self.create_test_pdf("test_invalid_page.pdf")
        result = self.extractor.extract_from_file(pdf_path)

        invalid_page_text = self.extractor.get_text_by_page(result, 99)

        assert invalid_page_text is None

    def test_partial_extraction_mixed_content(self):
        """Test extraction from PDF with some pages having text and others blank."""
        pdf_path = self.create_mixed_content_pdf("test_mixed.pdf")

        result = self.extractor.extract_from_file(pdf_path)

        # Should succeed despite some pages having no text
        assert result['success'] is True
        assert result['total_pages'] == 4
        assert result['pages_with_text'] == 2  # Only pages 1 and 3 have text
        assert result['pages_without_text'] == [2, 4]  # Pages 2 and 4 are blank
        assert 'extraction_time_seconds' in result
        assert result['extraction_time_seconds'] >= 0

        # Verify text from readable pages is preserved with correct page numbers
        assert result['pages'][0]['page_number'] == 1
        assert "Text content on page 1" in result['pages'][0]['text']
        assert result['pages'][1]['page_number'] == 3
        assert "Text content on page 3" in result['pages'][1]['text']

        # Verify the extraction result does not claim entire document failed
        assert result['success'] is True
        assert result['total_characters'] > 0

    def test_no_extractable_text(self):
        """Test handling of PDF with pages but no extractable text."""
        pdf_path = self.create_no_text_pdf("test_no_text.pdf")

        with pytest.raises(NoExtractableTextError) as exc_info:
            self.extractor.extract_from_file(pdf_path)

        # Verify the error message accurately describes the limitation
        error_msg = str(exc_info.value)
        assert "No extractable text found" in error_msg
        assert "may be a scanned document" in error_msg
        assert "image-based PDF" in error_msg
        assert "otherwise non-extractable format" in error_msg

    def test_extraction_result_structure(self):
        """Test that extraction result has the expected structure."""
        pdf_path = self.create_test_pdf("test_structure.pdf")
        result = self.extractor.extract_from_file(pdf_path)

        # Check all expected keys are present
        expected_keys = ['pages', 'total_pages', 'pages_with_text',
                        'pages_without_text', 'total_characters', 'success']
        for key in expected_keys:
            assert key in result

        # Check types
        assert isinstance(result['pages'], list)
        assert isinstance(result['total_pages'], int)
        assert isinstance(result['pages_with_text'], int)
        assert isinstance(result['pages_without_text'], list)
        assert isinstance(result['total_characters'], int)
        assert isinstance(result['success'], bool)
