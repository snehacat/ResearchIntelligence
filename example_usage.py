"""
Example Usage of PDF Text Extractor

This script demonstrates how to use the TextExtractor to extract text from PDF files.
"""

from pdf_processing import TextExtractor


def main():
    """Demonstrate PDF text extraction."""
    extractor = TextExtractor()

    # Example 1: Extract text from a PDF file
    print("Example 1: Extract text from a PDF file")
    print("-" * 50)

    pdf_path = input("Enter path to PDF file: ").strip()

    try:
        result = extractor.extract_from_file(pdf_path)

        print(f"\nExtraction successful!")
        print(f"Total pages: {result['total_pages']}")
        print(f"Pages with text: {result['pages_with_text']}")
        print(f"Total characters: {result['total_characters']}")

        if result['pages_without_text']:
            print(f"Pages without text: {result['pages_without_text']}")

        # Show first page content
        if result['pages']:
            print(f"\nFirst page content (Page {result['pages'][0]['page_number']}):")
            print("-" * 50)
            print(result['pages'][0]['text'][:500] + "..." if len(result['pages'][0]['text']) > 500 else result['pages'][0]['text'])

        # Get text as a single string with page markers
        text_string = extractor.get_text_as_string(result)
        print(f"\n\nFull text with page markers (first 1000 chars):")
        print("-" * 50)
        print(text_string[:1000] + "..." if len(text_string) > 1000 else text_string)

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
