"""
Example Usage of Text Chunking

This script demonstrates how to use the TextChunker to split text into chunks.
"""

from chunking import TextChunker


def main():
    """Demonstrate text chunking."""
    print("Text Chunking Example")
    print("=" * 50)

    # Example 1: Chunk simple text
    print("\nExample 1: Chunk simple text")
    print("-" * 50)

    text = """This is a research paper about artificial intelligence.
The paper discusses various aspects of machine learning and natural language processing.
It includes multiple sections covering introduction, methodology, results, and conclusion.
The researchers conducted experiments on several datasets and reported their findings.
The conclusion highlights the importance of future work in this area."""

    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    result = chunker.chunk_simple_text(text, document_id="example_paper")

    print(f"Document ID: {result['document_id']}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Chunk size: {result['chunk_size']}")
    print(f"Chunk overlap: {result['chunk_overlap']}")

    for chunk in result['chunks']:
        print(f"\nChunk {chunk['chunk_index']}:")
        print(f"  Position: {chunk['start_char']}-{chunk['end_char']}")
        print(f"  Size: {chunk['chunk_size']} characters")
        print(f"  Text: {chunk['text'][:80]}...")

    # Example 2: Chunk with different configuration
    print("\n\nExample 2: Chunk with larger overlap")
    print("-" * 50)

    chunker_large_overlap = TextChunker(chunk_size=100, chunk_overlap=50)
    result2 = chunker_large_overlap.chunk_simple_text(text)

    print(f"Total chunks with larger overlap: {result2['total_chunks']}")
    print(f"More chunks created due to larger overlap")

    # Example 3: Show chunk ordering
    print("\n\nExample 3: Chunk ordering verification")
    print("-" * 50)

    for i, chunk in enumerate(result['chunks']):
        print(f"Chunk {i}: index={chunk['chunk_index']}, start={chunk['start_char']}, end={chunk['end_char']}")


if __name__ == "__main__":
    main()
