"""
Text Chunking Module

Responsible for:
- Splitting documents into smaller chunks
- Maintaining metadata
- Preparing documents for embedding generation
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


def split_document(
	documents: list[Document],
	chunk_size: int = settings.chunk_size,
	chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
	"""
	Split documents into smaller chunks.

	Args:
		documents:
			List of LangChain Document objects.

		chunk_size:
			Maximum number of characters per chunk.

		chunk_overlap:
			Number of overlapping characters between chunks.

	Returns:
		List of chunked Document objects with unique chunk IDs.

	Raises:
		ValueError:
			If the document list is empty or chunking parameters
			are invalid.
	"""

	if not documents:
		raise ValueError(
			"Document list cannot be empty."
		)

	if chunk_size <= 0:
		raise ValueError(
			"chunk_size must be greater than 0."
		)

	if chunk_overlap < 0:
		raise ValueError(
			"chunk_overlap cannot be negative."
		)

	if chunk_overlap >= chunk_size:
		raise ValueError(
			"chunk_overlap must be smaller than chunk_size."
		)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
	)

	chunks = text_splitter.split_documents(documents)

	# Add unique chunk IDs
	for chunk_id, chunk in enumerate(chunks, start=1):
		chunk.metadata["chunk_id"] = chunk_id

	return chunks