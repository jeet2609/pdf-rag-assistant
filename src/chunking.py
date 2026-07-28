"""
Text Chunking Module

Responsible for:
- Splitting documents into smaller chunks
- Maintaining metadata
- Preparing documents for embedding generation
"""


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document(
	documents: list[Document],
	chunk_size: int = 1000,
	chunk_overlap: int = 200
) -> list[Document]:
	"""
	Split documents into smaller chunks.

	Args:
		documents:
			List of LangChain Document objects.

		chunk_size:
			Maximum characters per chunk.

		chunk_overlap:
			Number of overlapping characters between chunks.

	Returns:
		List of chunked Document objects.
	"""

	if not documents:
		raise ValueError(
			"Document list cannot be empty"
		)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len
	)

	chunks = text_splitter.split_documents(
		documents=documents
	)

	# Add unique chunk IDs
	for index, chunk in enumerate(chunks):
		chunk.metadata["chunk_id"] = index + 1

	return chunks