"""
Document Loader Module

Responsible for:
- Loading PDF documents
- Extracting page-wise content
- Preserving document metadata

This module is the first step in the RAG pipeline.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(
	file_path : str
) -> list[Document]:
	"""
	Load a PDF file and convert it into LangChain Document objects.

	Each page of the PDF becomes a separate Document object.

	Args:
		file_path (str):
			Path to the PDF file.

	Returns:
		List[Document]:
			List of LangChain Document objects containing:
			- page_content: Extracted text
			- metadata: Source information and page number

	Raises:
		FileNotFoundError:
			If PDF file does not exist.

		ValueError:
			If provided file is not a PDF.

		Exception:
			For unexpected loading failures.
    """

	pdf_path = Path(file_path)

	# Validate file existence
	if not pdf_path.exists():
		raise ValueError(
			f"PDF file not found: {pdf_path}"
		)

	# Validate file extension
	if pdf_path.suffix.lower() != '.pdf':
		raise ValueError(
			"Only PDF files are supported."
		)

	try:
		# LangChain PDF loader
		loader = PyPDFLoader(
			str(pdf_path)
		)

		# Extract pages as Document objects
		documents = loader.load()

		return documents

	except Exception as error:
		raise Exception(  # noqa: TRY002
			f"Failed to load PDF '{file_path}': {error}"
		) from error


def get_pdf_metadata(
		documents: list[Document]
) -> dict:
	"""
	Extract basic metadata from loaded PDF documents.

	Args:
		documents:
			List of LangChain Document objects.

	Returns:
		Dictionary containing PDF information.
    """

	if not documents:
		return {}

	return {
		"total_page": len(documents),
		"source": documents[0].metadata.get("source")
	}