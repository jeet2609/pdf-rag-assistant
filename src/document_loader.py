"""
Document Loader Module

Responsible for:
- Loading PDF documents
- Extracting page-wise content
- Preserving document metadata

This module is the first step in the RAG pipeline.
"""

from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFMetadata(TypedDict):
	"""Metadata extracted from a PDF document."""

	total_page: int
	source: str | None


def load_pdf(
	file_path: str | Path,
) -> list[Document]:
	"""
	Load a PDF file and convert it into LangChain Document objects.

	Each page of the PDF becomes a separate Document object.

	Args:
		file_path:
			Path to the PDF file.

	Returns:
		List of LangChain Document objects containing:
		- page_content: Extracted text
		- metadata: Source information and page number

	Raises:
		FileNotFoundError:
			If the PDF file does not exist.

		ValueError:
			If the provided file is not a PDF or
			no documents are loaded.

		Exception:
			If an unexpected PDF loading error occurs.
	"""

	pdf_path = Path(file_path)

	# Validate file existence
	if not pdf_path.exists():
		raise FileNotFoundError(
			f"PDF file not found: {pdf_path}"
		)

	# Validate file extension
	if pdf_path.suffix.lower() != ".pdf":
		raise ValueError(
			"Only PDF files are supported."
		)

	# Load PDF using LangChain
	loader = PyPDFLoader(str(pdf_path))

	# Extract pages as Document objects
	documents = loader.load()

	if not documents:
		raise ValueError(
			f"No content could be extracted from PDF: {pdf_path}"
		)

	return documents


def get_pdf_metadata(
	documents: list[Document],
) -> PDFMetadata:
	"""
	Extract basic metadata from loaded PDF documents.

	Args:
		documents:
			List of LangChain Document objects.

	Returns:
		Dictionary containing PDF information.

	Raises:
		ValueError:
			If no documents are provided.
	"""

	if not documents:
		raise ValueError(
			"Cannot extract metadata from an empty document list."
		)

	return {
		"total_page": len(documents),
		"source": documents[0].metadata.get("source"),
	}