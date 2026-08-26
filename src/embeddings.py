"""
Embedding Generation Module

Responsible for:
- Loading HuggingFace embedding model
- Detecting the available compute device
- Providing embeddings for FAISS similarity search
"""

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


def get_device() -> str:
	"""
	Detect the available compute device.

	Priority:
		1. Apple Silicon GPU (MPS)
		2. CPU

	Returns:
		Device name used by the embedding model.
	"""

	if torch.backends.mps.is_available():
		return "mps"

	return "cpu"


def get_embedding_model() -> HuggingFaceEmbeddings:
	"""
	Initialize and return the HuggingFace embedding model.

	Returns:
		Configured HuggingFace embedding model.

	Raises:
		RuntimeError:
			If the embedding model fails to load.
	"""

	device = get_device()

	model_name = settings.embedding_model_name

	model_kwargs = {
		"device": device,
	}

	encode_kwargs = {
		"normalize_embeddings": True,
	}

	try:
		embedding_model = HuggingFaceEmbeddings(
			model_name=model_name,
			model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs,
		)

		return embedding_model

	except Exception as error:
		raise RuntimeError(
			f"Failed to load embedding model "
			f"'{model_name}' on device '{device}'."
		) from error

# No need of this, as we are doing this in create_vector_store in file vector_store.py
def generate_embeddings(
	documents: list[Document],
	embedding_model: HuggingFaceEmbeddings
) -> list[list[float]]:
	"""
	Generate embeddings for document chunks.

	Args:
		documents:
			List of LangChain Document chunks.

		embedding_model:
			HuggingFace embedding model.

	Returns:
		List of embedding vectors.
	"""

	if not documents:
		raise ValueError(
			"Document list cannot be empty."
		)

	try:
		texts = [
			document.page_content
			for document in documents
		]

		embeddings = embedding_model.embed_documents(
			texts
		)

		return embeddings

	except Exception as error:
		raise Exception(  # noqa: TRY002
			f"Failed to load embedding model: {error}"
		) from error