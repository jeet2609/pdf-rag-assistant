"""
FAISS Vector Store Module

Responsibilities:
- Create a FAISS vector store from document chunks
- Save the vector store locally
- Load an existing vector store
- Create a retriever for similarity search
"""

from pathlib import Path

from config.settings import settings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_vector_store(
	documents: list[Document],
	embedding_model: Embeddings
) -> FAISS:
	"""
	Create a FAISS vector store from document chunks.

	Args:
		documents:
			List of LangChain Document chunks.

		embedding_model:
			Embedding model used to generate vector embeddings.

	Returns:
		FAISS vector store.
	"""

	if not documents:
		raise ValueError(
			"Document list cannot be empty."
		)

	try:
		vectore_store = FAISS.from_documents(
			documents=documents,
			embedding=embedding_model
		)

		return vectore_store

	except Exception as error:
		raise Exception(  # noqa: TRY002
			f"Failed to create FAISS vector store: {error}"
		) from error



def save_vector_store(
	vector_store: FAISS,
	save_path: Path = settings.vector_store_directory
) -> None:
	"""
	Save FAISS index to disk.

	Args:
		vector_store:
			FAISS vector store.

		save_path:
			Directory where the index will be saved.
	"""

	try:
		save_path.mkdir(
			parents=True,
			exist_ok=True,
		)

		vector_store.save_local(
			folder_path=str(save_path)
		)
		
	except Exception as error:
		raise Exception(  # noqa: TRY002
			f"Failed to save FAISS index: {error}"
		) from error


def load_vector_store(
	embedding_model: Embeddings,
	load_path: Path = settings.vector_store_directory
) -> FAISS:
	"""
	Load an existing FAISS vector store.

	Args:
		embedding_model:
			Same embedding model used while creating the index.

		load_path:
			Location of saved FAISS index.

	Returns:
		Loaded FAISS vector store.
	"""

	if not load_path.exists():
		raise ValueError(
			f"FAISS index not found: {load_path}"
		)

	try:
		vectore_store = FAISS.load_local(
			folder_path=str(load_path),
			embeddings=embedding_model,
			allow_dangerous_deserialization=True,
		)

		return vectore_store
		
	except Exception as error:
		raise Exception(  # noqa: TRY002
			f"Failed to load FAISS index: {error}"
		) from error


# Why embedding_model is required ?
# Short answer
# FAISS itself does not create embeddings.
# FAISS only stores and searches vectors.
# So we must provide an embedding model whenever we create or query the FAISS index.