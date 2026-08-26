"""
FAISS Vector Store Module

Responsibilities:
- Create a FAISS vector store from document chunks
- Save the vector store locally
- Load an existing vector store
"""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings


def create_vector_store(
	documents: list[Document],
	embedding_model: Embeddings,
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

	Raises:
		ValueError:
			If the document list is empty.

		RuntimeError:
			If the FAISS vector store cannot be created.
	"""

	if not documents:
		raise ValueError(
			"Document list cannot be empty."
		)

	try:
		vector_store = FAISS.from_documents(
			documents=documents,
			embedding=embedding_model,
		)

		return vector_store

	except Exception as error:
		raise RuntimeError(
			f"Failed to create FAISS vector store: {error}"
		) from error


def save_vector_store(
	vector_store: FAISS,
	save_path: Path = settings.vector_store_directory,
) -> None:
	"""
	Save the FAISS vector store to disk.

	Args:
		vector_store:
			FAISS vector store to save.

		save_path:
			Directory where the FAISS index will be saved.

	Raises:
		RuntimeError:
			If the FAISS vector store cannot be saved.
	"""

	try:
		save_path.mkdir(
			parents=True,
			exist_ok=True,
		)

		vector_store.save_local(
			folder_path=str(save_path),
		)

	except Exception as error:
		raise RuntimeError(
			f"Failed to save FAISS index: {error}"
		) from error


def load_vector_store(
	embedding_model: Embeddings,
	load_path: Path = settings.vector_store_directory,
) -> FAISS:
	"""
	Load an existing FAISS vector store from disk.

	Args:
		embedding_model:
			Same embedding model used while creating the index.

		load_path:
			Location of the saved FAISS index.

	Returns:
		Loaded FAISS vector store.

	Raises:
		ValueError:
			If the FAISS index does not exist.

		RuntimeError:
			If the FAISS index cannot be loaded.
	"""

	if not load_path.exists():
		raise ValueError(
			f"FAISS index not found: {load_path}"
		)

	try:
		vector_store = FAISS.load_local(
			folder_path=str(load_path),
			embeddings=embedding_model,
			allow_dangerous_deserialization=True,
		)

		return vector_store

	except Exception as error:
		raise RuntimeError(
			f"Failed to load FAISS index: {error}"
		) from error


def save_index_metadata(
	filename: str,
	modified_time: float,
	total_page: int,
	chunks: int,
	save_path: Path = settings.vector_store_directory,
) -> None:
	"""
	Save metadata for the persisted FAISS index.
	"""

	metadata = {
		"filename": filename,
		"modified_time": modified_time,
		"total_page": total_page,
		"chunks": chunks,
	}

	try:
		save_path.mkdir(
			parents=True,
			exist_ok=True,
		)

		metadata_path = save_path / "metadata.json"

		metadata_path.write_text(
			json.dumps(metadata, indent=4),
			encoding="utf-8",
		)

	except OSError as error:
		raise RuntimeError(
			f"Failed to save FAISS index metadata: {error}"
		) from error

def load_index_metadata(
	load_path: Path = settings.vector_store_directory,
) -> dict:
	"""
	Load metadata for the persisted FAISS index.
	"""

	metadata_path = load_path / "metadata.json"

	if not metadata_path.exists():
		raise FileNotFoundError(
			f"FAISS index metadata not found: {metadata_path}"
		)

	try:
		return json.loads(
			metadata_path.read_text(
				encoding="utf-8"
			)
		)

	except (OSError, json.JSONDecodeError) as error:
		raise RuntimeError(
			f"Failed to load FAISS index metadata: {error}"
		) from error