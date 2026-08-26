"""
Document Retrieval Module

Responsibilities:
- Create a retriever from a FAISS vector store
- Retrieve relevant document chunks
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings


def create_retriever(
	vector_store: FAISS,
	k: int = settings.retrieval_k,
) -> VectorStoreRetriever:
	"""
	Create a retriever from a FAISS vector store.

	Args:
		vector_store:
			Loaded FAISS vector store.

		k:
			Number of document chunks to retrieve.

	Returns:
		Configured LangChain retriever.
	"""

	return vector_store.as_retriever(
		search_type="similarity",
		search_kwargs={
			"k": k,
		},
	)


def retrieve_documents(
	question: str,
	retriever: VectorStoreRetriever,
) -> list[Document]:
	"""
	Retrieve the most relevant document chunks for a question.

	Args:
		question:
			User question.

		retriever:
			Configured LangChain retriever.

	Returns:
		List of relevant document chunks.

	Raises:
		ValueError:
			If the question is empty.

		RuntimeError:
			If document retrieval fails.
	"""

	if not question.strip():
		raise ValueError(
			"Question cannot be empty."
		)

	try:
		documents = retriever.invoke(question)

		return documents

	except Exception as error:
		raise RuntimeError(
			f"Failed to retrieve documents: {error}"
		) from error