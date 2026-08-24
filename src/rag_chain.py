"""
RAG Pipeline Module

Responsibilities:
- Load the Gemini LLM
- Retrieve relevant document chunks
- Build the prompt
- Generate grounded answers
- Return answer with source documents
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from src.prompts import get_rag_prompt


def load_llm(
	gemini_model: str = settings.llm_model_name
) -> ChatGoogleGenerativeAI:
	"""
	Load the Gemini language model.

	Args:
		gemini_model:
			Set to default: gemini-3.5-flash

	Returns:
		Configured Gemini chat model.
	"""

	return ChatGoogleGenerativeAI(
		model=gemini_model,
		temperature=0,
		api_key=settings.google_api_key
	)


def generate_answer(
	question: str,
	retriever: VectorStoreRetriever,
	llm: ChatGoogleGenerativeAI
) -> str:
	"""
	Generate an answer using the RAG pipeline.

	Args:
		question:
			User question.

		retriever:
			Configured LangChain retriever.

		llm:
			Configured Gemini model.

	Returns:
		Generated answer.
	"""

	if not question.strip():
		raise ValueError(
			"Question cannot be empty."
		)

	try:
		# Retrieve relevant documents
		documents = retriever.invoke(question)

		# Combine retrieved chunks into one context string
		context = "\n\n".join(
			document.page_content
			for document in documents
		)

		# Build prompt
		prompt = get_rag_prompt()

		formatted_prompt = prompt.invoke(
			{
				"context": context,
				"question": question
			}
		)

		# Generate response
		response = llm.invoke(formatted_prompt)

		return response.content

	except Exception as error:
		raise RuntimeError(
			f"Failed to generate answer: {error}"
		) from error


def generate_answer_with_sources(
	question: str,
	retriever: VectorStoreRetriever,
	llm: ChatGoogleGenerativeAI
) -> dict:
	"""
	Generate an answer using the RAG pipeline.

	Args:
		question:
			User question.

		retriever:
			Configured LangChain retriever.

		llm:
			Configured Gemini model.

	Returns:
		Dictionary containing:
		- Generated answer
		- Retrieved source documents
	"""

	if not question.strip():
		raise ValueError(
			"Question cannot be empty"
		)

	try:
		# Retrieve relevant documents
		documents: list[Document] = retriever.invoke(question)

		# Combine retrieved chunks into one context string
		context = "\n\n".join(
			document.page_content
			for document in documents
		)

		# Build prompt
		prompt = get_rag_prompt()

		formatted_prompt = prompt.invoke(
			{
				"context": context,
				"question": question
			}
		)

		# Generate response
		response = llm.invoke(formatted_prompt)

		return {
			"answer": str(response.text),
			"documents": documents
		}

	except Exception as error:
		raise RuntimeError(
			f"Failed to generate answer: {error}"
		) from error