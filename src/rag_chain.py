"""
RAG Pipeline Module

Responsibilities:
- Load the Gemini LLM
- Retrieve relevant document chunks
- Build the prompt
- Generate grounded answers
- Return answers with source documents
"""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from src.prompts import get_rag_prompt


class RAGResult(TypedDict):
	"""
	Type definition for the RAG pipeline result.
	"""

	answer: str
	documents: list[Document]


def load_llm(
	gemini_model: str = settings.llm_model_name,
) -> ChatGoogleGenerativeAI:
	"""
	Load the Gemini language model.

	Args:
		gemini_model:
			Name of the Gemini model to use.

	Returns:
		Configured Gemini chat model.
	"""

	return ChatGoogleGenerativeAI(
		model=gemini_model,
		api_key=settings.google_api_key,
	)


def generate_answer(
	question: str,
	retriever: VectorStoreRetriever,
	llm: ChatGoogleGenerativeAI,
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

	Raises:
		ValueError:
			If the question is empty.

		RuntimeError:
			If answer generation fails.
	"""

	if not question.strip():
		raise ValueError(
			"Question cannot be empty."
		)

	try:
		documents: list[Document] = retriever.invoke(
			question
		)

		context = "\n\n---\n\n".join(
			document.page_content
			for document in documents
			if document.page_content.strip()
		)

		prompt = get_rag_prompt()

		formatted_prompt = prompt.invoke(
			{
				"context": context,
				"question": question,
			}
		)

		response = llm.invoke(
			formatted_prompt
		)

		return extract_response_text(response)

	except Exception as error:
		raise RuntimeError(
			f"Failed to generate answer: {error}"
		) from error


def generate_answer_with_sources(
	question: str,
	retriever: VectorStoreRetriever,
	llm: ChatGoogleGenerativeAI,
) -> RAGResult:
	"""
	Generate an answer using the RAG pipeline
	and return the retrieved source documents.

	Args:
		question:
			User question.

		retriever:
			Configured LangChain retriever.

		llm:
			Configured Gemini model.

	Returns:
		RAGResult containing:
		- Generated answer
		- Retrieved source documents

	Raises:
		ValueError:
			If the question is empty.

		RuntimeError:
			If answer generation fails.
	"""

	if not question.strip():
		raise ValueError(
			"Question cannot be empty."
		)

	try:
		documents: list[Document] = retriever.invoke(
			question
		)

		context = "\n\n---\n\n".join(
			document.page_content
			for document in documents
			if document.page_content.strip()
		)

		prompt = get_rag_prompt()

		formatted_prompt = prompt.invoke(
			{
				"context": context,
				"question": question,
			}
		)

		response = llm.invoke(
			formatted_prompt
		)

		return {
			"answer": extract_response_text(response),
			"documents": documents,
		}

	except Exception as error:
		raise RuntimeError(
			f"Failed to generate answer: {error}"
		) from error


def extract_response_text(response: AIMessage) -> str:
	"""
	Extract plain text from a LangChain AI response.

	Args:
		response:
			Response returned by the LLM.

	Returns:
		Plain text answer.
	"""

	content = response.content

	if isinstance(content, str):
		return content

	if isinstance(content, list):
		text_parts = []

		for block in content:
			if isinstance(block, dict) and block.get("type") == "text":
				text = block.get("text")

				if isinstance(text, str):
					text_parts.append(text)

		return "".join(text_parts)

	return str(content)