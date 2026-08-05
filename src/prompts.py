"""
Prompt templates for the RAG pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt() -> ChatPromptTemplate:
	"""
	Create the prompt template for the RAG pipeline.

	The model is instructed to answer questions using only
	the provided context.

	Returns:
		Configured LangChain ChatPromptTemplate.
	"""

	template = """
	You are a helpful AI assistant for answering questions from PDF documents.

	Answer the user's question using ONLY the provided context.

	Instructions:
	- Use only the information from the context.
	- Do not make up information.
	- If the answer is not available in the context, reply: "I couldn't find the answer in the provided document."
	- Keep the answer clear and concise.

	Context:
	{context}

	Question:
	{question}

	Answer:
	"""

	return ChatPromptTemplate.from_template(template)