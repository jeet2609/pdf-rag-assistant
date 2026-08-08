"""
Sidebar UI

Responsibilities:
- Display project information
- Display technology stack
"""

import streamlit as st


def render_sidebar() -> None:
	"""
	Render the Streamlit sidebar.
	"""

	with st.sidebar:

		st.header("Project Information")

		st.markdown(
			"""
				### Features

				- 📄 Upload PDF
				- ✂️ Text Chunking
				- 🧠 Embedding Generation
				- 🔍 FAISS Similarity Search
				- 🤖 Gemini-powered Answers
				- 📚 Source Citations

				---

				### Model

				**Embedding Model**

				`BAAI/bge-small-en-v1.5`

				**Vector Store**

				`FAISS`

				**LLM**

				`Google Gemini: gemini-3.5-flash`
			"""
		)