"""
PDF RAG Assistant

Streamlit User Interface

Commit 1:
- Basic application layout
- Sidebar
- PDF upload widget
- Question input
- Ask button
- Placeholders for answer and citations
"""

import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
	page_title="PDF RAG Assistant",
	page_icon="📄",
	layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📄 PDF RAG Assistant")

st.markdown(
    """
Ask questions about your PDF documents using Retrieval Augmented Generation (RAG).

Upload a PDF, ask questions, and receive grounded answers with source citations.
"""
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

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

			`Google Gemini`
		"""
    )

# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
	label="Upload a PDF",
	type=["pdf"],
)

# --------------------------------------------------
# Question Section
# --------------------------------------------------

question = st.text_input(
	label="Ask a question",
	placeholder="Example: What architecture does BERT use?"
)

# --------------------------------------------------
# Ask Button
# --------------------------------------------------

ask_button = st.button(
	label="Ask Question",
	type="primary",
	use_container_width=True,
)

# --------------------------------------------------
# Answer Placeholder
# --------------------------------------------------

st.divider()

st.subheader("Answer")

if ask_button:
	st.info(
		"RAG pipeline will be connected in the next step."
	)
else:
	st.empty()

# --------------------------------------------------
# Source Citations
# --------------------------------------------------

st.divider()

st.subheader("Source Citations")

st.info(
	"Source will appear here after the RAG pipeline is connected."
)

# --------------------------------------------------
# Retrieved Chunks
# --------------------------------------------------

st.divider()

st.subheader("Retrieved Chunks")

st.info(
	"Retrieved documents chunks will appear here."
)