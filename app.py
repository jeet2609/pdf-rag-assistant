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

import tempfile
from pathlib import Path

import streamlit as st
from src.chunking import split_document
from src.document_loader import get_pdf_metadata, load_pdf
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "documents" not in st.session_state:
	st.session_state.documents = None

if "metadata" not in st.session_state:
	st.session_state.metadata = None

if "chunks" not in st.session_state:
	st.session_state.chunks = None

if "embedding_model" not in st.session_state:
	st.session_state.embedding_model = None

if "vector_store" not in st.session_state:
	st.session_state.vector_store = None

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

if uploaded_file is not None:

	# Why use tempfile?
	# Your loader expects a file path: load_pdf(pdf_path)
	# However, st.file_uploader() returns an UploadedFile object in memory, not a path. Writing it to a temporary file bridges that gap without modifying your existing production code.

	with tempfile.NamedTemporaryFile(
		delete=False,
		suffix=".pdf",
	) as temp_file:
		temp_file.write(uploaded_file.getbuffer())
		pdf_path = Path(temp_file.name)

	try:
		with st.spinner("Loading PDF..."):

			documents = load_pdf(pdf_path)
			metadata = get_pdf_metadata(documents)
			chunks = split_document(documents)
			embedding_model = get_embedding_model()
			vector_store = create_vector_store(
				documents=chunks,
				embedding_model=embedding_model
			)

		st.session_state.documents = documents
		st.session_state.metadata = metadata
		st.session_state.chunks = chunks
		st.session_state.embedding_model = embedding_model
		st.session_state.vector_store = vector_store

		st.success("PDF loaded successfully. FAISS index created.")

	except Exception as error:  # noqa: BLE001
		st.error(f"Failed to load PDF. \n{error}")

if st.session_state.metadata is not None and uploaded_file is not None and st.session_state.chunks is not None:
	metadata = st.session_state.metadata

	st.subheader("PDF Information")

	col1, col2, col3 = st.columns(3)

	with col1:
		st.metric(
			"File Name",
			uploaded_file.name
		)

	with col2:
		st.metric(
			"Total Pages",
			metadata["total_page"]
		)

	with col3:
		st.metric(
			"Chunks",
			len(st.session_state.chunks)
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
	type="primary"
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