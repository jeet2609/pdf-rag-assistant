"""
PDF Upload UI

Responsibilities:
- Upload PDF
- Load PDF
- Split into chunks
- Create embeddings
- Build FAISS vector store
- Create retriever
- Load LLM
- Store everything in Streamlit session state
- Display PDF information
"""

import tempfile
from pathlib import Path

import streamlit as st
from src.chunking import split_document
from src.document_loader import (
	get_pdf_metadata,
	load_pdf,
)
from src.embeddings import get_embedding_model
from src.rag_chain import load_llm
from src.retrieval import create_retriever
from src.vector_store import create_vector_store
from ui import constants
from ui.reset_session import reset_session_state


def render_pdf_upload() -> None:
	"""
	Render the PDF upload section.
	"""

	uploaded_file = st.file_uploader(
		label=constants.UPLOAD_PDF_MESSAGE,
		type=["pdf"],
	)

	if uploaded_file is None:
		# If uploaded file is removed reset the session state
		reset_session_state()
		return

	file_id = (
		uploaded_file.name,
		uploaded_file.size
	)

	if st.session_state.uploaded_file_id == file_id:
		return

	# New PDF is uploaded, then reset old session states
	reset_session_state()

	st.session_state.uploaded_file_id = file_id

	# Why use tempfile?
	# Your loader expects a file path: load_pdf(pdf_path)
	# However, st.file_uploader() returns an UploadedFile object in memory, not a path. Writing it to a temporary file bridges that gap without modifying your existing production code.
	with tempfile.NamedTemporaryFile(
		delete=False,
		suffix=".pdf",
	) as temp_file:

		temp_file.write(
			uploaded_file.getbuffer()
		)

		pdf_path = Path(temp_file.name)

	try:

		with st.spinner(constants.PROCESSING_PDF_MESSAGE):

			documents = load_pdf(pdf_path)
			metadata = get_pdf_metadata(documents)
			chunks = split_document(documents)
			embedding_model = (get_embedding_model())
			vector_store = create_vector_store(
				documents=chunks,
				embedding_model=embedding_model,
			)
			retriever = create_retriever(
				vector_store=vector_store,
				k=4,
			)
			llm = load_llm()

		st.session_state.documents = documents
		st.session_state.metadata = metadata
		st.session_state.chunks = chunks
		st.session_state.embedding_model = embedding_model
		st.session_state.vector_store = vector_store
		st.session_state.retriever = retriever
		st.session_state.llm = llm

		st.success(
			constants.PDF_SUCCESS_MESSAGE
		)
		render_pdf_information(
			file_name=uploaded_file.name,
		)

	except Exception as error:  # noqa: BLE001
		st.error(
			f"Failed to process PDF.\n\n{error}"
		)


def render_pdf_information(
	file_name: str,
) -> None:
	"""
	Display uploaded PDF information.

	Args:
		file_name:
			Name of uploaded PDF.
	"""

	if (
		st.session_state.metadata is None
		and st.session_state.chunks is None
	):
		return

	metadata = st.session_state.metadata

	st.subheader("PDF Information")

	col1, col2, col3 = st.columns(3)

	with col1:
		st.metric(
			"File Name",
			file_name,
		)

	with col2:
		st.metric(
			"Total Pages",
			metadata["total_page"],
		)

	with col3:
		st.metric(
			"Chunks",
			len(
				st.session_state.chunks
			),
		)