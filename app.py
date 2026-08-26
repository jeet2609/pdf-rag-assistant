"""
PDF RAG Assistant

Streamlit Application Entry Point

Responsibilities:
- Configure Streamlit page
- Initialize session state
- Render sidebar
- Handle PDF upload
- Render question section
- Generate and display answer
- Display source citations and retrieved chunks
"""

import streamlit as st

from ui.answer import render_answer
from ui.citations import render_citations
from ui.page_config import set_page_config
from ui.pdf_upload import render_pdf_upload
from ui.question import render_question_section
from ui.session_state import initialize_session_state
from ui.sidebar import render_sidebar

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

set_page_config()


# --------------------------------------------------
# Session State
# --------------------------------------------------

initialize_session_state()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

render_sidebar()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📄 PDF RAG Assistant")

st.markdown(
	"""
		Ask questions about your PDF documents using
		Retrieval Augmented Generation (RAG).

		Upload a PDF, ask questions, and receive grounded
		answers with source citations.
	"""
)


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

render_pdf_upload()


# --------------------------------------------------
# Question
# --------------------------------------------------

question, ask_button = render_question_section()


# --------------------------------------------------
# Answer
# --------------------------------------------------

render_answer(
	question=question,
	ask_button=ask_button,
)


# --------------------------------------------------
# Source Citations + Retrieved Chunks
# --------------------------------------------------

render_citations()