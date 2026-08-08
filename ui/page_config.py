"""
Streamlit UI package.

This package contains reusable UI components for the
PDF RAG Assistant application.

Modules:
	session_state		Initialize Streamlit session state
	sidebar				Render sidebar
	pdf_upload			Handle PDF upload and processing
	question			Render question input
	answer				Generate and display answers
	citations			Display source citations
	retrieved_chunks	Display retrieved document chunks
"""

import streamlit as st
from ui import constants


def set_page_config() -> None:
	st.set_page_config(
		page_title=constants.PAGE_TITLE,
		page_icon=constants.PAGE_ICON,
		layout=constants.LAYOUT
	)
