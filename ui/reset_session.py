import streamlit as st
from ui.constants import session_state


def reset_session_state() -> None:
	"""
		Reset all RAG-related Streamlit session state.
	"""

	for key in session_state:
		st.session_state[key] = None