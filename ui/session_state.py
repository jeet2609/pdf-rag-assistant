"""
Session State

Responsibilities:
- Initialize Streamlit session state variables
"""

import streamlit as st
from ui.constants import session_state


def initialize_session_state() -> None:
	"""
	Initialize Streamlit session state variables.
	"""

	for key, value in session_state.items():
		if key not in st.session_state:
			st.session_state[key] = value