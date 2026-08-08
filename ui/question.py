"""
Question UI

Responsibilities:
- Render question input
- Render Ask Question button
"""

import streamlit as st
from ui import constants


def render_question_section() -> tuple[str, bool]:
	"""
	Render the question input section.

	Returns:
		A tuple containing:
			- User question
			- Whether the Ask Question button was clicked
	"""

	question = st.text_input(
		label="Ask a question",
		placeholder="Example: What architecture does BERT use?"
	)

	ask_button = st.button(
		label=constants.ASK_QUESTION_BUTTON_TEXT,
		type="primary",
	)

	return question, ask_button