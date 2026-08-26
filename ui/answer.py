"""
Answer UI

Responsibilities:
- Validate user input
- Generate answer using the RAG pipeline
- Store result in Streamlit session state
- Display generated answer
"""

import streamlit as st

from src.rag_chain import generate_answer_with_sources
from ui import constants


def render_answer(
	question: str,
	ask_button: bool,
) -> None:
	"""
	Render the answer section.

	Args:
		question:
			User question.

		ask_button:
			Whether the Ask Question button was clicked.
	"""

	st.divider()

	st.subheader("Answer")

	if not ask_button:
		return

	if st.session_state.vector_store is None:

		st.warning(
			constants.UPLOAD_WARNING
		)
		return

	if not question.strip():

		st.warning(
			constants.QUESTION_WARNING
		)
		return

	try:

		with st.spinner(constants.GENERATING_ANSWER_MESSAGE):

			result = generate_answer_with_sources(
				question=question,
				retriever=st.session_state.retriever,
				llm=st.session_state.llm,
			)

		st.session_state.result = result

		st.success(
			constants.ANSWER_SUCCESS_MESSAGE
		)

		st.markdown(result["answer"])

	except Exception as error:  # noqa: BLE001

		st.error(
			f"Failed to generate answer.\n\n{error}"
		)