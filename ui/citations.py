"""
Source Citations UI

Responsibilities:
- Display source citations for the generated answer
"""

import streamlit as st


def render_citations() -> None:
	"""
	Render the source citations section.
	"""

	st.divider()

	st.subheader("Source Citations")

	if st.session_state.result is None:
		return

	for index, document in enumerate(
		st.session_state.result["documents"],
		start=1,
	):

		metadata = document.metadata

		with st.expander(
			f"Source {index}",
			expanded=False,
		):

			st.markdown(
				f"""
					**Page:** {metadata['page'] + 1} \\
					**Chunk ID:** {metadata['chunk_id']}
				"""
			)

			with st.expander(
				"Chunk data"
			):
				# st.markdown("**Retrieved Chunk**")
				st.write(document.page_content)