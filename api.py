"""
PDF RAG Assistant API

FastAPI Application

Run this using command: uvicorn api:app --reload
"""

from fastapi import FastAPI


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
	title="PDF RAG Assistant API",
	description=(
		"REST API for the PDF-based "
		"Retrieval Augmented Generation assistant."
	),
	version="1.0.0"
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, str]:
	"""
		Check whether the API is running.

		Returns:
			API health status.
	"""

	return {
		"status": "healthy",
		"message": "PDF RAG Assistant API is running."
	}