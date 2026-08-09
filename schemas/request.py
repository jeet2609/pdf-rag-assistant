"""
API Request Schemas

Pydantic models used for validating incoming API requests.
"""

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
	"""
	Request model for the FAISS index creation endpoint.
	"""

	# Here the '...' means "this field is required; there is no default value."
	filename: str = Field(
		...,
		min_length=1,
		description="Name of the uploaded PDF to index.",
	)

class QueryRequest(BaseModel):
	"""
	Request model for the question/query endpoint.

	Attributes:
		question:
			User's question about the indexed PDF.
	"""

	question: str = Field(
		...,
		min_length=1,
		description="Question to ask about the indexed PDF."
	)