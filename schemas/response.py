"""
API Response Schemas

Pydantic models used for API responses.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	"""
	Response model for the health check endpoint.
	"""

	status: str = Field(
		description="Current API health status.",
	)

	message: str = Field(
		description="Health check message.",
	)


class UploadResponse(BaseModel):
	"""
	Response model for the PDF upload endpoint.
	"""

	status: str = Field(
		description="Upload status.",
	)

	message: str = Field(
		description="Upload result message.",
	)

	filename: str = Field(
		description="Name of the uploaded PDF.",
	)


class IndexResponse(BaseModel):
	"""
	Response model for the FAISS index creation endpoint.
	"""

	status: str = Field(
		description="Index creation status.",
	)

	message: str = Field(
		description="Index creation result message.",
	)

	filename: str = Field(
		description="Name of the indexed PDF.",
	)

	pages: int = Field(
		description="Number of pages in the PDF.",
	)

	chunks: int = Field(
		description="Number of chunks created from the PDF.",
	)


class SourceResponse(BaseModel):
	"""
	Response model for a retrieved source document.
	"""

	page: int = Field(
		description="Page number containing the source chunk.",
	)

	chunk_id: int = Field(
		description="Identifier of the retrieved chunk.",
	)


class QueryResponse(BaseModel):
	"""
	Response model for the question/query endpoint.
	"""

	status: str = Field(
		description="Query processing status.",
	)

	question: str = Field(
		description="Original user question.",
	)

	answer: str = Field(
		description="Generated answer.",
	)

	sources: list[SourceResponse] = Field(
		description="Source chunks used to generate the answer.",
	)