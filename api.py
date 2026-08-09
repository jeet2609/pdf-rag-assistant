"""
PDF RAG Assistant API

FastAPI Application

Run this using command: uvicorn api:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.request import IndexRequest, QueryRequest
from schemas.response import (
	HealthResponse,
	IndexResponse,
	QueryResponse,
	UploadResponse,
)
from src.chunking import split_document
from src.document_loader import get_pdf_metadata, load_pdf
from src.embeddings import get_embedding_model
from src.rag_chain import generate_answer_with_sources, load_llm
from src.retrieval import create_retriever
from src.vector_store import create_vector_store

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
# RAG Application State
# --------------------------------------------------

documents: list | None = None
metadata: dict | None = None
chunks: list | None = None
embedding_model: Embeddings | None = None
vector_store: FAISS | None = None
retriever: VectorStoreRetriever | None = None
llm: ChatGoogleGenerativeAI | None = None


# --------------------------------------------------
# Configuration
# --------------------------------------------------

UPLOAD_DIRECTORY = Path("./data/pdfs")

UPLOAD_DIRECTORY.mkdir(
	parents=True,
	exist_ok=True
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get(
	"/health",
	status_code=status.HTTP_200_OK,
	response_model=HealthResponse
)
def health_check() -> HealthResponse:
	"""
	Check whether the API is running.

	Returns:
		API health status.
	"""

	return HealthResponse(
		status="healthy",
		message="PDF RAG Assistant API is running."
	)


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

@app.post(
	"/upload",
	status_code=status.HTTP_201_CREATED,
	response_model=UploadResponse
)
async def upload_pdf(
	file: UploadFile = File(...)  # noqa: B008	(The ... means no default value — the file is required.)
) -> UploadResponse:
	"""
	Upload a PDF document.

	Args:
		file:
			PDF file uploaded by the client.

	Returns:
		Upload status and saved file information.

	Raises:
		HTTPException:
			If the uploaded file is not a PDF or
			the file cannot be saved.
	"""

	if file.content_type != "application/pdf":
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Only PDF files are supported."
		)
	
	safe_filename = Path(file.filename).name
	file_path = UPLOAD_DIRECTORY / safe_filename

	try:
		file_content = await file.read()

		if not file_content:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Uploaded PDF is empty."
			)

		file_path.write_bytes(file_content)

	except HTTPException:
		raise
		
	except OSError as error:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to save the uploaded PDF."
		) from error

	finally:
		await file.close()

	return UploadResponse(
		status="success",
		message="PDF uploaded successfully.",
		filename=safe_filename
	)


# --------------------------------------------------
# FAISS Index Creation
# --------------------------------------------------

@app.post(
	"/index",
	status_code=status.HTTP_201_CREATED,
	response_model=IndexResponse
)
def create_index(
	request: IndexRequest
) -> IndexResponse:
	"""
	Create a FAISS index for an uploaded PDF.

	Args:
		request:
			Request containing the name of the previously uploaded PDF.

	Returns:
		Information about the created FAISS index.

	Raises:
		HTTPException:
			If the PDF does not exist or indexing fails.
	"""

	# Use global because we assign values to these module-level variables.
	global documents
	global metadata
	global chunks
	global embedding_model
	global vector_store
	global retriever
	global llm

	safe_filename = Path(request.filename).name

	if not safe_filename.lower().endswith(".pdf"):
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Only PDF files are supported.",
		)

	pdf_path = UPLOAD_DIRECTORY / safe_filename

	if not pdf_path.exists():
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"PDF '{safe_filename}' was not found. Upload the PDF first.",
		)

	try:
		documents = load_pdf(pdf_path)
		metadata = get_pdf_metadata(documents)
		chunks = split_document(documents)
		embedding_model = get_embedding_model()
		vector_store = create_vector_store(
			documents=chunks,
			embedding_model=embedding_model,
		)
		retriever = create_retriever(
			vector_store=vector_store,
			k=4,
		)
		llm = load_llm()
		
	except Exception as error:

		# Reset partially created state
		documents = None
		metadata = None
		chunks = None
		embedding_model = None
		vector_store = None
		retriever = None
		llm = None

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to create FAISS index.",
		) from error

	return IndexResponse(
		status="success",
		message="FAISS index created successfully.",
		filename=safe_filename,
		pages=metadata["total_page"],
		chunks=len(chunks),
	)


# --------------------------------------------------
# Question / Query
# --------------------------------------------------

@app.post(
	"/query",
	status_code=status.HTTP_200_OK,
	response_model=QueryResponse,
)
def query_pdf(
	request: QueryRequest
) -> QueryResponse:
	"""
	Ask a question about the indexed PDF.

	Args:
		request:
			Request containing the user's question.

	Returns:
		Generated answer and source documents.

	Raises:
		HTTPException:
			If no FAISS index exists or query generation fails.
	"""

	if not request.question.strip():
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Question cannot be empty",
		)

	if retriever is None or llm is None:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=(
				"No FAISS index is available. "
				"Upload a PDF and create an index first."
			),
		)

	try:

		result = generate_answer_with_sources(
			question=request.question,
			retriever=retriever,
			llm=llm,
		)

	except Exception as error:
		print(f"############### {error} ###############")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to generate an answer.",
		) from error

	sources = []

	for document in result["documents"]:
		sources.append(
			{
				"page": document.metadata.get("page", 0) + 1,
				"chunk_id": document.metadata.get("chunk_id"),
			}
		)

	return QueryResponse(
		status="success",
		question=request.question,
		answer=result["answer"],
		sources=sources,
	)