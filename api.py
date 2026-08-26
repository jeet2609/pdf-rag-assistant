"""
PDF RAG Assistant API.

FastAPI application.

Run this using command:
    uvicorn api:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from config.exceptions import unexpected_exception_handler
from config.logging import get_logger
from config.settings import settings
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
from src.vector_store import (
	create_vector_store,
	load_index_metadata,
	load_vector_store,
	save_index_metadata,
	save_vector_store,
)

logger = get_logger(__name__)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""
	Manage application startup and shutdown resources.

	Initializes the embedding model once during application startup
	so it can be reused across multiple indexing requests.
	"""

	logger.info("Initializing application resources.")

	# Initialize embedding model once.
	app.state.embedding_model = get_embedding_model()

	logger.info(
		"Embedding model initialized successfully: %s",
		settings.embedding_model_name,
	)

	# Initialize LLM client once.
	app.state.llm = load_llm()

	logger.info(
		"LLM initialized successfully: %s",
		settings.llm_model_name,
	)

	app.state.vector_store = None
	app.state.retriever = None

	app.state.indexed_filename = None
	app.state.indexed_modified_time = None
	app.state.indexed_metadata = None
	app.state.indexed_chunks_count = None

	faiss_index_path = (
		settings.vector_store_directory / "index.faiss"
	)

	index_metadata_path = (
		settings.vector_store_directory / "metadata.json"
	)

	# Load existing FAISS index if available.
	if faiss_index_path.exists() and index_metadata_path.exists():

		app.state.vector_store = load_vector_store(
			embedding_model=app.state.embedding_model,
		)

		app.state.retriever = create_retriever(
			vector_store=app.state.vector_store,
		)

		index_metadata = load_index_metadata()

		app.state.indexed_filename = index_metadata["filename"]
		app.state.indexed_modified_time = index_metadata["modified_time"]
		app.state.indexed_metadata = {
			"total_page": index_metadata["total_page"],
		}
		app.state.indexed_chunks_count = index_metadata["chunks"]

		logger.info(
			"Existing FAISS vector store and metadata "
			"loaded successfully."
		)

	elif faiss_index_path.exists() or index_metadata_path.exists():

		logger.warning(
			"Incomplete FAISS persistence detected. "
			"Removing incomplete index files."
		)

		faiss_index_path.unlink(missing_ok=True)
		index_metadata_path.unlink(missing_ok=True)

		app.state.vector_store = None
		app.state.retriever = None

	else:
		
		app.state.vector_store = None
		app.state.retriever = None

		logger.info(
			"No existing FAISS vector store found."
		)

	yield

	logger.info("Application shutdown completed.")


app = FastAPI(
	title=settings.app_name,
	description=(
		"REST API for the PDF-based "
		"Retrieval Augmented Generation assistant."
	),
	version=settings.app_version,
	lifespan=lifespan,
)

app.add_exception_handler(
	Exception,
	unexpected_exception_handler,
)

logger.info(
	"Starting %s version %s",
	settings.app_name,
	settings.app_version,
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get(
	"/health",
	status_code=status.HTTP_200_OK,
	response_model=HealthResponse,
)
def health_check() -> HealthResponse:
	"""
	Check whether the API is running.

	Returns:
		API health status.
	"""

	logger.debug("Health check requested.")

	return HealthResponse(
		status="healthy",
		message="PDF RAG Assistant API is running.",
	)


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

@app.post(
	"/upload",
	status_code=status.HTTP_201_CREATED,
	response_model=UploadResponse,
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
			If the uploaded file is not a PDF, is empty,
			or cannot be saved.
	"""

	logger.info(
		"PDF upload requested: %s",
		file.filename,
	)

	if file.content_type != "application/pdf":
		logger.warning(
			"Rejected non-PDF upload: %s",
			file.filename,
		)

		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Only PDF files are supported.",
		)

	safe_filename = Path(file.filename).name
	file_path = settings.upload_directory / safe_filename

	try:

		bytes_written = 0

		with file_path.open("wb") as buffer:
			while chunk := await file.read(1024 * 1024):
				buffer.write(chunk)
				bytes_written += len(chunk)

		if bytes_written == 0:
			file_path.unlink(missing_ok=True)

			logger.warning(
				"Rejected empty PDF upload: %s",
				safe_filename,
			)

			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Uploaded PDF is empty.",
			)

	except OSError as error:
		logger.exception(
			"Failed to save uploaded PDF: %s",
			safe_filename,
		)

		file_path.unlink(missing_ok=True)

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to save the uploaded PDF.",
		) from error

	finally:
		await file.close()

	logger.info(
		"PDF uploaded successfully: %s",
		safe_filename,
	)

	return UploadResponse(
		status="success",
		message="PDF uploaded successfully.",
		filename=safe_filename,
	)


# --------------------------------------------------
# FAISS Index Creation
# --------------------------------------------------

@app.post(
	"/index",
	status_code=status.HTTP_201_CREATED,
	response_model=IndexResponse,
)
def create_index(
	request: IndexRequest,
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

	logger.info(
		"FAISS index creation requested for: %s",
		request.filename,
	)

	safe_filename = Path(request.filename).name

	if not safe_filename.lower().endswith(".pdf"):
		logger.warning(
			"Rejected indexing request for non-PDF file: %s",
			safe_filename,
		)

		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Only PDF files are supported.",
		)

	pdf_path = settings.upload_directory / safe_filename

	if not pdf_path.exists():
		logger.warning(
			"PDF not found for indexing: %s",
			safe_filename,
		)

		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=(
				f"PDF '{safe_filename}' was not found. "
				"Upload the PDF first."
			),
		)

	modified_time = pdf_path.stat().st_mtime

	if (
		app.state.indexed_filename == safe_filename
		and app.state.indexed_modified_time == modified_time
		and app.state.retriever is not None
	):
		logger.info(
			"PDF '%s' is already indexed and unchanged. "
			"Skipping re-indexing.",
			safe_filename,
		)

		return IndexResponse(
			status="success",
			message="PDF is already indexed. Re-indexing skipped.",
			filename=safe_filename,
			pages=app.state.indexed_metadata["total_page"],
			chunks=app.state.indexed_chunks_count,
		)

	try:
		logger.info(
			"Loading PDF: %s",
			safe_filename,
		)

		documents = load_pdf(pdf_path)

		logger.info(
			"PDF loaded successfully: %s",
			safe_filename,
		)

		metadata = get_pdf_metadata(documents)

		logger.info(
			"PDF metadata extracted: %s pages",
			metadata["total_page"],
		)

		chunks = split_document(documents)

		logger.info(
			"Document split into %s chunks.",
			len(chunks),
		)

		embedding_model = app.state.embedding_model

		app.state.vector_store = create_vector_store(
			documents=chunks,
			embedding_model=embedding_model,
		)

		logger.info(
			"FAISS vector store created successfully.",
		)

		save_vector_store(
			vector_store=app.state.vector_store
		)

		save_index_metadata(
			filename=safe_filename,
			modified_time=modified_time,
			total_page=metadata["total_page"],
			chunks=len(chunks),
		)

		logger.info(
			"FAISS vector store saved successfully: %s",
			settings.vector_store_directory,
		)

		app.state.retriever = create_retriever(
			vector_store=app.state.vector_store,
		)

		logger.info(
			"Retriever created successfully.",
		)

	except Exception:
		logger.exception(
			"Failed to create FAISS index for: %s",
			safe_filename,
		)

		# Reset partially created RAG state.
		app.state.vector_store = None
		app.state.retriever = None

		# Let the global exception handler generate
		# the final 500 response.
		raise

	logger.info(
		"FAISS index creation completed successfully: %s",
		safe_filename,
	)

	app.state.indexed_filename = safe_filename
	app.state.indexed_modified_time = modified_time
	app.state.indexed_metadata = metadata
	app.state.indexed_chunks_count = len(chunks)

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
	request: QueryRequest,
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
			If no FAISS index exists or the query is invalid.
	"""

	logger.info("PDF query requested.")

	if not request.question.strip():
		logger.warning("Rejected empty PDF query.")

		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Question cannot be empty",
		)

	retriever = app.state.retriever
	
	if retriever is None:
		logger.warning(
			"Query rejected because RAG pipeline is not initialized.",
		)

		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=(
				"No FAISS index is available. "
				"Upload a PDF and create an index first."
			),
		)

	logger.info(
		"Generating answer using the RAG pipeline.",
	)

	llm = app.state.llm

	result = generate_answer_with_sources(
		question=request.question,
		retriever=retriever,
		llm=llm,
	)

	sources = [
		{
			"page": document.metadata.get("page", 0) + 1,
			"chunk_id": document.metadata.get("chunk_id"),
		}
		for document in result["documents"]
	]

	logger.info(
		"Answer generated successfully with %s source(s).",
		len(sources),
	)

	return QueryResponse(
		status="success",
		question=request.question,
		answer=result["answer"],
		sources=sources,
	)