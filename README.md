# PDF RAG Assistant with Source Citations

A Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents, ask questions about their content, and receive answers with source citations.

The application uses semantic search to retrieve relevant PDF chunks and a Large Language Model (LLM) to generate answers based on the retrieved context.

---

## Project Overview

The PDF RAG Assistant follows a complete Retrieval Augmented Generation pipeline:

$$
\begin{gathered}
\text{PDF Upload} \\
\downarrow \\
\text{PDF Text Extraction} \\
\downarrow \\
\text{Document Chunking} \\
\downarrow \\
\text{Text Embeddings} \\
\downarrow \\
\text{FAISS Vector Store} \\
\downarrow \\
\text{Similarity Retrieval} \\
\downarrow \\
\text{Relevant Context} \\
\downarrow \\
\text{Gemini LLM} \\
\downarrow \\
\text{Answer + Source Citations}
\end{gathered}
$$

The application is designed to provide answers grounded in the uploaded PDF rather than relying only on the LLM's general knowledge.

---

## Key Features

- Upload PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate semantic embeddings
- Store embeddings using FAISS
- Retrieve the most relevant document chunks
- Generate answers using Gemini LLM
- Return source page and chunk information
- Avoid unnecessary PDF re-indexing
- Persist FAISS indexes across application restarts
- Restore index metadata after restart
- Validate uploaded files
- Enforce maximum PDF upload size
- Centralized exception handling
- Structured application logging
- FastAPI REST API
- Local deployment support

---

## Architecture

The application consists of the following major components:

### 1. PDF Upload

The client uploads a PDF through the `/upload` endpoint.

The API:

- Validates the file type
- Validates the file size
- Sanitizes the filename
- Saves the PDF locally

### 2. PDF Indexing

The `/index` endpoint creates the searchable representation of the uploaded PDF.

The indexing pipeline is:

$$
\begin{gathered}
\text{PDF} \\
\downarrow \\
\text{Text Extraction} \\
\downarrow \\
\text{Chunking} \\
\downarrow \\
\text{Embedding Generation} \\
\downarrow \\
\text{FAISS Index} \\
\downarrow \\
\text{Persistent Storage}
\end{gathered}
$$

### 3. Retrieval

When a user asks a question, the question is converted into an embedding and compared against the vectors stored in FAISS.

The most relevant chunks are retrieved based on semantic similarity.

### 4. Answer Generation

The retrieved chunks are provided as context to the Gemini LLM.

The LLM generates an answer using the retrieved document context.

### 5. Source Citations

The API returns source information along with the generated answer.

Each source contains:

- PDF page number
- Chunk ID

---

## RAG Pipeline

$$
\begin{gathered}
\text{PDF Upload} \\
\downarrow \\
\text{Text Extraction} \\
\downarrow \\
\text{Document Chunking} \\
\downarrow \\
\text{Text Embeddings} \\
\downarrow \\
\text{FAISS Vector Store} \\
\downarrow \\
\text{User Question} \\
\downarrow \\
\text{Semantic Retrieval} \\
\downarrow \\
\text{Relevant Chunks} \\
\downarrow \\
\text{Gemini LLM} \\
\downarrow \\
\text{Answer + Source Citations}
\end{gathered}
$$

---

## Tech Stack

- Python
- FastAPI
- LangChain
- FAISS
- HuggingFace Embeddings
- Gemini LLM
- pypdf
- Pydantic
- Uvicorn

---

## Models

### Embedding Model

`BAAI/bge-small-en-v1.5`

The embedding model converts text into numerical vectors that represent semantic meaning.

These vectors are stored in FAISS for similarity-based retrieval.

### LLM

`Gemini`

The Gemini model generates the final answer using the retrieved PDF context.

### Evaluation Model

`sentence-transformers/all-MiniLM-L6-v2`

This model is used for evaluating semantic similarity between expected and generated answers.

---

## Installation

### 1. Clone the Repository

```bash
git clone repository-url
cd pdf-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment.

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```bash
GOOGLE_API_KEY=your_google_api_key
```

The API key is required for Gemini LLM access.

---

## Running the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn api:app --reload
```

The API will be available at:

http://127.0.0.1:8000

FastAPI also provides interactive API documentation at:

http://127.0.0.1:8000/docs

---

## API Usage

The application exposes the following main endpoints.

### Health Check

`GET /health`

Checks whether the API is running.

Example response:

```json
{
    "status": "healthy",
    "message": "PDF RAG Assistant API is running."
}
```

---

### Upload PDF

`POST /upload`

Uploads a PDF document.

The endpoint validates:

- File type
- File size
- Empty files

Example request:

curl -X POST \
  -F "file=@document.pdf" \
  http://127.0.0.1:8000/upload

Example response:

```json
{
    "status": "success",
    "message": "PDF uploaded successfully.",
    "filename": "document.pdf"
}
```

---

### Create FAISS Index

`POST /index`

Creates a FAISS index for an uploaded PDF.

Example request:

```json
{
    "filename": "document.pdf"
}
```

Example response:

```json
{
    "status": "success",
    "message": "FAISS index created successfully.",
    "filename": "document.pdf",
    "pages": 10,
    "chunks": 42
}
```

If the PDF has already been indexed and has not changed, re-indexing is skipped.

---

### Query PDF

`POST /query`

Ask a question about the indexed PDF.

Example request:

```json
{
    "question": "What is the main topic discussed in the document?"
}
```

Example response:

```json
{
    "status": "success",
    "question": "What is the main topic discussed in the document?",
    "answer": "The document discusses...",
    "sources": [
        {
            "page": 3,
            "chunk_id": 7
        },
        {
            "page": 4,
            "chunk_id": 8
        }
    ]
}
```

---

## Index Persistence

The FAISS vector store is persisted locally so that the application does not need to rebuild the index every time the server restarts.

The persisted files are stored in:

`vectorstore/faiss_index/`

The directory contains the FAISS index and associated metadata.

The application restores the existing index during startup.

---

## Avoiding Unnecessary Re-indexing

The application stores metadata about the currently indexed PDF.

The metadata includes:

- Filename
- File modification time
- Total page count
- Number of chunks

Before creating a new index, the application checks whether the requested PDF matches the previously indexed PDF and whether its modification time is unchanged.

If both conditions match, re-indexing is skipped.

$$
\begin{gathered}
\text{Same PDF} \\
\text{+} \\
\text{Same modification time} \\
\text{+} \\
\text{Existing retriever} \\
\downarrow \\
\text{Skip re-indexing}
\end{gathered}
$$

This prevents unnecessary embedding generation and FAISS index creation.

---

## Error Handling

The application uses centralized exception handling for unexpected errors.

Expected API errors return appropriate HTTP status codes.

Examples include:

- 400 Bad Request
- 404 Not Found
- 413 Request Entity Too Large
- 500 Internal Server Error

Unexpected exceptions are logged with their traceback while clients receive a generic error message.

---

## File Upload Security

The upload endpoint includes several protections:

- Only PDF files are accepted
- Filenames are sanitized using `Path(...).name`
- Maximum upload size is enforced
- Empty files are rejected
- Failed uploads are cleaned up
- Uploaded files are stored inside the configured upload directory

---

## Configuration

Application configuration is centralized in:

config/settings.py

Configuration includes:

- Application name
- Application version
- Upload directory
- FAISS directory
- Logging level
- Chunk size
- Chunk overlap
- Retrieval count
- Embedding model
- LLM model
- Evaluation model
- Environment
- Google API key
- Maximum upload size

Environment-specific secrets are loaded from environment variables or `.env`.

---

## Performance Optimizations

The project includes several performance optimizations:

- Embedding model initialized once during application startup
- LLM client initialized once during application startup
- Existing FAISS index loaded during startup
- Existing index reused across queries
- Unchanged PDFs are not re-indexed
- PDF files are uploaded using chunked reads
- Retrieval is limited to the configured top-k documents
- RAG context is limited to relevant retrieved chunks
- Logging avoids unnecessary large payloads
- Performance testing performed on the complete RAG pipeline

---

## Evaluation

The project includes an evaluation workflow for measuring the quality of generated answers.

The evaluation process compares generated answers with expected answers using semantic similarity.

The evaluation model used is:

`sentence-transformers/all-MiniLM-L6-v2`

Evaluation helps identify how closely the generated response matches the expected answer.

---

## Limitations

The current implementation has the following limitations:

- Local file storage is used for uploaded PDFs
- FAISS indexes are stored locally
- The application currently maintains a single active indexed PDF
- The system is designed primarily for English-language documents
- OCR is not included for scanned/image-only PDFs
- Retrieval quality depends on chunking and embedding quality
- LLM responses depend on the quality of retrieved context
- The application requires a valid Gemini API key
- No authentication or user-management system is included
- The current persistence design is intended for a single application instance
- FAISS persistence is local rather than distributed

---

## Future Improvements

Potential future improvements include:

- Support for multiple PDFs
- Document management and deletion
- Persistent document registry
- OCR support for scanned PDFs
- Improved hybrid retrieval
- Reranking retrieved chunks
- Better citation formatting
- Authentication and authorization
- Multi-user support
- Cloud-based vector storage
- Cloud object storage for PDFs
- Streaming LLM responses
- Background indexing jobs
- Asynchronous indexing pipeline
- Improved evaluation datasets
- Retrieval and answer quality dashboards
- Production monitoring and metrics

---

## Development

The project was developed incrementally with separate stages for:

1. Project setup
2. PDF extraction
3. Document chunking
4. Embedding generation
5. FAISS integration
6. Retrieval
7. RAG chain
8. Evaluation
9. Performance optimization
10. Production-readiness review
11. Documentation
12. Containerization

---

## Project Status

The core PDF RAG Assistant is complete and has gone through:

- Performance optimization
- Security review
- API validation review
- File upload security review
- FAISS persistence and recovery review
- Logging and observability review
- Code quality review
- Dependency review
- API endpoint testing
- Edge-case and failure testing
- Production configuration review
- Final cleanup
- End-to-end production testing

Documentation and deployment preparation are being finalized.

---

## License

This project is intended as an AI/ML engineering portfolio project.