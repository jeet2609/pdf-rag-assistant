"""
PDF RAG Assistant API

FastAPI Application

Run this using command: uvicorn api:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, File, UploadFile, status, HTTPException

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
	status_code=status.HTTP_200_OK
)
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


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

@app.post(
	"/upload",
	status_code=status.HTTP_201_CREATED
)
async def upload_pdf(
	file: UploadFile = File(...)  # noqa: B008	(The ... means no default value — the file is required.)
) -> dict[str, str]:
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

	return {
		"status": "success",
		"message": "PDF uploaded successfully.",
		"filename": safe_filename,
		"path": str(file_path)
	}