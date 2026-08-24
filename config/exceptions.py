"""
Application exception handlers.

Provides centralized handling for unexpected exceptions
raised by the FastAPI application.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from config.logging import get_logger

logger = get_logger(__name__)

async def unexpected_exception_handler(
	request: Request,
	error: Exception,
) -> JSONResponse:
	"""
	Handle unexpected application exceptions.

	Unexpected exceptions are logged with their full traceback,
	while a generic error response is returned to the client
	to avoid exposing internal implementation details.

	Args:
		request:
			Incoming HTTP request.

		error:
			Unexpected exception raised by the application.

	Returns:
		JSON response containing a generic error message.
	"""

	logger.exception(
		"Unexpected error while processing %s %s",
		request.method,
		request.url.path,
	)

	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		content={
			"status": "error",
			"message": "An unexpected error occurred.",
		},
	)