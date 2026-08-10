"""
Application configuration.

Centralizes application settings and environment variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
	"""
	Application settings.
	"""

	# --------------------------------------------------
	# Application
	# --------------------------------------------------

	app_name: str = "PDF RAG Assistant"
	app_version: str = "1.0.0"

	# --------------------------------------------------
	# File Storage
	# --------------------------------------------------

	upload_directory: Path = Path(
		PROJECT_ROOT / "data" / "pdfs"
	)

	vector_store_directory: Path = (
		PROJECT_ROOT / "vectorstore" / "faiss_index"
	)

	# --------------------------------------------------
	# RAG Configuration
	# --------------------------------------------------

	chunk_size = 1000

	chunk_overlap = 200

	retrievel_k: int = 4

	embedding_model_name: str = (
		"BAAI/bge-small-en-v1.5"
	)

	llm_model_name: str = "gemini-3.5-flash"

	sentence_transformers_model = "sentence-transformers/all-MiniLM-L6-v2"

	# --------------------------------------------------
	# Environment
	# --------------------------------------------------

	environment: str = "development"

	# --------------------------------------------------
	# Pydantic Settings Configuration
	# --------------------------------------------------

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)


# --------------------------------------------------
# Settings Instance
# --------------------------------------------------

settings = Settings()

# Ensure required directories exist.
settings.upload_directory.mkdir(
	parents=True,
	exist_ok=True,
)