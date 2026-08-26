"""
Application configuration.

Centralizes application settings and environment variables.
"""

from pathlib import Path

from pydantic import Field
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

	upload_directory: Path = (
		PROJECT_ROOT / "data" / "pdfs"
	)

	vector_store_directory: Path = (
		PROJECT_ROOT / "vectorstore" / "faiss_index"
	)

	# --------------------------------------------------
	# Logging
	# --------------------------------------------------

	log_level: str = "INFO"

	# --------------------------------------------------
	# RAG Configuration
	# --------------------------------------------------

	chunk_size: int = 1000
	chunk_overlap: int = 200
	retrieval_k: int = 4

	embedding_model_name: str = (
		"BAAI/bge-small-en-v1.5"
	)

	llm_model_name: str = "gemini-3.6-flash"

	sentence_transformers_model: str = (
		"sentence-transformers/all-MiniLM-L6-v2"
	)

	# --------------------------------------------------
	# Environment
	# --------------------------------------------------

	environment: str = "development"

	# --------------------------------------------------
	# API Keys
	# --------------------------------------------------

	# Required API key loaded from the GOOGLE_API_KEY environment variable in .env.
	# Pydantic Settings maps GOOGLE_API_KEY to google_api_key because
	# SettingsConfigDict uses case_sensitive=False for environment variable matching.
	# Field(...) marks this setting as required, so application startup fails
	# if GOOGLE_API_KEY is missing.
	google_api_key: str = Field(
		...,
		description="Google Gemini API Key.",
	)

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

settings.vector_store_directory.mkdir(
	parents=True,
	exist_ok=True,
)