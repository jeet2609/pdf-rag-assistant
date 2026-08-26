"""
Streamlit UI Constants
"""

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

PAGE_TITLE = "PDF RAG Assistant"

PAGE_ICON = "📄"

LAYOUT = "wide"

# --------------------------------------------------
# Session State
# --------------------------------------------------

session_state = {
	"uploaded_file_id": None,
	"documents": None,
	"metadata": None,
	"chunks": None,
	"embedding_model": None,
	"vector_store": None,
	"retriever": None,
	"llm": None,
	"result": None,
}

# --------------------------------------------------
# Application
# --------------------------------------------------

TOP_K = 4

# --------------------------------------------------
# Messages
# --------------------------------------------------

UPLOAD_PDF_MESSAGE = (
	"Upload a PDF."
)

PDF_SUCCESS_MESSAGE = (
	"PDF processed successfully."
)

ANSWER_SUCCESS_MESSAGE = (
	"Answer generated successfully."
)

UPLOAD_WARNING = (
	"Please upload a PDF first."
)

QUESTION_WARNING = (
	"Please enter a question."
)

PROCESSING_PDF_MESSAGE = (
	"Processing PDF..."
)

GENERATING_ANSWER_MESSAGE = (
	"Generating answer..."
)

ASK_QUESTION_TEXT = (
	"Ask a question"
)

ASK_QUESTION_BUTTON_TEXT = (
	"Ask Question"
)