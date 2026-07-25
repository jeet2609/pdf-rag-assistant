# PDF Extraction - PDF RAG Assistant with Source Citations

## Overview

PDF extraction is the first processing step in the Retrieval Augmented Generation (RAG) pipeline.

The goal of this step is to convert an unstructured PDF document into structured text that can be processed by the next stages of the pipeline.

The application needs to transform:

$$
\begin{gathered}
\text{PDF File}
\\
\downarrow
\\
\text{Extracted Text}
\\
\downarrow
\\
\text{LangChain Document Objects}
\\
\downarrow
\\
\text{Text Chunking}
\\
\downarrow
\\
\text{Embeddings}
\end{gathered}
$$

The extracted document must also preserve metadata such as:

- Source file name
- Page number
- Document information

This metadata will later help the system provide source citations.

---

# Role of PDF Extraction in RAG Pipeline

The complete RAG pipeline:

$$
\begin{gathered}
\text{PDF Upload}
\\
\downarrow
\\
\text{PDF Text Extraction}
\\
\downarrow
\\
\text{Text Chunking}
\\
\downarrow
\\
\text{Embedding Generation}
\\
\downarrow
\\
\text{FAISS Vector Search}
\\
\downarrow
\\
\text{Top-k Retrieved Chunks}
\\
\downarrow
\\
\text{Gemini LLM}
\\
\downarrow
\\
\text{Answer Generation}
\\
\downarrow
\\
\text{Source Citation}
\end{gathered}
$$

PDF extraction is the foundation of the entire pipeline.

If extraction is incorrect:

$$
\begin{gathered}
\text{Wrong Extracted Text}
\\
\downarrow
\\
\text{Wrong Chunks}
\\
\downarrow
\\
\text{Wrong Embeddings}
\\
\downarrow
\\
\text{Incorrect Retrieval}
\\
\downarrow
\\
\text{Incorrect Answer}
\end{gathered}
$$

---

# Why PDF Extraction is Required?

Large Language Models cannot directly understand PDF files.

A PDF contains:

- Text
- Images
- Tables
- Formatting
- Multiple pages
- Metadata

Before an LLM can answer questions, the document needs to be converted into machine-readable text.

Example:

Input:

```
Bert_Model.pdf
```

$$
\downarrow
$$

Extraction:

```
BERT: Pre-training of Deep Bidirectional Transformers...

Abstract:

We introduce a new language representation model...
```

---

# Technology Used

## LangChain PyPDFLoader

Library:

```
langchain-community
```

Loader:

```
PyPDFLoader
```

Installation:

```bash
pip install langchain langchain-community pymupdf
```

---

# Why Use LangChain Document Objects?

LangChain provides a standard document format:

```
Document
```

A Document object contains two important components:

1. page_content
2. metadata

---

# Document Structure

Example:

```python
Document(
    page_content="BERT is a transformer based model...",
    metadata={
        "source": "Bert_Model.pdf",
        "page": 0
    }
)
```

---

# page_content

Contains the extracted text.

Example:

```
BERT is designed using bidirectional Transformer architecture.
```

This text will later be:

$$
\text{Text}
\\
\downarrow
\\
\text{Chunk}
\\
\downarrow
\\
\text{Embedding}
\\
\downarrow
\\
\text{Vector Search}
$$

---

# Metadata

Metadata stores additional information about the document.

Example:

```python
{
    "source": "Bert_Model.pdf",
    "page": 5
}
```

Metadata is important for source citation.

Example:

User Question:

```
What architecture does BERT use?
```

Generated Answer:

```
BERT uses a Transformer-based architecture.

Source:

Page: 1
Chunk ID: 12
```

Without metadata:

```
Answer generated

❌ No evidence location
```

With metadata:

```
Answer generated

✅ Source page
✅ Chunk information
```

---

# Implementation

File:

```
src/document_loader.py
```

Responsibilities:

- Validate PDF file
- Load PDF document
- Extract page-wise text
- Preserve metadata
- Return LangChain Document objects

---

# Extraction Workflow

$$
\begin{gathered}
\text{PDF File}
\\
\downarrow
\\
\text{PyPDFLoader}
\\
\downarrow
\\
\text{Page Extraction}
\\
\downarrow
\\
\text{LangChain Documents}
\\
\downarrow
\\
\text{Text Chunking}
\end{gathered}
$$

---

# File Validation

Before processing the PDF, the system validates:

## 1. File Exists

Example:

```python
if not pdf_path.exists():
    raise FileNotFoundError()
```

Purpose:

Avoid processing invalid paths.

---

## 2. File Type

Only PDF files are accepted.

Example:

```
Allowed:
document.pdf ✅

Rejected:
document.txt ❌
```

---

# Error Handling

Production applications should handle failures properly.

Example:

```python
try:
    documents = loader.load()

except Exception as error:
    raise Exception(
        "Failed to load PDF"
    )
```

Benefits:

- Easier debugging
- Better error messages
- More reliable application

---

# Page-wise Extraction

The PDF is extracted page by page.

Example:

Input:

```
Bert_Model.pdf

16 pages
```

Output:

```
Document 1 → Page 0

Document 2 → Page 1

Document 3 → Page 2

...

Document 16 → Page 15
```

Each page becomes an independent Document object.

---

# Why Page-wise Extraction?

Page-level separation helps with:

## Source Citation

Example:

```
Answer:
Attention mechanism is explained in BERT.

Source:
Page 6
```

---

## Better Retrieval

The retriever can identify:

$$
\text{Question}
\\
\downarrow
\\
\text{Relevant Page}
\\
\downarrow
\\
\text{Relevant Chunk}
$$

---

# Notebook Implementation

Notebook:

```
notebooks/01_pdf_extraction.ipynb
```

Purpose:

- Test PDF loading
- Inspect extracted text
- Verify metadata

---

# Loading PDF

Example:

```python
documents = load_pdf(
    "../data/pdfs/Bert_Model.pdf"
)
```

---

# Check Number of Pages

```python
len(documents)
```

Example output:

```
16
```

Meaning:

$$
\begin{gathered}
\text{PDF pages}
\\
\text{(16)}
\\
\downarrow
\\
\text{LangChain Documents}
\\
\text{(16)}
\end{gathered}
$$

---

# Inspect First Page

```python
documents[0]
```

Example output:

```python
Document(
    page_content="BERT: Pre-training of Deep Bidirectional...",
    metadata={
        "source":"Bert_Model.pdf",
        "page":0
    }
)
```

---

# Testing Extracted Text

Example:

```python
print(
    documents[0].page_content[:500]
)
```

Output:

```
BERT: Pre-training of Deep Bidirectional Transformers...

Abstract:

We introduce a new language representation model...
```

---

# Production Considerations

## 1. Preserve Metadata

Never remove metadata during processing.

The metadata flow:

$$
\begin{gathered}
\text{PDF}
\\
\downarrow
\\
\text{Page Metadata}
\\
\downarrow
\\
\text{Chunk Metadata}
\\
\downarrow
\\
\text{FAISS Retrieval}
\\
\downarrow
\\
\text{Source Citation}
\end{gathered}
$$

---

## 2. Handle Large PDFs

Large PDFs may contain:

- Hundreds of pages
- Millions of characters

Production systems should consider:

- Streaming
- Batch processing
- File size limits
- Processing status updates

---

## 3. Handle Scanned PDFs

Some PDFs contain only images.

Example:

$$
\begin{gathered}
\text{Scanned Book.pdf}
\\
\downarrow
\\
\text{No extractable text}
\end{gathered}
$$

Solution:

Use OCR:

- Tesseract
- Vision models
- Document AI tools

---

# Current Implementation Status

Completed:

✅ PDF loading  
✅ Page-wise extraction  
✅ LangChain Document creation  
✅ Metadata preservation  
✅ PDF validation  
✅ Error handling  

---

# Output of This Stage

Input:

```
Bert_Model.pdf
```

Output:

```py
[
Document(
 page_content="...",
 metadata={
    source:"Bert_Model.pdf",
    page:0
 }
),

Document(
 page_content="...",
 metadata={
    source:"Bert_Model.pdf",
    page:1
 }
)
]
```

These Document objects are passed to the next stage:

```
Text Chunking
```

---

# Key Learnings

Through this stage, we learned:

- How PDF files are processed in RAG systems
- Why extraction quality affects final answers
- How LangChain Document objects work
- Why metadata is important for citations
- How page-wise extraction improves traceability

Next implementation phase:

```
Text Chunking
```

Goal:

Split extracted documents into smaller meaningful chunks while preserving metadata.