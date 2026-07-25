# Project Setup - PDF RAG Assistant with Source Citations

## Overview

This document covers the initial setup of the PDF RAG Assistant project.

The goal of this project is to build a production-style Retrieval Augmented Generation (RAG) application that can answer questions from PDF documents while providing source citations.

The application will allow users to:

- Upload PDF documents
- Extract text from PDFs
- Split documents into meaningful chunks
- Generate embeddings
- Store vectors for similarity search
- Retrieve relevant information
- Generate grounded answers using an LLM
- Display source page numbers and chunk evidence

---

# Project Objective

Build a local PDF Question Answering Assistant using Retrieval Augmented Generation.

The system should answer questions based only on the uploaded PDF content and provide evidence for every answer.

Example:

User:

```
What architecture does BERT use?
```

System:

```
Answer:
BERT uses a Transformer-based architecture.

Sources:
Page: 1
Chunk ID: 12

Context:
"BERT is designed using bidirectional Transformer..."
```

---

# RAG Pipeline Architecture

The complete pipeline:

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
\text{Top-k Relevant Chunks}
\\
\downarrow
\\
\text{Prompt Construction}
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

---

# Technology Stack

## Programming Language

### Python 3.12

Reason:

- Current development environment
- Modern Python version
- Good support for AI/ML libraries
- Strong ecosystem support

---

# LLM

## Gemini API

Used for:

- Answer generation
- Context understanding
- Natural language responses


Reason:

- Strong reasoning capability
- Fast response generation
- Easy API integration
- Suitable for RAG applications

---

# LLM Framework

## LangChain

Used for:

- Document loading
- Text splitting
- Prompt templates
- Retriever creation
- RAG pipeline orchestration


LangChain helps connect different components:

$$
\begin{gathered}
\text{Documents}
\\
\downarrow
\\
\text{Retriever}
\\
\downarrow
\\
\text{Prompt}
\\
\downarrow
\\
\text{LLM}
\\
\downarrow
\\
\text{Response}
\end{gathered}
$$

---

# Embedding Model

## HuggingFace BGE Embeddings

Model:

```
BAAI/bge-small-en-v1.5
```

Purpose:

Convert text chunks into numerical vectors.

Example:

Text:

```
BERT is a transformer-based model.
```

$$
\downarrow
$$

Embedding:

```
[0.23, -0.15, 0.87, ...]
```

---

## Why HuggingFace Embeddings?

Advantages:

- Open source
- Runs locally
- No embedding API cost
- Better portability
- Commonly used in RAG systems

---

# Vector Database

## FAISS

FAISS (Facebook AI Similarity Search) is used for storing and searching document embeddings.

Purpose:

Find chunks that are semantically similar to the user's question.

Example:

Question:

```
How does BERT work?
```

Converted into vector:

```
Question Embedding
```

FAISS searches:

$$
\begin{gathered}
\text{Stored Document Embeddings}
\\
\downarrow
\\
\text{Similarity Search}
\\
\downarrow
\\
\text{Top matching chunks}
\end{gathered}
$$

---

## Why FAISS?

Reasons:

- Required by project specification
- Fast similarity search
- Lightweight
- Industry-recognized vector search library
- Helps understand retrieval fundamentals

---

# User Interface

## Streamlit

Used for building the local application interface.

Features:

- Upload PDF
- Enter questions
- Display answers
- Show citations
- Display retrieved context

---

# API Layer

## FastAPI

Used for creating backend APIs.

Responsibilities:

- Handle PDF upload requests
- Process user queries
- Connect frontend with RAG pipeline

---

# Development Environment

## Virtual Environment

A project-level virtual environment is used.

Structure:

```
pdf-rag-assistant/
├── .venv/
├── src/
├── notebooks/
└── README.md
```

---

## Creating Virtual Environment

Command:

```bash
python -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

# Why Use Virtual Environment?

Virtual environments isolate project dependencies.

Without virtual environments:

```
System Python
├── Project A dependencies
├── Project B dependencies
└── Version conflicts
```

With virtual environment:

$$
\begin{gathered}
\text{pdf-rag-assistant}
\\
\downarrow
\\
\text{.venv}
\\
\downarrow
\\
\text{Project-specific dependencies}
\end{gathered}
$$

Benefits:

- Dependency isolation
- Reproducibility
- Easier deployment
- Cleaner development

---

# Git Workflow

This project follows a feature branch workflow.

Main branch:

```
main
```

Feature branches:

```
project-setup

pdf-extraction

text-chunking

embeddings

faiss-integration

retrieval

rag-pipeline

source-citations

evaluation

streamlit-ui

fastapi

docker-deployment
```

---

# Development Flow

Each feature follows:

$$
\begin{gathered}
\text{Create Branch}
\\
\downarrow
\\
\text{Implement Feature}
\\
\downarrow
\\
\text{Test}
\\
\downarrow
\\
\text{Commit Changes}
\\
\downarrow
\\
\text{Push Branch}
\\
\downarrow
\\
\text{Create Pull Request}
\\
\downarrow
\\
\text{Merge into Main}
\end{gathered}
$$

Example:

$$
\begin{gathered}
\text{main}
\\
\downarrow
\\
\text{pdf-extraction}
\\
\downarrow
\\
\text{Implement PDF loader}
\\
\downarrow
\\
\text{Pull Request}
\\
\downarrow
\\
\text{Merge}
\end{gathered}
$$

---

# Project Structure

Final project structure:

```
pdf-rag-assistant/

│
├── app.py
├── api.py
├── README.md
├── requirements.txt
├── .env.example
│
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── prompts.py
│   ├── rag_chain.py
│   └── evaluation.py
│
├── data/
│   └── pdfs/
│
├── vectorstore/
│   └── faiss_index/
│
├── notebooks/
│
├── docs/
│
├── reports/
│
└── screenshots/
```

---

# Folder Explanation

## src/

Contains reusable production code.

Example:

```
src/document_loader.py
```

Handles PDF extraction.

```
src/retrieval.py
```

Handles FAISS similarity search.

---

## notebooks/

Contains experiments and testing.

Examples:

```
01_pdf_extraction.ipynb
02_text_chunking.ipynb
03_embeddings.ipynb
```

Used for:

- Understanding concepts
- Testing implementations
- Debugging

---

## docs/

Contains project learning notes.

Purpose:

- Revision
- Interview preparation
- Technical documentation

Example:

```
docs/

01_project_setup.md
02_pdf_extraction.md
03_text_chunking.md
```

---

## data/

Stores input PDF files.

Example:

```
data/
└── pdfs/
    └── Bert_Model.pdf
```

---

## vectorstore/

Stores generated FAISS indexes.

Example:

```
vectorstore/
└── faiss_index/
```

---

# Environment Configuration

Sensitive information is stored using environment variables.

Example:

```
.env
```

Contains:

```
GOOGLE_API_KEY=your_api_key
```

The `.env` file is never committed.

Instead:

```
.env.example
```

is committed.

---

# Dependency Management

Dependencies are maintained in:

```
requirements.txt
```

Example packages:

```
langchain

langchain-community

langchain-google-genai

faiss-cpu

sentence-transformers

streamlit

fastapi
```

---

# Current Project Status

Completed:

✅ Git repository setup  
✅ Feature branch workflow  
✅ Python virtual environment  
✅ Project folder structure  
✅ Documentation structure  
✅ Technology stack selection  

Next implementation phase:

```
PDF Extraction
```

Goal:

Convert PDF files into LangChain Document objects with page metadata