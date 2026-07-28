# FAISS Integration - PDF RAG Assistant with Source Citations

## Overview

FAISS integration is the fourth step in the Retrieval Augmented Generation (RAG) pipeline.

The purpose of this step is to store document embeddings in a vector index so that relevant information can be retrieved efficiently during question answering.

Until now, we have:

- Extracted text from PDF documents
- Split text into smaller chunks
- Converted each chunk into a numerical embedding

The next step is to store those embeddings inside FAISS.

---

## Role of FAISS in RAG Pipeline

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
\textbf{FAISS Vector Store}
\\
\downarrow
\\
\text{Retriever}
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

## What is FAISS?

FAISS stands for:

```text
Facebook AI Similarity Search
```

It is an open-source library developed by Meta for efficient similarity search over dense vector embeddings.

Unlike traditional databases that search using exact text matching, FAISS searches using vector similarity.

Example:

```text
Question:

How does BERT work?
```

$$
↓
$$

```text
Embedding Model
```

$$
↓
$$

```text
Question Vector
```

$$
↓
$$

FAISS searches millions of stored vectors to find the most semantically similar document chunks.

---

## Why Do We Need FAISS?

Large Language Models cannot search through hundreds or thousands of document chunks efficiently.

Instead, we first retrieve only the most relevant chunks.

The retrieval process:

$$
\begin{gathered}
\text{User Question}
\\
\downarrow
\\
\text{Embedding Model}
\\
\downarrow
\\
\text{Question Vector}
\\
\downarrow
\\
\text{FAISS Similarity Search}
\\
\downarrow
\\
\text{Top-k Relevant Chunks}
\\
\downarrow
\\
\text{Gemini LLM}
\end{gathered}
$$

This greatly reduces:

- Token usage
- Response latency
- Hallucinations

while improving answer quality.

---

## What Does FAISS Store?

FAISS stores numerical vectors.

Example:

```text
Chunk

"BERT is a transformer-based model."
```

$$
↓
$$

```text
Embedding

[0.23, -0.15, 0.81, ...]
```

$$
↓
$$

```text
Stored inside FAISS.
```

FAISS does **not** store the meaning of the text.

It stores only the vector representation and uses distance calculations to find similar vectors.

---

## Why is the Embedding Model Required?

This is one of the most important concepts in a RAG system.

Many beginners assume FAISS generates embeddings.

It does not.

The embedding model is responsible for converting text into vectors.

FAISS is responsible only for storing and searching those vectors.

The workflow:

$$
\begin{gathered}
\text{Text Chunk}
\\
\downarrow
\\
\text{Embedding Model}
\\
\downarrow
\\
\text{Vector}
\\
\downarrow
\\
\text{FAISS}
\end{gathered}
$$

---

## Why is `embedding_model` Needed During Index Creation?

When we create a FAISS index using LangChain:

```python
FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)
```

LangChain internally performs:

```python
vectors = embedding_model.embed_documents(texts)

faiss.store(vectors)
```

Notice that FAISS never creates the embeddings.

It simply stores the vectors produced by the embedding model.

---

## Why is `embedding_model` Needed While Loading the Index?

This is often confusing.

When we load an existing FAISS index:

```python
FAISS.load_local(
    ...,
    embeddings=embedding_model
)
```

the vectors are already stored.

However, users will ask questions using text.

Example:

```text
How does BERT work?
```

FAISS cannot compare text with stored vectors.

The question must first be converted into a vector.

The process:

$$
\begin{gathered}
\text{User Question}
\\
\downarrow
\\
\text{Embedding Model}
\\
\downarrow
\\
\text{Question Vector}
\\
\downarrow
\\
\text{FAISS Search}
\end{gathered}
$$

Therefore, LangChain requires the same embedding model when loading the vector store.

---

## Why Must the Same Embedding Model Be Used?

Suppose the index was built using:

```text
BAAI/bge-small-en-v1.5
```

Later we load it using another model.

The document vectors and question vectors now belong to different vector spaces.

Example:

$$
\begin{gathered}
\text{Chunks}
\\
\downarrow
\\
\text{BGE Embeddings}
\\
\downarrow
\\
\text{Stored Vectors}
\end{gathered}
$$

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{Different Embedding Model}
\\
\downarrow
\\
\text{Different Vector Space}
\end{gathered}
$$

The similarity scores become unreliable, resulting in poor retrieval quality.

For this reason, the same embedding model must always be used for both indexing and querying.

---

## Implementation

File:

```
src/vector_store.py
```

Responsibilities:

- Create FAISS vector store
- Save vector store
- Load vector store
- Create retriever

---

## Vector Store Creation

Input:

- Document chunks
- Embedding model

Process:

$$
\begin{gathered}
\text{Document Chunks}
\\
\downarrow
\\
\text{Embedding Model}
\\
\downarrow
\\
\text{Embeddings}
\\
\downarrow
\\
\text{FAISS Index}
\end{gathered}
$$

---

# Saving the Vector Store

The FAISS index is persisted locally.

Example location:

```text
vectorstore/
└── faiss_index/
```

Typical files:

```
index.faiss
index.pkl
```

Benefits:

- No need to recreate embeddings every time the application starts.
- Faster application startup.
- Supports persistent storage.

---

## Loading the Vector Store

Instead of recreating the index every time:

$$
\begin{gathered}
\text{Application Start}
\\
\downarrow
\\
\text{Load Existing FAISS Index}
\\
\downarrow
\\
\text{Ready for Retrieval}
\end{gathered}
$$

This significantly reduces initialization time.

---

## Retriever Creation

After loading the vector store, we create a retriever.

The retriever performs similarity search over stored vectors.

The workflow:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{Question Embedding}
\\
\downarrow
\\
\text{Retriever}
\\
\downarrow
\\
\text{Top-k Chunks}
\end{gathered}
$$

The retriever will be used in the next stage of the project.

---

## Notebook Implementation

Notebook:

```text
notebooks/04_faiss_integration.ipynb
```

Purpose:

- Create FAISS index
- Save index
- Load index
- Verify stored vectors
- Verify metadata persistence

---

## Production Considerations

### 1. Persist the Index

Never recreate embeddings every time the application starts.

Instead:

$$
\begin{gathered}
\text{Load Existing Index}
\\
\downarrow
\\
\text{Reuse Stored Vectors}
\end{gathered}
$$

---

### 2. Preserve Metadata

Every chunk stored in FAISS should retain:

- Source filename
- Page number
- Chunk ID

This enables accurate source citations later.

---

### 3. Use the Same Embedding Model

Changing the embedding model after indexing can lead to incorrect similarity search results.

Always use the same model for:

- Index creation
- Query embedding
- Retrieval

---

## Current Implementation Status

Completed:

✅ PDF extraction

✅ Text chunking

✅ Embedding generation

✅ FAISS vector store creation

✅ FAISS persistence

✅ FAISS loading

---

## Output of This Stage

Input:

```text
Document Chunks

+

Embedding Model
```

Output:

```text
Persistent FAISS Vector Store
```

Stored on disk for future retrieval.

---

## Key Learnings

Through this stage, we learned:

- What FAISS is and why it is used
- How vector databases differ from traditional databases
- Why embeddings are required
- Why FAISS does not generate embeddings
- Why the same embedding model must be used for indexing and querying
- How to persist and reload a FAISS index
- How FAISS prepares the system for semantic retrieval

Next implementation phase:

```text
Retriever
```

Goal:

Retrieve the most semantically relevant document chunks from the FAISS vector store using a user's question.