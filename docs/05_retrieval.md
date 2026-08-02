# Retrieval - PDF RAG Assistant with Source Citations

## Overview

Retrieval is the fifth step in the Retrieval Augmented Generation (RAG) pipeline.

The purpose of retrieval is to find the most relevant document chunks from the FAISS vector store based on the user's question.

At this stage, **no Large Language Model (LLM) is involved**.

The retriever only performs semantic search and returns the most relevant document chunks.

---

# RAG Pipeline

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
\text{FAISS Vector Store}
\\
\downarrow
\\
\textbf{Retriever}
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

## What is Retrieval?

Retrieval is the process of finding the most semantically similar document chunks for a user's question.

Example:

User Question:

```text
What architecture does BERT use?
```

The retriever searches the FAISS vector store and returns the most relevant chunks instead of scanning the entire PDF.

---

## Why is Retrieval Needed?

Large Language Models have context window limitations.

Passing an entire PDF to an LLM would:

- Increase token usage
- Increase response time
- Increase API cost
- Reduce answer quality
- Increase hallucinations

Instead, only the most relevant chunks are retrieved.

The process:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{Retriever}
\\
\downarrow
\\
\text{Top-k Chunks}
\\
\downarrow
\\
\text{Gemini}
\end{gathered}
$$

This is the foundation of Retrieval Augmented Generation.

---

## FAISS vs Retriever

Many beginners think FAISS and Retriever are the same.

They are different.

### FAISS

Responsibilities:

- Store embeddings
- Perform similarity search
- Return nearest vectors

FAISS is the vector database.

---

### Retriever

Responsibilities:

- Accept a user's question
- Convert the question into an embedding
- Query FAISS
- Return LangChain `Document` objects

The retriever acts as the interface between the application and FAISS.

---

## Retrieval Workflow

The complete retrieval process:

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
\text{LangChain Documents}
\end{gathered}
$$

---

## Similarity Search

Similarity search compares the user's question vector with every stored document vector.

Instead of matching words, it compares meanings.

Example:

Question:

```text
How does BERT work?
```

Stored Chunk:

```text
BERT is based on the Transformer architecture.
```

Even though the wording is different, the embeddings are close together in vector space.

FAISS retrieves the chunk because of semantic similarity.

---

## What is Top-k Retrieval?

The retriever does not return every document.

Instead, it returns only the top **k** most relevant chunks.

Example:

```text
k = 4
```

Output:

```text
Chunk 12

Chunk 28

Chunk 35

Chunk 47
```

These are the four chunks most similar to the user's question.

---

## Why Use Top-k?

Retrieving fewer chunks provides several advantages:

- Faster responses
- Lower token usage
- Reduced prompt size
- Better answer quality
- Less irrelevant context

Choosing an appropriate value of **k** is important.

Very small values may miss useful information.

Very large values may introduce unnecessary context.

---

## Implementation

File:

```text
src/retrieval.py
```

Responsibilities:

- Create retriever
- Retrieve relevant document chunks
- Validate user input
- Return LangChain `Document` objects

---

## Creating the Retriever

Input:

- FAISS vector store

Output:

- LangChain Retriever

Workflow:

$$
\begin{gathered}
\text{FAISS Vector Store}
\\
\downarrow
\\
\text{Similarity Search Configuration}
\\
\downarrow
\\
\text{Retriever}
\end{gathered}
$$

---

## Retrieving Documents

$$
\begin{gathered}
\text{Input:}
\\
(\text{User Question})
\\
\downarrow
\\
\text{Retriever}
\\
\downarrow
\\
\text{Output:}
\\
\text{(List[Document])}
\end{gathered}
$$

Each document contains:

- Chunk content
- Page number
- Source file
- Chunk ID

---

## Why Use `invoke()`?

The retriever is implemented using:

```python
retriever.invoke(question)
```

Modern LangChain follows the **Runnable** interface.

Older versions commonly used:

```python
retriever.get_relevant_documents(question)
```

Using `invoke()` keeps the code aligned with the latest LangChain API.

---

## Example Retrieval

Question:

```text
What architecture does BERT use?
```

Output:

```python
[
    Document(
        page_content="BERT is based on the Transformer architecture...",
        metadata={
            "page": 1,
            "chunk_id": 12,
            "source": "bert.pdf"
        }
    )
]
```

Notice that no answer has been generated.

Only the relevant document chunks have been retrieved.

---

## Notebook Implementation

Notebook:

```text
notebooks/05_retrieval.ipynb
```

Purpose:

- Load FAISS index
- Create retriever
- Ask questions
- Retrieve document chunks
- Verify metadata
- Verify retrieval quality

---

## Production Considerations

### 1. Keep Retrieval Independent

The retriever should only retrieve relevant documents.

It should not generate answers.

Keeping responsibilities separate makes the code easier to maintain and test.

---

### 2. Preserve Metadata

Every retrieved document should retain:

- Source filename
- Page number
- Chunk ID

This information is required for source citation in the next stage.

---

### 3. Tune the Value of k

Choosing the correct value of **k** is important.

Small values:

- Faster
- Less context

Large values:

- More context
- Higher token usage

The optimal value depends on the application and document size.

---

### 4. Validate User Input

Empty questions should not be processed.

Example:

```python
if not question.strip():
    raise ValueError("Question cannot be empty.")
```

Input validation prevents unnecessary processing and improves robustness.

---

## Current Implementation Status

Completed:

✅ PDF extraction

✅ Text chunking

✅ Embedding generation

✅ FAISS integration

✅ Retriever creation

✅ Semantic document retrieval

---

## Output of This Stage

Input:

```text
User Question
```

Output:

```text
Top-k Relevant Document Chunks
```

These retrieved chunks will become the context passed to the LLM.

---

## Key Learnings

Through this stage, we learned:

- What retrieval is
- Why retrieval is required in RAG systems
- Difference between FAISS and a Retriever
- How semantic search works
- What Top-k retrieval means
- Why metadata is preserved
- Why `invoke()` is used in modern LangChain
- Why retrieval and answer generation should remain separate

---

## Next Implementation Phase

```text	
RAG Pipeline
```

Goal:

Combine:

- Retriever
- Prompt Template
- Gemini LLM

to generate grounded answers using the retrieved document chunks while providing accurate source citations.