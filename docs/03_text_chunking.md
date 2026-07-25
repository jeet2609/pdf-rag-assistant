# Text Chunking - PDF RAG Assistant with Source Citations

## Overview

Text chunking is the second processing step in the Retrieval Augmented Generation (RAG) pipeline.

The goal of text chunking is to divide large extracted documents into smaller meaningful pieces that can be converted into embeddings and efficiently retrieved during question answering.

After PDF extraction, we have complete page-level documents.

Example:

```
PDF Page

"Large amount of extracted text from research paper..."
```

This cannot be directly used efficiently for retrieval.

Therefore, we split the document into smaller chunks.

The process:

$$
\begin{gathered}
\text{Extracted PDF Documents}
\\
\downarrow
\\
\text{Text Chunking}
\\
\downarrow
\\
\text{Smaller Document Chunks}
\\
\downarrow
\\
\text{Embedding Generation}
\end{gathered}
$$

---

# Role of Text Chunking in RAG Pipeline

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

# Why Do We Need Text Chunking?

Large documents create challenges for Retrieval Augmented Generation systems.

A research paper may contain:

- Multiple pages
- Thousands of words
- Different topics
- Multiple sections

Sending the complete document to the LLM creates problems:

- Higher token usage
- Slower responses
- Less accurate retrieval
- More irrelevant context

Instead:

$$
\begin{gathered}
\text{Large Document}
\\
\downarrow
\\
\text{Small Meaningful Chunks}
\\
\downarrow
\\
\text{Better Embeddings}
\\
\downarrow
\\
\text{Accurate Retrieval}
\end{gathered}
$$

---

# Problems Without Chunking

Example:

Document:

```
BERT Research Paper

Page 1 - Introduction

Page 2 - Related Work

Page 3 - Model Architecture

Page 4 - Experiments
```

Question:

```
What architecture does BERT use?
```

Without chunking:

$$
\begin{gathered}
\text{Entire PDF}
\\
\downarrow
\\
\text{Embedding}
\\
\downarrow
\\
\text{Retrieval}
\end{gathered}
$$

The retrieved information may contain unrelated sections.

---

# Chunking Strategy Used

We use:

## RecursiveCharacterTextSplitter

Provided by:

```
langchain-text-splitters
```

---

# Why RecursiveCharacterTextSplitter?

RecursiveCharacterTextSplitter attempts to split text while preserving semantic meaning.

It uses separators in order:

$$
\begin{gathered}
\text{Paragraph}
\\
\downarrow
\\
\text{Sentence}
\\
\downarrow
\\
\text{Word}
\\
\downarrow
\\
\text{Character}
\end{gathered}
$$

This prevents unnecessary breaking of meaningful text.

---

# Chunking Parameters

The two main parameters are:

## Chunk Size

Example:

```python
chunk_size=1000
```

Meaning:

Each chunk will contain approximately 1000 characters.

Example:

```
Chunk 1:

BERT is a transformer-based model...
(1000 characters)
```

---

## Chunk Overlap

Example:

```python
chunk_overlap=200
```

Meaning:

The next chunk shares 200 characters from the previous chunk.

---

# Why Chunk Overlap is Required?

Without overlap:

```
Chunk 1:
"The attention mechanism allows the model to understand relationships between words"


Chunk 2:
"The model was trained on large datasets"
```

Important context between chunks can be lost.

With overlap:

$$
\begin{gathered}
\text{Chunk 1}
\\
\downarrow
\\
\text{Shared Context}
\\
\downarrow
\\
\text{Chunk 2}
\end{gathered}
$$

This improves retrieval quality.

---

# Chunk Size Trade-off

Chunk size affects retrieval performance.

## Large Chunks

Advantages:

- More context
- Better understanding of long explanations

Disadvantages:

- More irrelevant information
- Higher token usage
- Less precise retrieval


---

## Small Chunks

Advantages:

- More precise retrieval
- Less unnecessary context

Disadvantages:

- May lose important information
- Less semantic meaning

---

# Selected Configuration

Current configuration:

```python
chunk_size=1000

chunk_overlap=200
```

Reason:

- Good starting point for research papers
- Maintains context
- Provides enough information for LLM generation

This value can later be optimized during evaluation.

---

# Implementation

File:

```
src/chunking.py
```

Responsibilities:

- Accept extracted Document objects
- Split documents into smaller chunks
- Preserve metadata
- Add unique chunk IDs

---

# Input

From PDF extraction:

```python
Document(
    page_content="Long extracted PDF text...",
    metadata={
        "source":"Bert_Model.pdf",
        "page":0
    }
)
```

---

# Processing

$$
\begin{gathered}
\text{LangChain Document}
\\
\downarrow
\\
\text{RecursiveCharacterTextSplitter}
\\
\downarrow
\\
\text{Multiple Smaller Documents}
\end{gathered}
$$

---

# Output

After chunking:

```python
Document(
    page_content="Small text chunk...",
    metadata={
        "source":"Bert_Model.pdf",
        "page":0,
        "chunk_id":5
    }
)
```

---

# Chunk ID Generation

Each chunk receives a unique identifier.

Example:

```python
chunk.metadata["chunk_id"] = index
```

Purpose:

- Track retrieved chunks
- Provide source evidence
- Debug retrieval results

Example:

User:

```
How does BERT use attention?
```

Response:

```
Answer:
BERT uses Transformer attention mechanisms.

Source:
Page: 3
Chunk ID: 24
```

---

# Notebook Implementation

Notebook:

```
notebooks/02_text_chunking.ipynb
```

Purpose:

- Test chunking implementation
- Analyze chunk sizes
- Verify metadata preservation

---

# Loading Documents

First load extracted documents:

```python
documents = load_pdf(
    "../data/pdfs/Bert_Model.pdf"
)
```

---

# Creating Chunks

Example:

```python
chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)
```

---

# Checking Chunk Count

Example:

```python
len(documents), len(chunks)
```

Output:

```
(16, 120)
```

Meaning:

```
Original pages:

16

Generated chunks:

120
```

---

# Inspecting a Chunk

Example:

```python
chunks[0]
```

Output:

```python
Document(
    page_content="BERT: Pre-training of Deep Bidirectional...",
    metadata={
        "source":"Bert_Model.pdf",
        "page":0,
        "chunk_id":0
    }
)
```

---

# Chunk Quality Validation

## Check Chunk Length

Example:

```python
len(chunks[0].page_content)
```

Expected:

```
Approximately 1000 characters
```

---

## Check Metadata

Example:

```python
chunks[0].metadata
```

Expected:

```python
{
    "source":"Bert_Model.pdf",
    "page":0,
    "chunk_id":0
}
```

---

# Production Considerations

## 1. Domain-Specific Chunking

Different documents require different chunk strategies.

Examples:

Research papers:

```
1000-1500 characters
```

Legal documents:

```
Smaller chunks with more overlap
```

Technical documentation:

```
Section-based chunking
```

---

## 2. Semantic Chunking

Character-based chunking is a good baseline.

Advanced RAG systems may use:

- Semantic chunking
- Sentence transformers
- Document structure-aware splitting

---

## 3. Chunking Evaluation

Chunking quality affects:

- Retrieval accuracy
- Answer quality
- Hallucination rate

During evaluation, we should analyze:

- Retrieved chunk relevance
- Missing information
- Incorrect retrieval

---

# Metadata Flow

Metadata must be preserved throughout the pipeline.

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
\text{FAISS Storage}
\\
\downarrow
\\
\text{Source Citation}
\end{gathered}
$$

---

# Current Implementation Status

Completed:

✅ PDF extraction  
✅ Page-wise Document creation  
✅ Recursive text splitting  
✅ Chunk overlap implementation  
✅ Metadata preservation  
✅ Chunk ID generation  

---

# Output of This Stage

Input:

```
LangChain Documents from PDF extraction
```

Output:

```
List of smaller Document chunks
```

Example:

```python
[
Document(
    page_content="Chunk 1...",
    metadata={
        "page":0,
        "chunk_id":0
    }
),

Document(
    page_content="Chunk 2...",
    metadata={
        "page":0,
        "chunk_id":1
    }
)
]
```

---

# Key Learnings

Through this stage, we learned:

- Why large documents need chunking
- How chunk size affects retrieval quality
- Why overlap preserves context
- How RecursiveCharacterTextSplitter works
- Why metadata must be maintained
- How chunking impacts final RAG performance

Next implementation phase:

```
Embedding Generation
```

Goal:

Convert text chunks into numerical vector representations using HuggingFace BGE embeddings.