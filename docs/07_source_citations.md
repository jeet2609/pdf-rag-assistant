# Source Citations - PDF RAG Assistant with Source Citations

## Overview

In the previous step, we built a complete RAG pipeline that could answer questions using information retrieved from a PDF.

However, the user still has one important question:

> **"How do I know this answer is correct?"**

This is where **Source Citations** become important.

Instead of returning only the answer, the application also returns the document chunks that were used to generate that answer.

This allows users to verify:

- Which PDF was used
- Which page contains the information
- Which chunk was retrieved
- The exact context used by the LLM

This makes the system transparent, trustworthy, and significantly reduces the impact of hallucinations.

---

# What are Source Citations?

A source citation is additional information that accompanies the generated answer.

Instead of only displaying:

```
BERT uses the Transformer architecture.
```

the application also displays:

```
Answer
BERT uses the Transformer architecture.

------------------------------------------------

Source 1
Page : 1
Chunk ID : 12
File : bert.pdf
Context Preview : "BERT is based on the Transformer architecture..."
```

The user can immediately verify where the answer came from.

---

# Why are Source Citations Important?

Without citations:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{LLM}
\\
\downarrow
\\
\text{Answer}
\end{gathered}
$$

Problems:

- User cannot verify the answer.
- Model may hallucinate.
- No evidence is shown.
- Difficult to debug incorrect answers.

---

With citations:

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
\text{Relevant Chunks}
\\
\downarrow
\\
\text{LLM}
\\
\downarrow
\\
\text{Answer}
\\
\downarrow
\\
\text{Source Citations}
\end{gathered}
$$

Advantages:

- Transparent
- Explainable
- Easy to verify
- Easier debugging
- Increased user trust

---

# Citation Workflow

The complete workflow becomes:

$$
\begin{gathered}
\text{User Question}
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
\text{Prompt Construction}
\\
\downarrow
\\
\text{Gemini}
\\
\downarrow
\\
\text{Answer}
\\
\downarrow
\\
\text{Return Retrieved Documents}
\\
\downarrow
\\
\text{Display Citations}
\end{gathered}
$$

---

# Previous Implementation

Previously our function returned only the generated answer.

```python
return response.content
```

This was sufficient for answering questions, but it discarded the retrieved documents.

As a result, we had no way to display:

- Page number
- Chunk ID
- File name
- Context preview

---

# Updated Implementation

Instead of modifying the existing function, we created a new function:

```python
generate_answer_with_sources()
```

This preserves the original implementation while adding new functionality.

The new function returns:

```python
{
    "answer": response.content,
    "documents": documents,
}
```

Now we have both:

- Generated answer
- Retrieved documents

---

# Why Create a New Function?

One implementation decision was whether to update:

```python
generate_answer()
```

or create:

```python
generate_answer_with_sources()
```

We chose the second approach.

Reasons:

- Keeps the original implementation intact.
- Easier to compare both versions.
- Existing notebooks continue to work.
- Better Git history.
- Demonstrates project evolution.

Instead of replacing previous work, we extended it.

This is a common approach in production systems where backward compatibility is important.

---

# Why Return Documents Instead of a Sources List?

Another question during development was:

> Why return

```python
documents
```

instead of

```python
sources
```

The reason is flexibility.

Each LangChain `Document` already contains:

```python
page_content
```

and

```python
metadata
```

The metadata includes:

- page number
- source file
- chunk ID

Nothing is lost by returning the original `Document` objects.

Later, the UI can decide how to display the information.

---

# Understanding LangChain Documents

Each retrieved item is a `Document` object.

Example:

```python
Document(
    page_content="BERT is based on the Transformer architecture...",
    metadata={
        "page": 1,
        "chunk_id": 12,
        "source": "bert.pdf"
    }
)
```

This object contains everything needed for citations.

---

# Available Metadata

Every retrieved document contains useful metadata.

Example:

```python
document.metadata["page"]
```

Returns:

```
1
```

---

```python
document.metadata["chunk_id"]
```

Returns:

```
12
```

---

```python
document.metadata["source"]
```

Returns:

```
bert.pdf
```

---

```python
document.page_content
```

Returns:

```
BERT is based on the Transformer architecture...
```

These fields are later displayed in the application.

---

# Context Preview

Displaying an entire chunk can overwhelm the user.

Instead, we show only a short preview.

Example:

```python
preview = document.page_content[:300]
```

Output:

```
BERT is based on the Transformer architecture and is designed using...
```

This gives enough context while keeping the interface clean.

---

# Example Output

```
Question: What architecture does BERT use?

------------------------------------------------

Answer: BERT uses the Transformer architecture.

------------------------------------------------

Source
Page : 1
Chunk ID : 12
Source : bert.pdf
Context Preview : BERT is based on the Transformer architecture...
```

This format is much more useful than displaying only the answer.

---

# Why Context Preview Instead of Full Chunk?

Imagine each chunk contains 1000 characters.

Displaying all retrieved chunks would create a very long response.

Instead:

$$
\begin{gathered}
\text{Retrieved Chunk}
\\
\downarrow
\\
\text{First 300 Characters}
\\
\downarrow
\\
\text{Context Preview}
\end{gathered}
$$

The user gets enough information to verify the answer without reading unnecessary text.

---

# Source Citations and Hallucination Prevention

Source citations play an important role in reducing hallucinations.

Without citations:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{LLM}
\\
\downarrow
\\
\text{Answer}
\end{gathered}
$$

The model may produce an answer that sounds correct but is actually unsupported.

With citations:

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
\text{Relevant Context}
\\
\downarrow
\\
\text{LLM}
\\
\downarrow
\\
\text{Answer}
\\
\downarrow
\\
\text{Evidence}
\end{gathered}
$$

Now every answer has supporting evidence from the uploaded document.

Although citations cannot completely eliminate hallucinations, they make them much easier to identify and verify.

---

# Explainable AI (XAI)

One of the biggest challenges with Large Language Models is explainability.

Traditional LLMs often behave like black boxes.

Example:

```
Question: What architecture does BERT use?

Answer: Transformer Architecture
```

The answer might be correct, but the user has no idea where it came from.

With source citations:

```
Question: What architecture does BERT use?

Answer: Transformer Architecture

Source
Page : 1
Chunk ID : 12
Context : "BERT is based on the Transformer architecture..."
```

The reasoning becomes much more transparent.

This is one of the main goals of Explainable AI (XAI).

---

# Benefits of Source Citations

Source citations provide several advantages.

- Improves transparency
- Builds user trust
- Helps verify generated answers
- Makes debugging easier
- Supports explainable AI
- Reduces the impact of hallucinations
- Makes RAG systems production-ready

---

# Production Best Practices

## 1. Never Return Only an Answer

Instead of:

```
Answer
```

Return:

```
Answer

+

Evidence
```

Users should always be able to verify the generated response.

---

## 2. Preserve Metadata

During preprocessing, we carefully preserved metadata such as:

- Page number
- Chunk ID
- Source filename

This metadata becomes extremely valuable during retrieval.

Without metadata:

```
Answer

No idea where it came from.
```

With metadata:

```
Answer

Page : 4
Chunk ID : 28
Source : bert.pdf
```

---

## 3. Keep Business Logic Separate from UI

The RAG pipeline returns:

```python
{
    "answer": response.content,
    "documents": documents,
}
```

Notice that it does **not** format the citations.

The UI is responsible for deciding how the information should be displayed.

Example:

```
Streamlit

↓

Answer

↓

Expandable Sources

↓

Context Preview
```

This follows the Separation of Concerns principle.

---

## 4. Return Raw Documents

Instead of converting the retrieved documents into custom dictionaries inside the RAG pipeline, we return the original LangChain `Document` objects.

Advantages:

- More flexible
- Less duplicate code
- Easier to extend
- Compatible with LangChain
- Preserves all metadata

The presentation layer can extract only the fields it needs.

---

# Why We Kept Both Functions

Current implementation:

```python
generate_answer()
```

Returns:

```
Answer
```

---

```python
generate_answer_with_sources()
```

Returns:

```python
{
    "answer": "...",
    "documents": [...]
}
```

Keeping both implementations has several advantages.

- Easier learning
- Demonstrates project evolution
- Older notebooks continue to work
- Better Git history
- Easier comparison between both approaches

Later, the Streamlit application will use only:

```python
generate_answer_with_sources()
```

---

# Complete Project Flow

At this stage, the complete system works as follows:

$$
\begin{gathered}
\text{Upload PDF}
\\
\downarrow
\\
\text{Extract Text}
\\
\downarrow
\\
\text{Split into Chunks}
\\
\downarrow
\\
\text{Generate Embeddings}
\\
\downarrow
\\
\text{Store in FAISS}
\\
\downarrow
\\
\text{User Question}
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
\text{Prompt Construction}
\\
\downarrow
\\
\text{Gemini}
\\
\downarrow
\\
\text{Answer}
\\
\downarrow
\\
\text{Source Citations}
\end{gathered}
$$

---

# Project Structure

Current project structure:

```
src/

document_loader.py

chunking.py

embeddings.py

vector_store.py

retrieval.py

prompts.py

rag_chain.py
```

Notebook progression:

```
01_pdf_extraction.ipynb

02_text_chunking.ipynb

03_embeddings.ipynb

04_faiss_integration.ipynb

05_retrieval.ipynb

06_rag_pipeline.ipynb

07_source_citations.ipynb
```

Documentation:

```
01_project_setup.md

02_pdf_extraction.md

03_text_chunking.md

04_faiss_integration.md

05_retrieval.md

06_rag_pipeline.md

07_source_citations.md
```

---

# Current Project Status

Completed:

✅ Project Setup

✅ PDF Extraction

✅ Text Chunking

✅ Embedding Generation

✅ FAISS Integration

✅ Retrieval

✅ Prompt Templates

✅ RAG Pipeline

✅ Source Citations

---

# What We Have Built

The application can now:

- Upload PDF documents
- Extract page-wise text
- Split text into chunks
- Generate embeddings
- Store embeddings in FAISS
- Retrieve the most relevant chunks
- Generate grounded answers using Gemini
- Display page numbers
- Display chunk IDs
- Display source filename
- Display context preview

This satisfies the core functional requirements of a production-style PDF RAG Assistant.

---

# Key Learnings

During this stage, we learned:

- What source citations are
- Why evidence is important in RAG
- How LangChain `Document` objects store metadata
- Why returning raw `Document` objects is better than formatting citations inside the pipeline
- Why context previews improve readability
- How citations help reduce hallucinations
- The role of Explainable AI (XAI) in modern AI systems
- Why presentation logic should remain separate from business logic

---

# Next Implementation Phase

```
Evaluation
```

Goal:

Evaluate the quality of the RAG system by measuring:

- Retrieval quality
- Answer correctness
- Hallucination cases
- Context relevance
- Faithfulness of generated answers

We will use:

- Custom evaluation metrics
- RAGAS (Retrieval-Augmented Generation Assessment)

This step helps determine how well the system performs and identifies areas for improvement before deployment.