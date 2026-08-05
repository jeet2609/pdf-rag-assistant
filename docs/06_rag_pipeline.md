# RAG Pipeline - PDF RAG Assistant with Source Citations

## Overview

The RAG (Retrieval Augmented Generation) Pipeline is the core of the application.

It combines everything we have built so far into a complete question-answering system.

At this stage, the application can:

- Accept a user's question
- Retrieve the most relevant document chunks
- Build a prompt using the retrieved context
- Send the prompt to Gemini
- Generate a grounded answer

Unlike a normal LLM, a RAG system does not rely only on the model's internal knowledge.

Instead, it first retrieves relevant information from the uploaded document and then asks the LLM to answer using that information.

---

## RAG Pipeline

The complete pipeline:

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
\text{Top-k Relevant Chunks}
\\
\downarrow
\\
\text{Prompt Template}
\\
\downarrow
\\
\text{Gemini LLM}
\\
\downarrow
\\
\text{Grounded Answer}
\end{gathered}
$$

---

## Why Do We Need a RAG Pipeline?

Without Retrieval:

$$
\begin{gathered}
\text{User Question}
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

- The model may hallucinate.
- It cannot access private PDF documents.
- It may answer using outdated knowledge.
- It cannot provide evidence.

---

With Retrieval:

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
\text{Relevant Context}
\\
\downarrow
\\
\text{LLM}
\\
\downarrow
\\
\text{Grounded Answer}
\end{gathered}
$$

Advantages:

- More accurate
- Less hallucination
- Uses uploaded documents
- Provides trustworthy answers

---

## Components Used

The RAG pipeline combines the modules built in previous steps.

```
document_loader.py
```

↓

```
chunking.py
```

↓

```
embeddings.py
```

↓

```
vector_store.py
```

↓

```
retrieval.py
```

↓

```
prompts.py
```

↓

```
rag_chain.py
```

---

## Responsibilities of rag_chain.py

The `rag_chain.py` module is responsible for:

- Loading Gemini
- Retrieving relevant documents
- Combining retrieved chunks
- Building the prompt
- Sending the prompt to Gemini
- Returning the generated answer

It does **not**:

- Load PDFs
- Split text
- Generate embeddings
- Create the FAISS index

Each module has a single responsibility.

---

## Step 1 — Retrieve Documents

The user's question is first sent to the retriever.

Example:

```
What architecture does BERT use?
```

The retriever returns:

```
Chunk 12

Chunk 27

Chunk 41

Chunk 58
```

These are the most semantically relevant chunks.

---

## Step 2 — Build Context

The retrieved chunks are merged into one context string.

Example:

```
Chunk 12

Chunk 27

Chunk 41
```

↓

```
Large Context String
```

This context becomes part of the prompt.

---

## Step 3 — Prompt Construction

The prompt contains:

- Instructions
- Retrieved context
- User question

Example:

```
Context:
...

Question:
What architecture does BERT use?

Answer:
```

Gemini receives only this prompt.

---

## Step 4 — Answer Generation

Gemini generates an answer using only the supplied context.

Example:

```
BERT is based on the Transformer architecture.
```

The response is therefore grounded in the retrieved document.

---

## Why Do We Use ChatPromptTemplate?

In `prompts.py` we used:

```python
ChatPromptTemplate.from_template(template)
```

instead of:

```python
return template
```

This is because LangChain expects a **Prompt Template object**, not just a string.

Using `ChatPromptTemplate` provides several advantages:

- Integrates directly with LangChain chains
- Validates prompt variables
- Makes prompts reusable
- Keeps prompt formatting separate from business logic
- Easier to extend in the future

Later we can simply do:

```python
prompt = get_rag_prompt()

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)
```

instead of manually formatting strings.

This follows the production-style architecture encouraged by LangChain.

---

# Why Don't We Return the Prompt String?

Returning:

```python
return template
```

would still work.

However, it becomes plain Python string formatting.

Example:

```python
prompt = template.format(
    context=context,
    question=question,
)
```

This bypasses LangChain's prompt abstraction.

Since our project is built around LangChain, using `ChatPromptTemplate` keeps every component compatible with the rest of the framework.

---

# Why Pass the Retriever Instead of the Vector Store?

One question during implementation was:

> Why do we pass the **retriever** to `generate_answer()` instead of passing the **FAISS vector store**?

The answer is separation of responsibilities.

The retriever already knows:

- Search type
- Value of **k**
- Search configuration
- Metadata filtering (if added later)

Therefore the RAG pipeline should simply use the retriever.

Workflow:

$$
\begin{gathered}
\text{FAISS}
\\
\downarrow
\\
\text{Retriever}
\\
\downarrow
\\
\text{RAG Pipeline}
\end{gathered}
$$

This makes each module independent.

---

# Why Does rag_chain.py Not Generate Embeddings?

Another question that came up during development was:

> Why don't we generate embeddings again inside `rag_chain.py`?

Because the embeddings already exist inside the FAISS vector store.

Workflow:

$$
\begin{gathered}
\text{Chunks}
\\
\downarrow
\\
\text{Embeddings}
\\
\downarrow
\\
\text{FAISS Index}
\\
\downarrow
\\
\text{Retriever}
\\
\downarrow
\\
\text{LLM}
\end{gathered}
$$

The retriever internally converts the question into an embedding and performs similarity search.

The RAG pipeline only consumes the retrieved documents.

---

# Why Use Temperature = 0?

Gemini is configured with:

```python
temperature = 0
```

Reason:

- More deterministic responses
- Less randomness
- Better reproducibility
- Preferred for factual question-answering systems

For RAG systems, consistency is usually more important than creativity.

---

# Project Structure

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

Each module performs exactly one task.

This follows the Single Responsibility Principle (SRP).

---

# Production Considerations

## 1. Keep Modules Independent

Each module should perform one responsibility only.

This improves:

- Readability
- Testing
- Reusability
- Maintainability

---

## 2. Keep Prompts Separate

Prompts should never be hardcoded inside business logic.

Keeping prompts inside `prompts.py` makes experimentation much easier.

---

## 3. Avoid Hallucinations

The prompt explicitly instructs Gemini:

- Use only the provided context.
- Do not make up information.
- If the answer is unavailable, say so.

This significantly reduces hallucinations.

---

## 4. Use Retrieved Context Only

The LLM should never receive the full PDF.

Instead:

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
\text{LLM}
\end{gathered}
$$

This reduces:

- Token usage
- API cost
- Latency

while improving answer quality.

---

# Current Project Status

Completed:

✅ Project Setup

✅ PDF Extraction

✅ Text Chunking

✅ Embedding Generation

✅ FAISS Integration

✅ Retrieval

✅ Prompt Template

✅ Complete RAG Pipeline

---

# Output of This Stage

Input:

```
User Question
```

Output:

```
Grounded Answer
```

At this stage, the answer is generated correctly, but source details are not yet shown.

---

# Key Learnings

Through this stage, we learned:

- What a RAG pipeline is
- Why retrieval happens before generation
- How Gemini is integrated
- Why prompts are separated from logic
- Why `ChatPromptTemplate` is used instead of plain strings
- Why the retriever is passed instead of the vector store
- Why embeddings are not regenerated
- How grounding helps reduce hallucinations
- Why temperature is set to 0 for factual tasks

---

# Next Implementation Phase

```
Source Citations
```

Goal:

Enhance the generated answer by displaying:

- PDF page number
- Chunk ID
- Source filename
- Context preview

This will make the application more transparent and allow users to verify where each answer came from.