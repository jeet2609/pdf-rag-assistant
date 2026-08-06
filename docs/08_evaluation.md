# Evaluation

## Overview

After building the complete Retrieval Augmented Generation (RAG) pipeline, the next step is to evaluate how well it performs.

Evaluation helps answer important questions such as:

- Is the retriever finding the correct document chunks?
- Is the LLM generating accurate answers?
- Are the answers grounded in the retrieved context?
- Is the system hallucinating?
- How similar is the generated answer to the expected answer?

A production-ready AI system should always be evaluated before deployment.

---

# Why Evaluation is Important

Building a RAG system is only half the work.

The other half is measuring its performance.

Without evaluation:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{Answer}
\\
\downarrow
\\
\text{Looks correct?}
\\
\text{(Maybe...)}
\end{gathered}
$$

With evaluation:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{Generate Answer}
\\
\downarrow
\\
\text{Compare with Ground Truth}
\\
\downarrow
\\
\text{Similarity Score}
\\
\downarrow
\\
\text{Performance Report}
\end{gathered}
$$

Evaluation provides confidence that the RAG system is working correctly.

---

# What We Evaluate

Our evaluation focuses on the generated answer.

For every test question we have:

- Question
- Expected Answer
- Generated Answer

Example:

Question:

```
What architecture does BERT use?
```

Expected Answer:

```
BERT is based on the Transformer architecture.
```

Generated Answer:

```
BERT uses a multi-layer bidirectional Transformer encoder.
```

Even though the wording is different, both answers have the same meaning.

Therefore, exact string comparison is not suitable.

Instead, we use semantic similarity.

---

# Semantic Similarity

Semantic similarity measures whether two sentences have the same meaning.

Example:

Sentence 1

```
BERT uses the Transformer architecture.
```

Sentence 2

```
BERT is based on a Transformer encoder.
```

The wording differs, but the meaning is nearly identical.

Their similarity score should therefore be very high.

---

# Why Not Compare Strings?

Exact comparison:

Sentence A

```
Transformer architecture
```

Sentence B

```
Transformer encoder
```

String comparison:

```
False
```

Semantic comparison:

```
Very Similar
```

This is why semantic embeddings are used instead of plain text comparison.

---

# Evaluation Pipeline

The complete evaluation process:

$$
\begin{gathered}
\text{Question}
\\
\downarrow
\\
\text{RAG Pipeline}
\\
\downarrow
\\
\text{Generated Answer}
\\
\downarrow
\\
\text{Sentence Embeddings}
\\
\downarrow
\\
\text{Cosine Similarity}
\\
\downarrow
\\
\text{Evaluation Score}
\end{gathered}
$$

---

# Similarity Model

For evaluation we use a separate Sentence Transformer model.

Model:

```
sentence-transformers/all-MiniLM-L6-v2
```

Purpose:

Convert complete answers into embeddings.

Example:

$$
\begin{gathered}
\text{Expected Answer}
\rightarrow
\text{Embedding}
\end{gathered}
$$

$$
\begin{gathered}
\text{Generated Answer}
\rightarrow
\text{Embedding}
\end{gathered}
$$

The two embeddings are compared using cosine similarity.

---

# Why Use a Separate Model?

During retrieval we used:

```
BAAI/bge-small-en-v1.5
```

Purpose:

Retrieve relevant document chunks.

During evaluation we use:

```
all-MiniLM-L6-v2
```

Purpose:

Measure semantic similarity between answers.

Although the same embedding model could be used for both tasks, using a dedicated similarity model is a common practice because retrieval and semantic comparison are different objectives.

---

# Cosine Similarity

Cosine similarity measures the angle between two embedding vectors.

$$
\text{Expected Answer}
\\
\downarrow
\\
\text{Embedding}
\\
\\
\text{Generated Answer}
\\
\downarrow
\\
\text{Embedding}
\\
\downarrow
\\
\text{Cosine Similarity}
$$

Interpretation:

$$
1.00
\rightarrow
\text{Identical M}eaning
$$

$$
0.90+
\rightarrow
\text{Excellent}
$$

$$
0.75+
\rightarrow
\text{Good}
$$

$$
0.50+
\rightarrow
\text{Fair}
$$

$$
\text{Below 0.50}
\rightarrow
\text{Poor}
$$

---

# Evaluation Functions

The evaluation module contains three production-ready functions.

## load_similarity_model()

Responsibilities:

- Load Sentence Transformer model
- Detect available device
- Use Apple MPS on macOS
- Fall back to CPU if needed

Returns:

```
SentenceTransformer
```

---

## calculate_similarity()

Responsibilities:

- Convert answers into embeddings
- Calculate cosine similarity
- Return similarity score

Returns:

```
float
```

Example:

```
0.9638
```

---

## evaluate_rag()

Responsibilities:

- Compare expected answer
- Compare generated answer
- Calculate similarity
- Assign performance category

Returns:

```python
{
    "question": "...",
    "expected_answer": "...",
    "generated_answer": "...",
    "similarity": 0.96,
    "status": "Excellent"
}
```

---

# Why We Used get_device()

Earlier in the project we created:

```python
get_device()
```

Instead of writing device detection again, we reused the same function.

Benefits:

- No duplicate code
- Single source of truth
- Easier maintenance
- Consistent device selection across the project

The similarity model is loaded as:

```python
SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=get_device(),
)
```

If Apple Silicon GPU (MPS) is available, inference runs on the GPU.

Otherwise, CPU is used automatically.

---

# Why We Do Not Pass Device to encode()

Once the model is loaded on a device:

```python
SentenceTransformer(
    ...,
    device=get_device()
)
```

every subsequent call to:

```python
similarity_model.encode(...)
```

automatically runs on the same device.

Therefore we do **not** need to pass the device again.

Flow:

$$
\begin{gathered}
\text{Load Model}
\\
\downarrow
\\
\text{Device Selected}
\\
\downarrow
\\
\text{encode()}
\\
\downarrow
\\
\text{Runs on Same Device}
\end{gathered}
$$

---

# Handling LangChain TextAccessor

During implementation we encountered an issue.

The generated answer returned by LangChain was not a plain Python string.

Instead it was:

```
langchain_core.messages.base.TextAccessor
```

Passing this object directly to:

```python
similarity_model.encode(...)
```

resulted in an error because Sentence Transformers expects a string.

To make the evaluation module robust, we convert both inputs to plain strings before encoding.

```python
expected_answer = str(expected_answer)

generated_answer = str(generated_answer)
```

This ensures compatibility with different LangChain versions and future LLM implementations.

---

# Example Evaluation

Input:

Question:

```
What architecture does BERT use?
```

Expected:

```
BERT uses the Transformer architecture.
```

Generated:

```
BERT is based on the Transformer encoder.
```

Output:

```python
{
    "similarity": 0.97,
    "status": "Excellent"
}
```

---

# Benefits of Semantic Evaluation

Compared to exact string matching, semantic evaluation:

- Understands meaning
- Handles paraphrasing
- Ignores wording differences
- Produces more realistic scores
- Better reflects LLM performance

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

✅ Evaluation

---

# Next Implementation Phase

```
Streamlit User Interface
```

Goal:

Build an interactive web application where users can:

- Upload PDF documents
- Ask questions
- View generated answers
- See source citations
- Review supporting context

This will transform the RAG pipeline into a complete end-user application.