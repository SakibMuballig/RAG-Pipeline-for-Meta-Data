# Retrieval‑Augmented Generation (RAG) Pipeline for Meta Q1 2024 Financial Report


## 1. Introduction

This project implements a three‑step RAG pipeline to answer factual and comparative questions about Meta’s Q1 2024 financial results.  
Data source: “Meta’s Q1 2024 Financial Report.pdf”  

Steps overview:
1. Basic RAG Pipeline – PDF text extraction, chunking, embedding, retrieval, and answer generation.  
2. Structured Data Integration – Table extraction into DataFrames, hybrid retrieval combining vector and structured lookup.  
3. Advanced RAG & Evaluation – Query rewriting, reranking, chunk‑size experiments, metrics (Precision@k, BLEU/ROUGE), test‑set execution, and proposed improvements.

---

## 2. Basic RAG Pipeline

### 2.1 Tools & Libraries
- PDF parsing: `pypdf`  
- Text splitting: `langchain`’s `RecursiveCharacterTextSplitter`  
- Embedding: `sentence-transformers` (`all-MiniLM-L6-v2`)  
- Vector index: `faiss-cpu`  
- Generation: Hugging Face `transformers` (`google/flan-t5-base` for speed on Colab)

### 2.2 Code Snippets

```python
# Load and extract all text
from pypdf import PdfReader
reader = PdfReader("Meta_Q1_2024.pdf")
text = "".join(p.extract_text() for p in reader.pages)

# Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Embeddings & FAISS index
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')
embs = model.encode(chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(embs.shape[1]); index.add(embs)
