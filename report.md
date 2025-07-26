# Retrieval‑Augmented Generation (RAG) Pipeline Report

Author: Md. Sakib Muballig  
Date: July 26, 2025  

---

## 1. Approach & Rationale

To build a reliable QA system over Meta’s Q1 2024 financial report, we followed a classic three‑phase RAG (Retrieval‑Augmented Generation) workflow:

1. Basic RAG Pipeline  
   - Extract all text from the PDF.  
   - Chunk into overlapping windows for context coverage.  
   - Embed each chunk and index with FAISS for fast similarity search.  
   - Retrieve & Generate: fetch top‑3 chunks, then use an open‑source LLM to answer queries.

   *Rationale:* This separates knowledge (PDF text) from modeling (LLM), ensuring factual grounding.

2. Structured Data Integration  
   - Extract tables (income statement, expense breakdown) with Camelot into pandas DataFrames.  
   - Hybrid Retrieval: combine vector search for narrative queries and direct DataFrame lookups for numeric/ comparative queries.

   *Rationale:* Financial reports mix prose and tables—leveraging both ensures accuracy on numbers and context.

3. Advanced RAG & Evaluation  
   - Query Optimization: rewrite user questions via a small LLM to improve retrieval precision.  
   - Reranking: rescore top‑k chunks with a cross‑encoder to boost relevance.  
   - Chunk‑Size Experiments: test multiple chunk lengths for optimal recall.  
   - Evaluation Framework: measure Precision@k, Recall@k, MRR for retrieval; BLEU/ROUGE for answer quality; manual rubric for end‑to‑end assessment.

   *Rationale:* Iterative refinement and quantitative evaluation surface weaknesses and guide improvements.

---

## 2. Tools & Frameworks Used

- PDF parsing: `pypdf`  
- Text splitting: `langchain`’s `RecursiveCharacterTextSplitter`  
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)  
- Vector search: `faiss-cpu`  
- Table extraction: `camelot-py[cv]`  
- Generation & rewriting: Hugging Face `transformers` (`google/flan-t5-base`, `flan-t5-small`)  
- Reranking: `sentence-transformers` CrossEncoder (`ms-marco-MiniLM-L-6-v2`)  
- Evaluation: `evaluate` (BLEU, ROUGE) + custom Precision/Recall/MRR code  
- Data handling/plots: `pandas`, `matplotlib`

---

## 3. Challenges & Solutions

| Challenge                                                      | Solution                                                                                                    |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| PDF text extraction produced irregular whitespace         | Used `pypdf`’s built‑in cleanup and manual `.strip()` on extracted strings.                                 |
| Misaligned column headers in Camelot tables                | Inspected raw headers, applied `.str.replace()` and manual renaming to standardize “2024”/“2023” columns. |
| Large LLM load times on Colab free GPU                     | Switched from Falcon‑7B to `google/flan-t5-base` for prompt rewriting and answer generation.                |
| Retrieval precision dropped on numeric/comparative queries | Implemented hybrid lookup: DataFrame queries for numbers, vector search for text, then merged results.     |
| Evaluating end‑to‑end quality                              | Built scripts for Precision@3, MRR, and BLEU/ROUGE, plus a small manual rubric for fluency & relevance.    |

---

## 4. Key Results & Observations

- Step 1 Retrieval  
  - Precision@3: ~0.80 on 5 factual queries  
  - Average answer latency: ~5 s per query  

- Step 2 Structured Integration  
  - Numeric queries (net income, expenses) were 100 % correct via DataFrame lookup.  
  - Combined responses read more naturally when narrative context was included alongside table values.

- Step 3 Advanced RAG  
  - Chunk‑size 500–600 chars maximized Recall@3 (~0.85)  
  - Reranking boosted Precision@3 by ~15 %  
  - BLEU: 0.72, ROUGE‑L: 0.65 on 15‑query test set  
  - Manual rubric (0/1 correctness, 1–5 fluency/relevance) averaged 4.2/5 across all queries.

---

## 5. Sample Outputs for Test Queries

| Query                                                      | Answer                                                                                                        |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| What was Meta’s revenue in Q1 2024?                    | “Meta reported \$36.46 billion in Q1 2024.”                                                                   |
| How did Q1 2024 net income compare to Q1 2023?         | “Net income was \$12 369 million in Q1 2024 vs \$5 709 million in Q1 2023, an increase of ~117 %.”            |
| Summarize operating expenses in Q1 2024.               | “Cost of revenue: \$6 640 M; R&D: \$9 978 M; Marketing & sales: \$2 564 M; G&A: \$3 455 M; Total: \$22 637 M.” |
| What guidance did Meta give for Q2 2024 revenue?       | “Meta expects Q2 2024 revenue to be in the range \$36.5–39 B, assuming a 1 % FX headwind.”                     |

---

## 6. Conclusion & Next Steps

This RAG pipeline demonstrates robust factual and numeric QA over a financial report. Future enhancements include:

1. Dynamic chunking based on document structure (tables vs. prose).  
2. Feedback loop to fine‑tune reranking on user‑provided relevance labels.  
3. Integration of a SQL‑like interface for complex multi‑table queries.

All code, detailed methodology, and output artifacts are available in this repository.
