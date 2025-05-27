# 📄 Research Paper Publishability Assessment Framework

## 🚀 Overview

This project proposes a novel framework to **automatically assess research paper quality** and classify them into target conferences using cutting-edge NLP and graph-based methods. The solution leverages **Knowledge Graphs**, **SciBERT**, **Vector Embeddings**, and **LLMs** to ensure transparency, accuracy, and scalability in peer-review automation.

---

## 🧠 Core Components

### 🧩 Task 1: Publishability Classification

- **Goal:** Classify papers as `Publishable` or `Non-Publishable`.
- **Highlights:**
  - Custom **PDF text extraction** algorithm using `pdfplumber` for layout-aware processing.
  - **Semantic Chunking** with SciBERT (512-token segments).
  - **Knowledge Graph Construction** using `LangChain` and `Gemma2-9b-It`.
  - **Graph Metrics:** PageRank, Graph Density, Degree Centrality.
  - **Leiden Clustering** for coherence visualization.
  - Classification threshold based on average top 5 PageRank scores.

### 🧭 Task 2: Conference Classification

- **Goal:** Classify papers into conferences like NeurIPS, CVPR, EMNLP, KDD, and TMLR.
- **Highlights:**
  - Semantic Embeddings via **SciBERT**.
  - **VectorStore**-backed similarity search with cosine distance.
  - **Supervised Sequence Classifier** with cross-entropy loss.
  - **FLAN-T5**-based rationale generation to ensure interpretability.
  - **K-means Clustering** for trend discovery.

---

## 📁 Dataset

- Custom dataset of **600 papers** (300 publishable, 300 non-publishable).
- Publishable papers sourced from conference proceedings.
- Non-publishable papers sourced from OpenReview, Rejecta Mathematica, and IEEE Xplore.
- Embeddings stored in Pathway’s `VectorStore`.

---

## 📈 Results

| Task | Accuracy |
|------|----------|
| Task 1 - Publishability | ✅ **100%** (labeled), ✅ **96.7%** (custom) |
| Task 2 - Conference Classification | ✅ **94.3%** on held-out test set |

---

## 🛠 Tech Stack

- **Languages**: Python
- **Libraries**: LangChain, Pathway, SciBERT, Neo4j, pdfplumber, FLAN-T5
- **Models**: Gemma2-9b-It, FLAN-T5
- **Visualization**: Neo4j Desktop, Matplotlib, Seaborn

---

## 📚 References

- SciBERT: https://arxiv.org/abs/1903.10676  
- LangChain: https://www.langchain.com  
- Pathway: https://pathway.com  
- FLAN-T5: https://huggingface.co/docs/transformers/model_doc/flan-t5  
- Leiden Clustering: https://www.nature.com/articles/s41598-019-41695-z  
- Knowledge Graphs for NLP: https://aclanthology.org/D19-1548/

---

## 🤝 Future Work

- Expand to broader scientific domains.
- Improve generalization across unknown paper formats.
- Integrate citation graphs for longitudinal analysis.
