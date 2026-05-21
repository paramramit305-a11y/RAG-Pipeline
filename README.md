<div align="center">

# 🔍 RAG Pipeline — Research Paper Q&A

### Retrieval-Augmented Generation from Scratch using LangChain & ChromaDB

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logoColor=white)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)](https://groq.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)

</div>

---

## 📌 Overview

A **Retrieval-Augmented Generation (RAG)** pipeline built entirely from scratch — no pre-built RAG abstractions. The system ingests research PDFs, stores semantic embeddings in ChromaDB, retrieves relevant chunks at query time, and generates grounded answers via an LLM.

The pipeline is **LLM-agnostic** — currently powered by **Groq (Qwen3-32b)** with plug-and-play support for OpenAI and Anthropic.

> ⚠️ OpenAI and Anthropic integrations are included as commented code. Groq was used as the primary LLM (free tier).

---

## 🎯 Problem Statement

LLMs hallucinate when asked about domain-specific or recent knowledge outside their training data. RAG solves this by grounding the model's responses in retrieved, verifiable source documents — making answers more accurate, traceable, and up-to-date.

---

## 🏗️ Architecture

```
                        INGESTION PIPELINE
─────────────────────────────────────────────────────
  PDF Files
      │
      ▼
┌─────────────────┐
│  PyPDFLoader    │  ← Load all PDFs from folder
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│  RecursiveCharacter      │  ← chunk_size=500, overlap=50
│  TextSplitter            │
└────────┬─────────────────┘
         │
         ▼
┌─────────────────┐
│ EmbeddingManager│  ← all-MiniLM-L6-v2 (local, CPU)
│ SentenceTransf. │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VectorStore     │  ← ChromaDB persistent store
│ Manager         │    UUID-based doc IDs + metadata
└─────────────────┘

                        RETRIEVAL PIPELINE
─────────────────────────────────────────────────────
  User Query
      │
      ▼
┌─────────────────┐
│ EmbeddingManager│  ← Embed query (same model)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAGRetriever   │  ← Cosine similarity search
│                 │    Score threshold filtering
│                 │    Returns top-K chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM (Groq)    │  ← Context + Query → Answer
│  Qwen3-32b      │
└────────┬────────┘
         │
         ▼
   Grounded Answer
```

---

## 🛠️ Tech Stack

<div align="center">

| Component | Technology |
|:----------|:-----------|
| 🔗 Framework | LangChain |
| 🧠 Embeddings | `all-MiniLM-L6-v2` (local, no API cost) |
| 🗄️ Vector Store | ChromaDB (persistent) |
| 🤖 LLM | Groq — `qwen/qwen3-32b` |
| 📄 PDF Loader | PyPDFLoader / PyMuPDFLoader |
| ✂️ Text Splitter | RecursiveCharacterTextSplitter |
| 🐍 Language | Python 3.10+ |

</div>

---

## 📁 Project Structure

```
RAG-Pipeline/
│
├── RAG_pipeline.ipynb     # Main notebook — full pipeline
├── data/
│   ├── pdfs/              # Add your PDF files here
│   └── Python.txt         # Sample text document
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/paramramit305-a11y/RAG-Pipeline.git
cd RAG-Pipeline
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
# Open .env and add your Groq API key
```

**4. Add your PDFs**
```
Place your PDF files inside data/pdfs/
```

**5. Run the notebook**
```bash
jupyter notebook RAG_pipeline.ipynb
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## 🤖 Switching LLMs

The pipeline is **LLM-agnostic**. Swap the LLM in one line:

```python
# ✅ Groq (free) — currently active
from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key=API_KEY_GROQ, model_name="qwen/qwen3-32b")

# OpenAI (commented)
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", api_key="...")

# Anthropic (commented)
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key="...")
```

---

## 🧪 Example Queries

```python
# Query 1 — from Attention Is All You Need paper
answer = generate_output("what is encoder-decoder?", rag_retriever, llm)

# Query 2 — from RAG Survey paper
answer = generate_output("what is RAG?", rag_retriever, llm)
```

---

## 📊 Key Design Decisions

| Decision | Reason |
|:---------|:-------|
| Local embeddings (`all-MiniLM-L6-v2`) | No API cost, runs on CPU |
| Persistent ChromaDB | No re-embedding needed on restart |
| Score threshold in retriever | Filters low-similarity chunks before LLM |
| UUID-based document IDs | Collision-free tracking across multiple PDFs |
| LLM-agnostic architecture | Swap any LangChain-compatible LLM in one line |

---

## 📚 Papers Used for Testing

- *Attention Is All You Need* — Vaswani et al. (2017)
- *Retrieval-Augmented Generation for LLMs: A Survey* — Gao et al. (2024)

---

## 🔮 Future Improvements

- [ ] Add reranking layer (Cohere / bge-reranker)
- [ ] Implement HyDE (Hypothetical Document Embeddings)
- [ ] Deploy as Gradio app on HuggingFace Spaces
- [ ] Add RAGAS evaluation metrics
- [ ] Support multi-modal RAG (images + text)

---

## 📄 References

- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [RAG for LLMs: A Survey — Gao et al. (2024)](https://arxiv.org/abs/2312.10997)
- [LangChain Documentation](https://python.langchain.com)
- [ChromaDB Documentation](https://docs.trychroma.com)

---

## 👤 Author

<div align="center">

**Parmar Amit**
*BSc IT (AIML) | Gokul Global University*

[![GitHub](https://img.shields.io/badge/GitHub-paramramit305--a11y-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/paramramit305-a11y)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Parmar%20Amit-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parmar-amit-97941a377)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-parmar--amit-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/parmar-amit)

</div>

---

<div align="center">

⭐ If you find this project useful, please consider giving it a star!

</div>
