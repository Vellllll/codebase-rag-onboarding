# 🚀 AI Codebase Onboarding & Architecture Assistant

An advanced Retrieval-Augmented Generation (RAG) system designed to dramatically reduce software engineer onboarding time. This tool parses entire codebases using **AST (Abstract Syntax Tree)**, indexes code blocks with hybrid search (Semantic + BM25), and acts as an interactive AI mentor capable of explaining code architecture, data flow, and dependencies.

---

## ✨ Features

- **🌳 AST-Based Chunking**: Code chunking based on actual code structure (functions, classes, interfaces) using `tree-sitter`, avoiding arbitrary character slicing.
- **🔍 Hybrid Search & Reranking**: Combines dense vector search (semantic context) with BM25 (exact keyword matching) and reranking for pinpoint context retrieval.
- **📊 Architecture Diagram Generation**: Automatically generates Mermaid.js flowcharts and sequence diagrams to visualize application flow.
- **📍 Precise Source Referencing**: Every answer includes exact file paths (`file_path`) and line numbers (`start_line`-`end_line`) to eliminate hallucinations.

---

## 🏗️ Architecture Blueprint

[ Input Query ] ──► [ Hybrid Retriever (Vector DB + BM25) ]
│
▼
[ Cohere/Flashrank Reranker ]
│
▼
[ LLM (GPT-4o/Claude) ]
│
▼
[ Structured Response + Mermaid Diagrams ]

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Parsing**: `tree-sitter`, `tree-sitter-languages`
* **Orchestration**: LlamaIndex / LangChain
* **Vector Store**: Qdrant / ChromaDB
* **Embeddings & LLM**: OpenAI `text-embedding-3-small` & `gpt-4o`
* **Interface**: Streamlit / Chainlit

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed and set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate