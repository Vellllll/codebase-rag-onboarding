# 🧩 Codebase AI Mentor — Codebase RAG & Onboarding Assistant

An AI mentor that reads an entire GitHub repository and answers questions about its code — architecture, data flow, specific functions, and even **who last changed what** — with every answer backed by precise file/line references.

The system clones a repo, parses it into code chunks with **AST (Abstract Syntax Tree)** via tree-sitter, enriches every chunk with **Git blame metadata**, indexes it into **Qdrant** with hybrid search (semantic + BM25 + reranking), and answers questions through a **LangGraph agent** equipped with code search, file-explorer, and web-search tools — all behind a login-protected **Streamlit** chat UI.

---

## ✨ Features

### 🔍 RAG Pipeline
- **AST-based chunking** — code is split by real structure (functions, classes, methods, interfaces, type aliases) using tree-sitter, not arbitrary character slicing.
  - **Multi-language**: Python, TypeScript, TSX (JSX-aware grammar), JavaScript/JSX.
  - **Fallback chunking**: files that fail AST parsing (unrecognized languages, non-structural content) are indexed as whole files or line blocks, so nothing is lost.
- **Hybrid retrieval** — dense vector search (`BAAI/bge-small-en-v1.5` embeddings) **+** BM25 keyword search, fused with **Reciprocal Rank Fusion (RRF)**.
- **Cross-encoder reranking** — FlashRank (ONNX cross-encoder) re-scores fused results so the final context is actually relevant.
- **Qdrant vector store** — local file mode by default, or server/cloud mode via `QDRANT_URL` + `QDRANT_API_KEY` (required when the Streamlit app and webhook server access the store concurrently).

### 📊 Git-aware answers
- **Git blame enrichment** — every indexed chunk carries author, commit hash, date, and commit message (one `git blame` per file, reused across chunks).
- Ask *"Who last changed the login function?"* and get the author, commit hash, and commit message — not a guess.

### 🤖 AI Agent (LangGraph)
- A **ReAct agent** (`create_react_agent`) orchestrates 4 tools per question:
  - `search_codebase` — hybrid retrieval over the vector index (top-3 chunks).
  - `list_directory` — browse the cloned repo's folder structure (with directory-traversal protection).
  - `read_file_content` — read a full file when chunks are truncated or imports/config need checking (15k char cap to protect context).
  - `web_search` — DuckDuckGo, used only when external documentation is needed.
- **Conversational memory** — thread-based memory (`MemorySaver` checkpointer) so follow-up questions keep context; a fresh thread is created per chat session and reset with "Hapus Riwayat Chat".
- **Loop guard** — `recursion_limit` prevents runaway agent loops; graceful fallback message if a question is too complex.
- **Streaming status events** — the UI live-updates which tool is running (searching, browsing, reading, web search) as the agent works.

### 🧭 Onboarding-friendly output
- **Mermaid diagrams** — ask for a flowchart/sequence diagram and the agent renders it live (`streamlit-mermaid`).
- **Source references** — every answer includes clickable `file_path` + line-range pills so claims can be verified (anti-hallucination).
- **Chat history** — full conversation history with diagrams and sources, persistent across reloads.

### 🗂️ Repo ingestion & filtering
- **Clone from GitHub URL** — supports **private repos** via a GitHub Personal Access Token.
- **Index-time folder targeting** — restrict indexing to specific folders (e.g. `src/api, libs/utils`) for fast indexing of large monorepos.
- **Chat-time folder filtering** — focus retrieval on folders on the fly (no re-indexing); implemented both in the Qdrant filter and BM25 search.
- **Persistent index status** — the sidebar shows the connected repo, chunk count, and when it was indexed (survives reloads via `qdrant_storage/index_metadata.json`).

### 🔄 Incremental re-indexing (webhook server)
- A **FastAPI webhook server** (`server.py`) receives GitHub push events at `/api/webhook/github`.
- **HMAC-SHA256 signature verification** (`X-Hub-Signature-256`) — requests without a valid signature get `401`. The server refuses to start without `GITHUB_WEBHOOK_SECRET` (no insecure defaults).
- Only **added/modified files are re-parsed and re-indexed** (`force_recreate=False`), in a background task so GitHub never times out.

### 🔐 Authentication
- Login-gated UI via **streamlit-authenticator** (bcrypt-hashed passwords, cookie-based sessions with configurable expiry days).
- Users configured in `.streamlit/config.yaml`; passwords hashed automatically on startup.

---

## 🏗️ Architecture

```
GitHub Repo URL
     │  (GitPython clone → ./cloned_repo)
     ▼
[ IncrementalCodebaseIndexer ]
     │  tree-sitter AST chunking (.py/.ts/.tsx/.js/.jsx)
     │  + git blame metadata per chunk
     ▼
[ Qdrant Vector Store ] ──► (BM25 corpus loaded at startup)
     │
     ▼
[ HybridCodeRetriever ]  =  Dense vector search + BM25
     │                      → Reciprocal Rank Fusion (RRF)
     │                      → FlashRank cross-encoder rerank
     ▼
[ LangGraph ReAct Agent ]
     │  tools: search_codebase · list_directory
     │         read_file_content · web_search
     │  memory: MemorySaver (per-thread)
     ▼
[ Answer + Mermaid diagram + file:line sources ]
```

GitHub push webhooks (optional) flow into the indexer directly for incremental re-indexing.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit, streamlit-authenticator, streamlit-mermaid |
| Agent | LangGraph (`create_react_agent`), LangChain core |
| LLM | DeepSeek (`deepseek-chat` via OpenAI-compatible API) |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace/sentence-transformers) |
| Parsing | tree-sitter + tree-sitter-languages |
| Vector store | Qdrant (`langchain-qdrant`) |
| Retrieval | rank-bm25, FlashRank cross-encoder, RRF fusion |
| Webhook server | FastAPI + uvicorn |
| Git | GitPython (clone + blame) |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Prerequisites
Python 3.10+ and a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables
Copy `.env` with at least:

```
DEEPSEEK_API_KEY=sk-...            # required — LLM won't load without it
GITHUB_WEBHOOK_SECRET=...          # required only for the webhook server
LOCAL_REPO_PATH=./sample_project   # webhook server: local repo to re-index
# Optional — Qdrant server mode (recommended when running app + webhook together):
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...
# Optional — tune BM25 corpus size:
BM25_CORPUS_HARD_CAP=100000
```

### 3. Configure authentication
Create `.streamlit/config.yaml` (passwords are hashed automatically on startup):

```yaml
credentials:
  usernames:
    yourname:
      email: you@example.com
      name: Your Name
      password: your-password   # hashed with bcrypt at runtime
cookie:
  expiry_days: 30
  key: random_secret_key
  name: codebase_rag_cookie
```

### 4. Run the app

```bash
streamlit run app.py
```

Log in, paste a GitHub URL in the sidebar (add a PAT for private repos, optionally restrict indexing to target folders), hit **🚀 Mulai Indexing**, and start chatting — e.g.:

- *"Di mana letak logika Autentikasi User?"*
- *"Buatkan diagram alur untuk checkout keranjang"*
- *"Jelaskan arsitektur folder src/components"*
- *"Siapa yang terakhir mengubah fungsi login?"*

### 5. (Optional) Webhook server for auto re-indexing

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Add a GitHub webhook pointing at `http://<your-host>:8000/api/webhook/github` with content type `application/json`, the secret set in `GITHUB_WEBHOOK_SECRET`, and select the **push** event. Each push re-indexes only the changed files in the background.

---

## 📁 Project Structure

```
app.py                  # Streamlit UI: login, sidebar indexing, chat, mermaid, sources
server.py               # FastAPI webhook server (incremental re-indexing)
src/
  ast_parser.py         # tree-sitter multi-language AST chunking + fallback
  indexer.py            # clone repo, walk files, git blame enrichment, index to Qdrant
  retriever.py          # hybrid retrieval: dense + BM25 + RRF + FlashRank rerank
  synthesizer.py        # LangGraph ReAct agent, tools, memory, streaming events
requirements.txt
.streamlit/config.yaml  # auth credentials (gitignored in a real deploy)
cloned_repo/            # clone of the currently indexed repo (kept for file explorer)
qdrant_storage/         # local Qdrant data + index metadata
```
