import streamlit as st
import os
from streamlit_mermaid import st_mermaid
import re

# Import Synthesizer & Indexer dari backend yang sudah kita buat
from src.synthesizer import CodebaseSynthesizer
from src.indexer import IncrementalCodebaseIndexer

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="AI Codebase Onboarding Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- CACHING BACKEND ENGINE ---
@st.cache_resource
def load_synthesizer():
    """Memuat engine Synthesizer RAG ke memori (caching agar cepat)."""
    return CodebaseSynthesizer()

@st.cache_resource
def load_indexer():
    """Memuat Indexer untuk kebutuhan re-indexing repositori."""
    return IncrementalCodebaseIndexer()

# --- INITIALIZATION ---
st.title("🤖 AI Codebase Onboarding & Architecture Assistant")
st.caption("Tanyakan arsitektur, alur data, atau lokasi fungsi pada codebase repositori kamu.")

# Inisialisasi Session State untuk menyimpan history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: REPOSITORY MANAGEMENT ---
with st.sidebar:
    st.header("⚙️ Repository Settings")
    
    # Ganti dari local path ke URL Input
    github_url_input = st.text_input(
        "Public GitHub Repository URL:", 
        placeholder="https://github.com/facebook/react"
    )
    
    if st.button("⚡ Index Public Repository", type="primary"):
        if github_url_input.startswith("https://github.com/"):
            with st.spinner("Cloning repository, parsing AST, & indexing chunks..."):
                try:
                    indexer = load_indexer()
                    total_chunks = indexer.index_from_github_url(github_url_input)
                    
                    if total_chunks > 0:
                        st.success(f"Berhasil meng-index {total_chunks} chunk kode dari GitHub!")
                        st.cache_resource.clear() # Reset cache
                    else:
                        st.warning("Tidak ditemukan file kode (TS/JS/Py) yang valid.")
                except Exception as e:
                    st.error(f"Gagal memproses repositori GitHub: {e}")
        else:
            st.error("Masukkan URL GitHub yang valid (contoh: https://github.com/user/repo)!")

    st.markdown("---")
    st.markdown("**Features Supported:**")
    st.markdown("- 🌳 AST Code Chunking")
    st.markdown("- 🔍 Hybrid Search (Vector + BM25)")
    st.markdown("- 📊 Automatic Mermaid.js Diagrams")
    st.markdown("- 📍 Precise File & Line Referencing")

# --- HELPER FUNCTION: PARSE MERMAID BLOCK ---
def extract_mermaid_code(text: str):
    """Mengekstrak blok kode mermaid dari teks jawaban LLM."""
    pattern = r"```mermaid\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return None

# --- RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render diagram Mermaid jika ada di history
        if "mermaid_code" in message and message["mermaid_code"]:
            st.caption("📊 Architecture Diagram:")
            st_mermaid(message["mermaid_code"])
            
        # Render referensi file sumber jika ada di history
        if "sources" in message and message["sources"]:
            with st.expander("📍 Lihat Sumber Kode Referensi"):
                for src in message["sources"]:
                    st.markdown(f"- **`{src['file_path']}`** *(Baris {src['start_line']}-{src['end_line']})* — `{src['name']}`")

# --- USER CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Contoh: Bagaimana alur autentikasi user dan buatkan diagramnya?"):
    
    # 1. Tampilkan pesan user di UI & simpan di history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Proses jawaban AI dengan streaming
    with st.chat_message("assistant"):
        synthesizer = load_synthesizer()
        
        # Ambil referensi dokumen kode terlebih dahulu
        relevant_docs = synthesizer.retriever.get_relevant_code(prompt, top_k=3)
        sources = [
            {
                "file_path": doc.metadata.get("file_path"),
                "start_line": doc.metadata.get("start_line"),
                "end_line": doc.metadata.get("end_line"),
                "name": doc.metadata.get("name")
            }
            for doc in relevant_docs
        ]

        # Stream teks jawaban seperti efek mengetik
        answer_text = st.write_stream(synthesizer.stream_answer(prompt))
        
        # Ekstrak & render diagram Mermaid jika ada
        mermaid_code = extract_mermaid_code(answer_text)
        if mermaid_code:
            st.caption("📊 Architecture Diagram:")
            st_mermaid(mermaid_code)
        
        # Render referensi file sumber jika ada
        if sources:
            with st.expander("📍 Lihat Sumber Kode Referensi"):
                for src in sources:
                    st.markdown(f"- **`{src['file_path']}`** *(Baris {src['start_line']}-{src['end_line']})* — `{src['name']}`")
        
        # Simpan jawaban ke session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "mermaid_code": mermaid_code,
            "sources": sources
        })