import streamlit as st
import os
import re
import yaml
from yaml.loader import SafeLoader
from streamlit_mermaid import st_mermaid
from dotenv import load_dotenv
import streamlit_authenticator as stauth

# Load environment variables
load_dotenv()

# Import Synthesizer & Indexer dari backend
from src.synthesizer import CodebaseSynthesizer
from src.indexer import IncrementalCodebaseIndexer

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="AI Codebase Onboarding Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- SISTEM LOGIN (STREAMLIT AUTHENTICATOR) ---
# Membaca data akun dari config.yaml
with open('.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Inisialisasi Authenticator
# Inisialisasi Authenticator (TANPA config['preauthorized'])
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render widget login (Pembaruan v3.x menggunakan keyword 'location')
authenticator.login(location="main")

# Pengecekan status menggunakan st.session_state
if st.session_state["authentication_status"] is False:
    st.error("❌ Username atau password salah!")
    st.stop() # Hentikan aplikasi jika gagal login
elif st.session_state["authentication_status"] is None:
    st.warning("🔒 Silakan masukkan username dan password Anda.")
    st.stop() # Hentikan aplikasi jika belum login

# =====================================================================
# --- KODE APLIKASI UTAMA (HANYA BERJALAN JIKA LOGIN BERHASIL) ---
# =====================================================================

# Tampilkan tombol Logout di Sidebar
with st.sidebar:
    st.write(f"Selamat datang, **{st.session_state['name']}**! 👋")
    authenticator.logout(location="sidebar")
    st.markdown("---")

# --- CACHING BACKEND ENGINE ---
@st.cache_resource
def load_synthesizer():
    return CodebaseSynthesizer()

@st.cache_resource
def load_indexer() -> IncrementalCodebaseIndexer:
    return IncrementalCodebaseIndexer()

# --- INITIALIZATION & SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: REPOSITORY MANAGEMENT ---
with st.sidebar:
    st.header("⚙️ Repository Settings")
    
    github_url_input = st.text_input(
        "GitHub Repository URL:", 
        placeholder="https://github.com/username/repo-name"
    )
    
    github_token_input = st.text_input(
        "GitHub Access Token (PAT):",
        type="password",
        help="Wajib diisi jika ingin meng-index Private Repository."
    )
    
    if st.button("⚡ Index Repository", type="primary"):
        if github_url_input.startswith("https://github.com/"):
            with st.spinner("Cloning repository, parsing AST, & indexing chunks..."):
                try:
                    indexer = IncrementalCodebaseIndexer()
                    total_chunks = indexer.index_from_github_url(
                        github_url=github_url_input,
                        github_token=github_token_input if github_token_input else None
                    )
                    
                    if total_chunks > 0:
                        st.success(f"Berhasil meng-index {total_chunks} chunk kode!")
                        st.cache_resource.clear()
                    else:
                        st.warning("Tidak ditemukan file kode yang valid.")
                except Exception as e:
                    st.error(f"Gagal memproses repositori: {e}")
        else:
            st.error("Masukkan URL GitHub yang valid!")

# --- MAIN APP INTERFACE (CHAT DLL) ---
st.title("🤖 AI Codebase Onboarding & Architecture Assistant")
# (Lanjutkan dengan kode chat interface, st_mermaid, dan synthesizer seperti sebelumnya...)

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