import streamlit as st
import uuid
import os
import re
import json
import yaml
from datetime import datetime
from yaml.loader import SafeLoader
from streamlit_mermaid import st_mermaid
from dotenv import load_dotenv
import streamlit_authenticator as stauth

# Load environment variables
load_dotenv()

try:
    from src.synthesizer import CodebaseSynthesizer
    from src.indexer import IncrementalCodebaseIndexer
except Exception as e:
    import traceback
    st.error(f"❌ Gagal mengimpor modul internal: {e}")
    traceback.print_exc()
    st.stop()

# =====================================================================
# --- KONFIGURASI HALAMAN ---
# =====================================================================
st.set_page_config(
    page_title="Codebase AI Mentor",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

INDEX_METADATA_PATH = "./qdrant_storage/index_metadata.json"

st.markdown("""
<style>
    .stAppDeployButton {display:none;}
    h1 {font-weight: 700; color: #1E3A8A;}
    .stChatInput {border-radius: 15px;}

    .status-card {
        background-color: #F0F4FF;
        border: 1px solid #C7D6FF;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    .status-card.empty {
        background-color: #FFF7ED;
        border: 1px solid #FED7AA;
    }
    .status-card b { color: #1E3A8A; }
    .status-card.empty b { color: #9A3412; }
    .status-sub { font-size: 0.8rem; color: #6B7280; margin-top: 2px; }

    .source-pill {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# --- HELPER: METADATA INDEX (agar status bertahan antar-reload) ---
# =====================================================================
def load_index_metadata():
    if os.path.exists(INDEX_METADATA_PATH):
        try:
            with open(INDEX_METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_index_metadata(repo_url: str, total_chunks: int):
    os.makedirs(os.path.dirname(INDEX_METADATA_PATH), exist_ok=True)
    with open(INDEX_METADATA_PATH, "w") as f:
        json.dump({
            "repo_url": repo_url,
            "total_chunks": total_chunks,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }, f)


def format_relative_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "baru saja"
        if seconds < 3600:
            return f"{seconds // 60} menit lalu"
        if seconds < 86400:
            return f"{seconds // 3600} jam lalu"
        return f"{seconds // 86400} hari lalu"
    except Exception:
        return ""


def is_valid_github_url(url: str) -> bool:
    return bool(re.match(r"^https://github\.com/[\w.-]+/[\w.-]+/?$", url.strip()))


# =====================================================================
# --- SISTEM LOGIN ---
# =====================================================================
if not os.path.exists('.streamlit/config.yaml'):
    st.error("⚠️ File konfigurasi `.streamlit/config.yaml` tidak ditemukan. Aplikasi tidak bisa dijalankan.")
    st.stop()

with open('.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login(location="main")

if st.session_state["authentication_status"] is False:
    st.error("❌ Username atau password salah!")
    st.stop()
elif st.session_state["authentication_status"] is None:
    st.info("👋 Selamat datang! Silakan login untuk mengeksplorasi codebase.")
    st.stop()

# =====================================================================
# --- APLIKASI UTAMA (SETELAH LOGIN) ---
# =====================================================================

@st.cache_resource
def load_synthesizer():
    return CodebaseSynthesizer()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexing_in_progress" not in st.session_state:
    st.session_state.indexing_in_progress = False
# BIKIN THREAD ID UNTUK MEMORY LANGGRAPH
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

index_meta = load_index_metadata()
has_index = index_meta is not None

# =====================================================================
# --- SIDEBAR ---
# =====================================================================
with st.sidebar:
    st.markdown(f"### 👤 Hai, {st.session_state['name']}!")
    authenticator.logout("Keluar", location="sidebar")
    st.divider()

    # --- STATUS REPO (persisten antar reload) ---
    if has_index:
        st.markdown(
            f"""<div class="status-card">
                ✅ <b>Repo terhubung</b><br>
                <code>{index_meta['repo_url']}</code>
                <div class="status-sub">
                    {index_meta['total_chunks']} chunk kode &middot;
                    diindeks {format_relative_time(index_meta['indexed_at'])}
                </div>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class="status-card empty">
                ⚠️ <b>Belum ada repo terhubung</b>
                <div class="status-sub">Hubungkan repositori di bawah untuk mulai bertanya.</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.header("⚙️ Hubungkan Repositori")

    github_url_input = st.text_input(
        "🔗 GitHub URL",
        placeholder="https://github.com/facebook/react"
    )

    with st.expander("🛠️ Pengaturan Lanjutan (Opsional)", expanded=False):
        github_token_input = st.text_input(
            "🔑 GitHub PAT (Untuk Private Repo)",
            type="password",
            help="Masukkan Personal Access Token jika ini repositori privat."
        )
        target_folders_input = st.text_input(
            "📁 Target Folder (saat Indexing)",
            placeholder="src/api, libs/utils",
            help=(
                "Membatasi file mana yang DI-INDEX sejak awal. File di luar folder ini "
                "tidak akan pernah diparse/disimpan sampai kamu re-index. Cocok untuk "
                "monorepo besar biar indexing lebih cepat dan hemat storage."
            )
        )

    index_button_label = "🔄 Re-index Repositori" if has_index else "🚀 Mulai Indexing"
    index_clicked = st.button(
        index_button_label,
        use_container_width=True,
        type="primary",
        disabled=st.session_state.indexing_in_progress
    )

    if index_clicked:
        url_clean = github_url_input.strip()
        if not url_clean:
            st.error("Isi dulu GitHub URL-nya, ya.")
        elif not is_valid_github_url(url_clean):
            st.error("Format URL GitHub tidak valid. Contoh: https://github.com/user/repo")
        else:
            st.session_state.indexing_in_progress = True
            try:
                with st.status("📦 Memproses Repositori...", expanded=True) as status:
                    st.write("⬇️ Mengunduh repository...")
                    indexer = IncrementalCodebaseIndexer()

                    st.write("🌳 Membedah Abstract Syntax Tree (AST)...")
                    total_chunks = indexer.index_from_github_url(
                        github_url=url_clean,
                        github_token=github_token_input if github_token_input else None,
                        target_folders_str=target_folders_input
                    )

                    if total_chunks > 0:
                        st.cache_resource.clear()
                        save_index_metadata(url_clean, total_chunks)
                        st.session_state.messages = []
                        status.update(
                            label=f"✅ Berhasil! {total_chunks} chunk kode tersimpan.",
                            state="complete", expanded=False
                        )
                        st.toast("🎉 Indexing selesai! Kamu bisa mulai bertanya.", icon="✅")
                    else:
                        status.update(
                            label="⚠️ Tidak ada file kode yang cocok untuk diproses.",
                            state="error"
                        )
            except Exception as e:
                st.error(f"❌ Detail Error: {e}")
                # Print ke console agar LOG STREAMLIT CLOUD PASTI MENCETAK ERRORNYA
                import traceback
                traceback.print_exc()
            finally:
                # Pastikan state selalu di-reset agar tombol tidak mengunci diam-diam
                st.session_state.indexing_in_progress = False
                st.rerun()

    st.divider()
    st.header("🔍 Filter Pencarian Chat")
    chat_target_folders = st.text_input(
        "Fokuskan AI pada folder (saat Chat):",
        placeholder="src/components, src/api",
        help=(
            "Membatasi PENCARIAN dari index yang sudah ada — tidak perlu re-index, "
            "bisa diganti kapan saja per pertanyaan. Index-nya sendiri tidak berubah. "
            "Catatan: kalau folder ini tidak termasuk dalam Target Folder saat indexing, "
            "hasilnya akan kosong karena memang belum pernah di-index. "
            "Kosongkan untuk mencari di seluruh repositori yang sudah ter-index."
        ),
        disabled=not has_index
    )

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4()) # RESET INGATAN AGENT
        st.rerun()

    st.caption("✨ **Fitur Aktif:** Hybrid Search, RRF, Cross-Encoder, Mermaid Diagrams, Folder Filtering.")

# =====================================================================
# --- MAIN CHAT INTERFACE ---
# =====================================================================
st.title("🧩 Codebase AI Mentor")

if not has_index:
    st.info("💡 **Mulai di sini:** Hubungkan repositori GitHub di panel sebelah kiri sebelum memulai obrolan.")

if not st.session_state.messages:
    st.markdown("### Coba tanyakan hal seperti ini:")
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔍 Di mana letak logika Autentikasi User?", use_container_width=True, disabled=True)
        st.button("📊 Buatkan diagram alur untuk checkout keranjang", use_container_width=True, disabled=True)
    with col2:
        st.button("⚙️ Jelaskan arsitektur folder src/components", use_container_width=True, disabled=True)
        st.button("👤 Siapa yang terakhir mengubah fungsi login?", use_container_width=True, disabled=True)


def extract_mermaid_code(text: str):
    pattern = r"```mermaid\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def render_sources(sources):
    with st.expander(f"📍 Sumber Kode ({len(sources)} referensi)"):
        for src in sources:
            st.markdown(
                f"<span class='source-pill'>{src['file_path']} : "
                f"{src['start_line']}-{src['end_line']}</span> "
                f"**{src['name']}**",
                unsafe_allow_html=True
            )


# Render Chat History
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message.get("mermaid_code"):
            st.caption("📊 Diagram Arsitektur")
            st_mermaid(message["mermaid_code"])

        if message.get("sources"):
            render_sources(message["sources"])

# User Input
chat_placeholder = (
    "Tanyakan sesuatu tentang codebase ini..."
    if has_index else
    "Hubungkan repositori dulu di panel kiri untuk mulai bertanya"
)

if prompt := st.chat_input(chat_placeholder, disabled=not has_index):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            synthesizer = load_synthesizer()
        except Exception as e:
            st.error(f"Gagal memuat mesin AI: {e}")
            st.stop()

        target_folders_list = None
        if chat_target_folders and chat_target_folders.strip():
            target_folders_list = [
                folder.strip()
                for folder in chat_target_folders.split(",")
                if folder.strip()
            ]

        # Inisialisasi UI dinamis
        # Inisialisasi UI dinamis
        # Ubah expanded menjadi False agar kotaknya tertutup sejak awal
        status = st.status("🧠 AI sedang berpikir...", expanded=False) 
        message_placeholder = st.empty()
        full_answer = ""
        
        import time
        try:
            # Membaca kejadian (event) dari Agent (DENGAN MEMORI)
            stream_generator = synthesizer.stream_answer_events(
                prompt, 
                target_folders=target_folders_list,
                thread_id=st.session_state.thread_id  # <--- PARAMETER BARU
            )
            
            for event in stream_generator:
                if event["type"] == "tool_start":
                    tool_name = event["tool"]
                    if tool_name == "search_codebase":
                        status.update(label="🔍 Sedang mencari relevansi kode (Vector Search)...")
                    elif tool_name == "list_directory":
                        status.update(label="📁 Sedang menjelajahi struktur folder repositori...")
                    elif tool_name == "read_file_content":
                        status.update(label="📄 Sedang membaca isi file secara utuh...")
                    elif tool_name == "web_search":
                        status.update(label="🌐 Sedang mencari referensi/dokumentasi dari internet...")
                        
                elif event["type"] == "final_answer":
                    # Update label saat selesai
                    status.update(label="✅ Selesai menganalisis", state="complete")
                    full_answer = event["content"]
                    
                    # Simulasi efek mengetik (typing effect)
                    typed_text = ""
                    for word in full_answer.split(" "):
                        typed_text += word + " "
                        message_placeholder.markdown(typed_text + "▌")
                        time.sleep(0.015) 
                    message_placeholder.markdown(full_answer)

        except Exception as e:
            status.update(label="❌ Terjadi Kesalahan", state="error")
            st.error(f"Gagal memproses jawaban: {e}")
            st.stop()

        # Ekstrak diagram Mermaid jika ada
        mermaid_code = extract_mermaid_code(full_answer)
        if mermaid_code:
            st.caption("📊 Diagram Arsitektur")
            st_mermaid(mermaid_code)

        # Mengambil daftar file kode sumber hasil penelusuran agent
        sources = synthesizer.last_sources
        if sources:
            if target_folders_list:
                st.caption(f"🔍 Pencarian difokuskan pada folder: {', '.join(target_folders_list)}")
            render_sources(sources)

        # Simpan ke riwayat chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "mermaid_code": mermaid_code,
            "sources": sources
        })