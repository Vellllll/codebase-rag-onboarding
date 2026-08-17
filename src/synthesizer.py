import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.retriever import HybridCodeRetriever

load_dotenv()

# Perbarui System Prompt agar Agent sadar punya tool baru
SYSTEM_PROMPT = """Anda adalah Senior Software Architect & Codebase Onboarding Mentor. 

ATURAN PENGGUNAAN TOOLS:
1. `search_codebase`: Gunakan pertama kali untuk mencari potongan kode spesifik (fungsi/class) dari Vector DB.
2. `list_directory`: Gunakan untuk melihat struktur folder/isi direktori proyek lokal.
3. `read_file_content`: Gunakan untuk membaca SATU FILE UTUH jika chunk dari `search_codebase` terpotong atau Anda butuh melihat bagian import/konfigurasi global.
4. `web_search`: Gunakan HANYA jika informasi butuh referensi dokumentasi internet.

ATURAN JAWABAN:
- Selalu sertakan file path jika merujuk ke kode.
- Jika pengguna bertanya tentang Git/Author, berikan Nama Author, Hash, dan Pesan Commit dari metadata.
- Buat diagram ```mermaid ... ``` jika user meminta visualisasi arsitektur.
"""

class CodebaseSynthesizer:
    def __init__(self, model_name: str = "deepseek-chat"):
        self.retriever = HybridCodeRetriever()
        self.last_sources = []
        
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY tidak ditemukan di file .env!")
            
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.2
        )
        
        # Direktori basis repositori yang sudah di-clone (Langkah 1)
        self.base_repo_path = os.path.abspath("./cloned_repo")
        
        # 1. Definisi Tool Web Search
        try:
            self.web_search = DuckDuckGoSearchRun(name="web_search")
        except Exception as e:
            print(f"⚠️ Warning: Gagal memuat web_search tool: {e}")
            self.web_search = None

        # 2. DEFINISIKAN TOOLS DULU SEBELUM DIMASUKKAN KE LIST
        @tool
        def search_codebase(query: str) -> str:
            """Mencari potongan struktur kode spesifik dari repositori."""
            docs = self.retriever.get_relevant_code(query, top_k=3)
            self.last_sources = [
                {
                    "file_path": doc.metadata.get("file_path"),
                    "start_line": doc.metadata.get("start_line"),
                    "end_line": doc.metadata.get("end_line"),
                    "name": doc.metadata.get("name")
                } for doc in docs
            ]
            if not docs:
                return "Tidak ada kode relevan di Vector DB."
            return self._format_context(docs)

        @tool
        def list_directory(dir_path: str = "") -> str:
            """Melihat daftar file dan folder."""
            try:
                target_path = os.path.abspath(os.path.join(self.base_repo_path, dir_path))
                if not target_path.startswith(self.base_repo_path):
                    return "Error: Akses ditolak."
                if not os.path.exists(target_path):
                    return f"Error: Direktori '{dir_path}' tidak ditemukan."
                items = os.listdir(target_path)
                return f"Isi direktori '{dir_path}':\n" + "\n".join(items)
            except Exception as e:
                return f"Gagal membaca direktori: {e}"

        @tool
        def read_file_content(file_path: str) -> str:
            """Membaca isi keseluruhan sebuah file."""
            try:
                target_path = os.path.abspath(os.path.join(self.base_repo_path, file_path))
                if not target_path.startswith(self.base_repo_path):
                    return "Error: Akses ditolak."
                if not os.path.exists(target_path):
                    return f"Error: File '{file_path}' tidak ditemukan."
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if len(content) > 15000:
                    return content[:15000] + "\n\n... [SISA KODE DIPOTONG]"
                return content
            except Exception as e:
                return f"Gagal membaca file: {e}"

        # 3. SETELAH SEMUA DIDEFINISIKAN, BARU MASUKKAN KE LIST self.tools
        self.tools = [search_codebase, list_directory, read_file_content]
        if self.web_search:
            self.tools.append(self.web_search)

        # 4. Inisialisasi LangGraph Agent
        self.memory = MemorySaver()
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=self.memory
        )

    # (Biarkan metode _format_context, answer_question, dan stream_answer_events seperti sebelumnya)

    def _format_context(self, docs: List[Document]) -> str:
        formatted_blocks = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata
            file_path = meta.get("file_path", "Unknown File")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            code_type = meta.get("code_type", "code")
            name = meta.get("name", "anonymous")
            
            author = meta.get("author", "Unknown")
            commit_hash = meta.get("commit_hash", "")
            commit_date = meta.get("commit_date", "")
            commit_msg = meta.get("commit_message", "")
            
            block = (
                f"--- [SNIPPET #{idx}] ---\n"
                f"File Path   : {file_path} (Baris {start_line}-{end_line})\n"
                f"Component   : {code_type} -> {name}\n"
                f"Last Commit : {commit_hash} by {author} ({commit_date}) - \"{commit_msg}\"\n"
                f"Code:\n{doc.page_content}\n"
            )
            formatted_blocks.append(block)
        return "\n".join(formatted_blocks)

    def stream_answer_events(self, question: str, target_folders: List[str] = None, thread_id: str = "default_thread"):
        """Menghasilkan (yield) status aktivitas agent satu-persatu dengan dukungan memori."""
        self.last_sources = [] 
        
        if target_folders:
            question = f"{question}\n\n[Catatan Sistem: Fokuskan pencarian codebase pada folder {', '.join(target_folders)} jika memungkinkan.]"

        # Tambahkan recursion_limit untuk mencegah infinite loop!
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20  # Maksimal 10 kali putaran (Thought -> Action)
        }

        # LangGraph Stream Event Loop (Pass config ke dalam stream)
        try:
            for event in self.agent_executor.stream({"messages": [("user", question)]}, config=config):
                if "agent" in event:
                    last_message = event["agent"]["messages"][-1]
                    
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tc in last_message.tool_calls:
                            yield {"type": "tool_start", "tool": tc["name"], "query": tc.get("args")}
                    elif last_message.content:
                        yield {"type": "final_answer", "content": last_message.content}
        except Exception as e:
            # Menangkap error jika batas rekursi terlewati
            yield {"type": "final_answer", "content": "Mohon maaf, saya menghentikan proses pencarian karena instruksi terlalu kompleks dan menyebabkan looping berlebihan. Coba pecah pertanyaan Anda menjadi langkah-langkah yang lebih kecil."}