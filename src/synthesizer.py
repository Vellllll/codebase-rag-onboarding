import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <-- IMPORT BARU

from src.retriever import HybridCodeRetriever

load_dotenv()

SYSTEM_PROMPT = """Anda adalah Senior Software Architect & Codebase Onboarding Mentor. 
Tugas Anda adalah membantu developer memahami arsitektur kode dan riwayat perubahan repositori, 
serta mencari informasi dari internet jika diperlukan (misalnya untuk dokumentasi eksternal).

ATURAN UTAMA:
1. Gunakan tool `search_codebase` untuk mencari konteks dari repositori lokal.
2. Gunakan tool `web_search` JIKA informasi tidak ada di codebase atau butuh referensi dokumentasi terbaru dari internet.
3. JIKA USER BERTANYA TENTANG RIWAYAT/PEMBUAT KODE: Sebutkan Nama Author, Commit Hash, Tanggal, dan Pesan Commit dari metadata yang tersedia.
4. REFERENSI LOKASI: Selalu sertakan file path dan nomor baris (contoh: `src/auth.ts` Baris 12-30) jika merujuk ke kode.
5. DIAGRAM MERMAID.JS: Buat diagram ```mermaid ... ``` jika user meminta gambaran alur data/arsitektur.
"""

class CodebaseSynthesizer:
    def __init__(self, model_name: str = "deepseek-chat"):
        self.retriever = HybridCodeRetriever()
        
        # Variabel untuk melacak kode lokal yang ditemukan oleh Tool
        self.last_sources = []
        
        # --- KONFIGURASI DEEPSEEK API ---
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY tidak ditemukan di file .env!")
            
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.2
        )
        
        # 1. Definisi Tools untuk Agent
        self.web_search = DuckDuckGoSearchRun(
            name="web_search",
            description="Gunakan ini untuk mencari informasi umum, dokumentasi library, atau error di internet."
        )
        
        @tool
        def search_codebase(query: str) -> str:
            """Gunakan tool ini HANYA untuk mencari struktur kode, fungsi, class, dan riwayat git di dalam repositori lokal."""
            docs = self.retriever.get_relevant_code(query, top_k=3)
            
            # Simpan metadata dokumen agar UI bisa menampilkannya
            self.last_sources = [
                {
                    "file_path": doc.metadata.get("file_path"),
                    "start_line": doc.metadata.get("start_line"),
                    "end_line": doc.metadata.get("end_line"),
                    "name": doc.metadata.get("name")
                } for doc in docs
            ]
            
            if not docs:
                return "Tidak ada kode yang relevan ditemukan di repositori."
            return self._format_context(docs)

        self.tools = [search_codebase, self.web_search]
        
        # 2. Inisialisasi Memory dan Agent LangGraph
        self.memory = MemorySaver()  # <-- INISIALISASI MEMORY
        
        self.agent_executor = create_react_agent(
            model=self.llm, 
            tools=self.tools, 
            prompt=SYSTEM_PROMPT,
            checkpointer=self.memory  # <-- PASANG MEMORY KE AGENT
        )

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

        # Konfigurasi unik per percakapan agar agent ingat konteks
        config = {"configurable": {"thread_id": thread_id}}

        # LangGraph Stream Event Loop (Pass config ke dalam stream)
        for event in self.agent_executor.stream({"messages": [("user", question)]}, config=config):
            if "agent" in event:
                last_message = event["agent"]["messages"][-1]
                
                # Jika agent memutuskan memanggil tools
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        yield {"type": "tool_start", "tool": tc["name"], "query": tc.get("args")}
                # Jika agent memberikan jawaban akhir
                elif last_message.content:
                    yield {"type": "final_answer", "content": last_message.content}