import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.retriever import HybridCodeRetriever

load_dotenv()

SYSTEM_PROMPT = """Anda adalah Senior Software Architect & Codebase Onboarding Mentor.
Tugas Anda adalah membantu developer memahami arsitektur kode dan riwayat perubahan repositori berdasarkan context yang diberikan.

ATURAN UTAMA:
1. JAWAB BERDASARKAN KONTEKS KODE & METADATA GIT YANG DIBERIKAN.
2. JIKA USER BERTANYA TENTANG RIWAYAT/PEMBUAT KODE: Sebutkan Nama Author, Commit Hash, Tanggal, dan Pesan Commit dari metadata yang tersedia.
3. REFERENSI LOKASI: Selalu sertakan file path dan nomor baris (contoh: `src/auth.ts` Baris 12-30).
4. DIAGRAM MERMAID.JS: Buat diagram ```mermaid ... ``` jika user meminta gambaran alur data/arsitektur.

KONTEKS KODE DARI REPOSITORI:
{context}
"""

class CodebaseSynthesizer:
    def __init__(self, model_name: str = "deepseek-chat"):
        self.retriever = HybridCodeRetriever()
        
        # --- KONFIGURASI DEEPSEEK API ---
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY tidak ditemukan di file .env!")

        # Mengarahkan ChatOpenAI ke endpoint DeepSeek
        self.llm = ChatOpenAI(
            model=model_name,                 # "deepseek-chat" atau "deepseek-coder"
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.2
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{question}")
        ])

    def _format_context(self, docs: List[Document]) -> str:
        formatted_blocks = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata
            file_path = meta.get("file_path", "Unknown File")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            code_type = meta.get("code_type", "code")
            name = meta.get("name", "anonymous")
            
            # Metadata Git
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

    def answer_question(self, question: str, target_folders: List[str] = None) -> Dict[str, Any]:
        relevant_docs = self.retriever.get_relevant_code(
            question, 
            top_k=3, 
            target_folders=target_folders
        )
        
        if not relevant_docs:
            return {
                "answer": "Maaf, saya tidak menemukan kode yang relevan di dalam repositori untuk menjawab pertanyaan tersebut.",
                "sources": []
            }

        context_str = self._format_context(relevant_docs)

        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": context_str,
            "question": question
        })

        sources = [
            {
                "file_path": doc.metadata.get("file_path"),
                "start_line": doc.metadata.get("start_line"),
                "end_line": doc.metadata.get("end_line"),
                "name": doc.metadata.get("name")
            }
            for doc in relevant_docs
        ]

        return {
            "answer": response.content,
            "sources": sources
        }
    
    def stream_answer(self, question: str, target_folders: List[str] = None):
        """Streaming jawaban dari LLM token demi token."""
        relevant_docs = self.retriever.get_relevant_code(
            question, 
            top_k=3, 
            target_folders=target_folders
        )
        
        if not relevant_docs:
            yield "Maaf, saya tidak menemukan kode yang relevan di dalam repositori untuk menjawab pertanyaan tersebut."
            return

        context_str = self._format_context(relevant_docs)

        chain = self.prompt_template | self.llm
        
        # Gunakan chain.stream() untuk mengalirkan teks secara real-time
        for chunk in chain.stream({"context": context_str, "question": question}):
            if chunk.content:
                yield chunk.content