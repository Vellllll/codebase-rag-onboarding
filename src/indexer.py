import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from git import Repo, GitCommandError
import tempfile
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from src.ast_parser import ASTCodeParser, CodeChunk

load_dotenv()

EXCLUDE_DIRS = {
    "node_modules", ".git", "venv", "env", "__pycache__", 
    "dist", "build", ".next", ".idea", ".vscode"
}
SUPPORTED_EXTENSIONS = {".ts", ".js", ".tsx", ".jsx", ".py"}

class IncrementalCodebaseIndexer:
    def __init__(self, collection_name: str = "codebase_rag"):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.parser = ASTCodeParser("typescript")

    def _get_git_metadata(self, repo: Repo, file_path: str, line_number: int) -> Dict[str, Any]:
        """Mengambil informasi Git Blame (Author, Date, Commit Message) untuk baris kode spesifik."""
        try:
            # Ambil relasi path terhadap root repositori
            rel_path = os.path.relpath(file_path, repo.working_dir)
            
            # Jalankan git blame pada baris tertentu
            blame_data = repo.blame('HEAD', rel_path, L=f"{line_number},{line_number}")
            if blame_data:
                commit, lines = blame_data[0]
                return {
                    "author": commit.author.name,
                    "author_email": commit.author.email,
                    "commit_hash": commit.hexsha[:7],
                    "commit_date": commit.committed_datetime.strftime("%Y-%m-%d %H:%M"),
                    "commit_message": commit.message.strip().split("\n")[0]
                }
        except Exception:
            pass
            
        return {
            "author": "Unknown",
            "commit_hash": "HEAD",
            "commit_date": "N/A",
            "commit_message": "No commit history"
        }

    def walk_and_parse(self, repo_path: str, changed_files_only: List[str] = None) -> List[Document]:
        """Memindai file di repositori dan menyisipkan metadata Git."""
        documents = []
        
        # Inisialisasi Git Repo jika folder tersebut adalah git repository
        git_repo = None
        try:
            git_repo = Repo(repo_path, search_parent_directories=True)
        except Exception:
            print("⚠️ Folder bukan repositori Git. Metadata Git Blame akan dilewati.")

        print(f"📂 Memulai pemindaian kode di: {repo_path}")
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    
                    # Jika mode incremental aktif, lewati file yang tidak mengalami perubahan
                    if changed_files_only and file_path not in changed_files_only:
                        continue

                    try:
                        chunks: List[CodeChunk] = self.parser.parse_file(file_path)
                        
                        for chunk in chunks:
                            # Ekstrak Git Metadata jika git repo tersedia
                            git_meta = {}
                            if git_repo:
                                git_meta = self._get_git_metadata(git_repo, file_path, chunk.start_line)

                            doc = Document(
                                page_content=chunk.content,
                                metadata={
                                    "id": chunk.id,
                                    "file_path": chunk.file_path,
                                    "code_type": chunk.code_type,
                                    "name": chunk.name,
                                    "start_line": chunk.start_line,
                                    "end_line": chunk.end_line,
                                    # Git Metadata
                                    "author": git_meta.get("author", "Unknown"),
                                    "commit_hash": git_meta.get("commit_hash", "HEAD"),
                                    "commit_date": git_meta.get("commit_date", "N/A"),
                                    "commit_message": git_meta.get("commit_message", "")
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"❌ Error parsing {file_path}: {e}")

        print(f"✅ Total {len(documents)} chunk berhasil diproses!")
        return documents

    def index_to_qdrant(self, documents: List[Document], vector_store_path: str = "./qdrant_storage", force_recreate: bool = True):
        print(f"⚡ Meng-index {len(documents)} dokumen ke Qdrant (force_recreate={force_recreate})...")
        
        qdrant = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=vector_store_path,
            collection_name=self.collection_name,
            force_recreate=force_recreate  # True untuk full re-index, False untuk incremental update
        )
        print("✅ Indeksasi Qdrant Selesai!")
        return qdrant
    
    def index_from_github_url(self, github_url: str):
        """Meng-clone repo publik ke temporary directory, mem-parse AST, lalu menghapus temp folder."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"📥 Cloned repo temporarily to: {temp_dir}")

        try:
            # 1. Clone repository dari GitHub
            Repo.clone_from(github_url, temp_dir, depth=1) # depth=1 agar clone sangat cepat (shallow clone)
            
            # 2. Parse AST & Extract Chunks
            docs = self.walk_and_parse(temp_dir)
            
            if docs:
                # 3. Index ke Qdrant Vector Store
                self.index_to_qdrant(docs, force_recreate=True)
                print(f"✅ Berhasil meng-index repositori dari URL: {github_url}")
                return len(docs)
            else:
                print("⚠️ Tidak ada file kode yang valid untuk di-index.")
                return 0

        finally:
            # 4. Hapus folder temporary agar tidak memakan penyimpanan lokal
            shutil.rmtree(temp_dir)
            print("🧹 Temporary folder berhasil dibersihkan.")