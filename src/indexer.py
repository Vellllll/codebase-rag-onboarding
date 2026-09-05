import os
import re
import select
import subprocess
import threading
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from git import Repo, GitCommandError
import tempfile
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from src.ast_parser import ASTCodeParser, CodeChunk

load_dotenv()


class IndexingCancelled(Exception):
    """Dilempar ketika user menekan tombol stop di tengah proses indexing."""
    pass


def _check_cancelled(cancel_event: Optional[threading.Event]):
    if cancel_event is not None and cancel_event.is_set():
        raise IndexingCancelled("Indexing dibatalkan oleh pengguna.")


EXCLUDE_DIRS = {
    "node_modules", ".git", "venv", "env", "__pycache__",
    "dist", "build", ".next", ".idea", ".vscode"
}
SUPPORTED_EXTENSIONS = {".ts", ".js", ".tsx", ".jsx", ".py"}


class IncrementalCodebaseIndexer:
    def __init__(self, collection_name: str = "codebase_rag"):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        # Parser sekarang memilih grammar tree-sitter secara otomatis per file
        # (lihat ast_parser.py) alih-alih hardcode ke satu bahasa.
        self.parser = ASTCodeParser()

    # -----------------------------------------------------------------
    # Git Blame: SATU panggilan `git blame` per FILE (bukan per chunk).
    # Hasilnya dipetakan baris -> metadata commit, lalu dipakai ulang
    # untuk semua chunk di file yang sama. Ini menghindari puluhan
    # subprocess `git blame` terpisah untuk file dengan banyak fungsi.
    # -----------------------------------------------------------------
    def _build_file_blame_map(self, repo: Repo, file_path: str) -> Dict[int, Dict[str, Any]]:
        blame_map: Dict[int, Dict[str, Any]] = {}
        try:
            rel_path = os.path.relpath(file_path, repo.working_dir)
            blame_data = repo.blame('HEAD', rel_path)
            if not blame_data:
                return blame_map

            line_no = 1
            for commit, lines in blame_data:
                meta = {
                    "author": commit.author.name,
                    "author_email": commit.author.email,
                    "commit_hash": commit.hexsha[:7],
                    "commit_date": commit.committed_datetime.strftime("%Y-%m-%d %H:%M"),
                    "commit_message": commit.message.strip().split("\n")[0],
                }
                for _ in lines:
                    blame_map[line_no] = meta
                    line_no += 1
        except Exception:
            pass

        return blame_map

    @staticmethod
    def _lookup_blame(blame_map: Dict[int, Dict[str, Any]], line_number: int) -> Dict[str, Any]:
        meta = blame_map.get(line_number)
        if meta:
            return meta
        return {
            "author": "Unknown",
            "commit_hash": "HEAD",
            "commit_date": "N/A",
            "commit_message": "No commit history",
        }

    def walk_and_parse(
        self,
        repo_path: str,
        changed_files_only: List[str] = None,
        target_folders: List[str] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> List[Document]:
        """Memindai file di repositori dan menyisipkan metadata Git, mendukung filter target folder."""
        documents = []

        git_repo = None
        try:
            git_repo = Repo(repo_path, search_parent_directories=True)
        except Exception:
            print("  Folder bukan repositori Git. Metadata Git Blame akan dilewati.")

        print(f"  Memulai pemindaian kode di: {repo_path}")
        if target_folders:
            print(f"  Fokus indeksasi hanya pada folder: {target_folders}")

        for root, dirs, files in os.walk(repo_path):
            _check_cancelled(cancel_event)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in SUPPORTED_EXTENSIONS:
                    _check_cancelled(cancel_event)
                    file_path = os.path.join(root, file)

                    # --- FILTER FOLDER ---
                    if target_folders:
                        rel_file_path = os.path.relpath(file_path, repo_path)
                        is_targeted = False

                        for target in target_folders:
                            target = os.path.normpath(target)
                            if rel_file_path == target or rel_file_path.startswith(target + os.sep):
                                is_targeted = True
                                break

                        if not is_targeted:
                            continue
                    # -------------------------------------

                    if changed_files_only and file_path not in changed_files_only:
                        continue

                    try:
                        chunks: List[CodeChunk] = self.parser.parse_file(file_path)
                        if not chunks:
                            continue

                        # Satu blame per file, dipakai ulang untuk semua chunk-nya.
                        blame_map: Dict[int, Dict[str, Any]] = {}
                        if git_repo:
                            blame_map = self._build_file_blame_map(git_repo, file_path)

                        for chunk in chunks:
                            git_meta = self._lookup_blame(blame_map, chunk.start_line)
                            doc = Document(
                                page_content=chunk.content,
                                metadata={
                                    "id": chunk.id,
                                    "file_path": chunk.file_path,
                                    "code_type": chunk.code_type,
                                    "name": chunk.name,
                                    "start_line": chunk.start_line,
                                    "end_line": chunk.end_line,
                                    "author": git_meta.get("author", "Unknown"),
                                    "commit_hash": git_meta.get("commit_hash", "HEAD"),
                                    "commit_date": git_meta.get("commit_date", "N/A"),
                                    "commit_message": git_meta.get("commit_message", "")
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"  [ERROR INDEXER] Gagal memproses {file_path}: {e}")
                        # Jangan ditelan diam-diam, tampilkan log ke console
                        import sys
                        sys.stdout.flush()

        print(f"  Total {len(documents)} chunk berhasil diproses!")
        return documents

    @staticmethod
    def _parse_clone_percent(line: str) -> Optional[float]:
        match = re.search(r"(\d{1,3})%", line)
        if not match:
            return None
        return min(max(int(match.group(1)) / 100.0, 0.0), 1.0)

    def _clone_with_progress(
        self,
        clone_url: str,
        repo_dir: str,
        progress_callback=None,
        cancel_event: Optional[threading.Event] = None
    ):
        """
        Clone manual via subprocess (bukan GitPython Repo.clone_from) supaya kita
        pegang handle proses `git` secara langsung -> bisa di-terminate() paksa
        saat user menekan tombol stop, dan bisa di-poll tanpa block permanen
        lewat select() dengan timeout.
        """
        cmd = ["git", "clone", "--progress", "--depth", "1", "--", clone_url, repo_dir]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        last_fraction = 0.0
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise IndexingCancelled("Indexing dibatalkan oleh pengguna saat proses clone.")

                ready, _, _ = select.select([proc.stderr], [], [], 0.5)
                if ready:
                    line = proc.stderr.readline()
                    if not line:
                        break
                    line = line.strip()
                    fraction = self._parse_clone_percent(line)
                    if fraction is not None:
                        last_fraction = fraction
                    if progress_callback:
                        progress_callback(last_fraction, line)
                elif proc.poll() is not None:
                    break
        finally:
            returncode = proc.wait()

        if returncode != 0:
            stderr_tail = proc.stderr.read() if proc.stderr else ""
            raise GitCommandError(["git", "clone"], returncode, stderr_tail)

    def index_from_github_url(
        self,
        github_url: str,
        github_token: str = None,
        target_folders_str: str = "",
        progress_callback=None,
        cancel_event: Optional[threading.Event] = None
    ) -> int:
        """Meng-clone repo ke folder permanen (cloned_repo), mem-parse AST, lalu membiarkannya agar bisa diakses oleh File Explorer Agent."""

        # 1. Tentukan folder permanen
        repo_dir = os.path.abspath("./cloned_repo")

        # Bersihkan folder jika sudah ada repo sebelumnya
        if os.path.exists(repo_dir):
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)

        os.makedirs(repo_dir, exist_ok=True)
        print(f"  Cloning repo ke direktori permanen: {repo_dir}")

        clone_url = github_url.strip()
        if github_token and github_token.strip():
            token = github_token.strip()
            if clone_url.startswith("https://"):
                clone_url = clone_url.replace("https://", f"https://{token}@")

        try:
            self._clone_with_progress(clone_url, repo_dir, progress_callback, cancel_event)

            target_folders = None
            if target_folders_str and target_folders_str.strip():
                target_folders = [folder.strip() for folder in target_folders_str.split(",") if folder.strip()]

            # Parse dan Index ke Qdrant
            docs = self.walk_and_parse(repo_dir, target_folders=target_folders, cancel_event=cancel_event)

            if docs:
                self.index_to_qdrant(docs, force_recreate=True, cancel_event=cancel_event)
                print(f"  Berhasil meng-index repositori dari URL: {github_url}")
                return len(docs)
            else:
                print("  Tidak ada file kode yang valid untuk di-index.")
                return 0

        except GitCommandError as e:
            print(f"  Error Git Cloning: {e}")
            raise Exception(f"Gagal meng-clone repositori: {e}")
        except IndexingCancelled:
            print("  Indexing dibatalkan oleh pengguna. Membersihkan folder clone...")
            shutil.rmtree(repo_dir, ignore_errors=True)
            raise

        # BLOK FINALLY (shutil.rmtree) DIHAPUS agar folder tidak lenyap saat error biasa!

    def _get_qdrant_connection_kwargs(self, vector_store_path: str) -> Dict[str, Any]:
        """
        Qdrant lokal berbasis `path=` hanya aman diakses oleh SATU proses pada satu
        waktu. Di proyek ini, Streamlit app dan webhook server (server.py) bisa
        mengakses koleksi yang sama secara bersamaan -> resiko file lock/corruption.

        Kalau QDRANT_URL diset di .env, kita pakai Qdrant dalam mode server (docker/
        Qdrant Cloud) yang memang didesain untuk multi-proses/multi-writer. Kalau
        tidak diset, tetap fallback ke mode lokal seperti sebelumnya, tapi user diberi
        peringatan eksplisit di log supaya tahu ini bukan setup yang direkomendasikan
        untuk dipakai bersamaan dengan webhook server.
        """
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            # Default httpx timeout terlalu pendek untuk batch upsert berisi chunk kode
            # yang besar (repo besar -> banyak dokumen sekaligus per batch) -> WriteTimeout.
            kwargs: Dict[str, Any] = {"url": qdrant_url, "timeout": int(os.getenv("QDRANT_TIMEOUT", "60"))}
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if qdrant_api_key:
                kwargs["api_key"] = qdrant_api_key
            return kwargs

        print(
            "  ⚠️ QDRANT_URL tidak diset — menggunakan Qdrant mode lokal (path=). "
            "Mode ini TIDAK aman diakses bersamaan oleh Streamlit app dan webhook "
            "server sekaligus. Untuk produksi/pemakaian bersamaan, jalankan Qdrant "
            "sebagai server (docker run qdrant/qdrant) dan set QDRANT_URL di .env."
        )
        return {"path": vector_store_path}

    def index_to_qdrant(
        self,
        documents: List[Document],
        vector_store_path: str = "./qdrant_storage",
        force_recreate: bool = True,
        cancel_event: Optional[threading.Event] = None,
        batch_size: int = 32
    ):
        print(f"⚡ Meng-index {len(documents)} dokumen ke Qdrant (force_recreate={force_recreate})...")

        connection_kwargs = self._get_qdrant_connection_kwargs(vector_store_path)

        # Upload batch pertama lewat from_documents (bikin/reset koleksi), sisanya
        # di-upload manual per-batch supaya ada titik pengecekan cancel_event di
        # tengah proses upload (repo besar -> ratusan batch, bisa makan waktu lama).
        first_batch, rest = documents[:batch_size], documents[batch_size:]
        qdrant = QdrantVectorStore.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            force_recreate=force_recreate,
            batch_size=batch_size,
            **connection_kwargs
        )

        for i in range(0, len(rest), batch_size):
            _check_cancelled(cancel_event)
            qdrant.add_documents(rest[i:i + batch_size])

        print("✅ Indeksasi Qdrant Selesai!")
        return qdrant