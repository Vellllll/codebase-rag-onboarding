import os
import hmac
import hashlib
from typing import List
from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from dotenv import load_dotenv

# Import indexer yang sudah mendukung incremental indexing
from src.indexer import IncrementalCodebaseIndexer

load_dotenv()

app = FastAPI(
    title="Codebase RAG Webhook Server",
    description="Automated incremental re-indexing server triggered by GitHub Webhooks."
)

# Ambil Secret Key untuk verifikasi keamanan signature dari GitHub.
# TIDAK ADA fallback default di sini dengan sengaja: default seperti
# "my_super_secret_token" yang sama untuk semua deployment berarti siapa pun
# yang tahu/menebak nilainya bisa memalsukan signature webhook kalau env var
# lupa di-set. Server lebih baik gagal start daripada berjalan dengan secret
# yang bisa ditebak.
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    raise RuntimeError(
        "GITHUB_WEBHOOK_SECRET belum diset di environment/.env. "
        "Server tidak akan dijalankan dengan secret default demi keamanan."
    )

LOCAL_REPO_PATH = os.getenv("LOCAL_REPO_PATH", "./sample_project")


def verify_signature(payload_body: bytes, secret: str, signature_header: str) -> bool:
    """Memverifikasi bahwa request benar-benar berasal dari GitHub (HMAC SHA-256)."""
    if not signature_header:
        return False
    
    hash_object = hmac.new(
        secret.encode("utf-8"), 
        msg=payload_body, 
        digestmod=hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    return hmac.compare_digest(expected_signature, signature_header)


def process_incremental_indexing(changed_files: List[str]):
    """Background Task: Menjalankan re-indexing parsial hanya untuk file yang berubah."""
    print(f"\n🔄 [BACKGROUND TASK] Memulai Incremental Re-indexing...")
    print(f"📁 File yang terdeteksi berubah/ditambah ({len(changed_files)}):")
    for f in changed_files:
        print(f"  - {f}")

    try:
        indexer = IncrementalCodebaseIndexer()
        # Parse hanya file yang berubah
        docs = indexer.walk_and_parse(LOCAL_REPO_PATH, changed_files_only=changed_files)
        
        if docs:
            # force_recreate=False agar data lama tidak terhapus, hanya di-update/ditambah
            indexer.index_to_qdrant(docs, force_recreate=False)
            print("✅ [BACKGROUND TASK] Incremental Re-indexing Selesai!")
        else:
            print("ℹ️ Tidak ada chunk kode baru yang perlu di-index.")
            
    except Exception as e:
        print(f"❌ [BACKGROUND TASK] Gagal melakukan incremental indexing: {e}")


@app.post("/api/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None)
):
    """Endpoint utama penerima payload webhook dari GitHub."""
    
    # 1. Verifikasi Keamanan Signature
    payload_bytes = await request.body()
    if not verify_signature(payload_bytes, WEBHOOK_SECRET, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid X-Hub-Signature-256")

    # 2. Hanya proses event 'push'
    if x_github_event != "push":
        return {"status": "ignored", "reason": f"Event '{x_github_event}' tidak diproses."}

    payload = await request.json()
    
    # 3. Kumpulkan semua file yang ditambah (added) atau diubah (modified) dari commit
    changed_files_set = set()
    commits = payload.get("commits", [])
    
    for commit in commits:
        for added_file in commit.get("added", []):
            full_path = os.path.join(LOCAL_REPO_PATH, added_file)
            changed_files_set.add(full_path)
            
        for modified_file in commit.get("modified", []):
            full_path = os.path.join(LOCAL_REPO_PATH, modified_file)
            changed_files_set.add(full_path)

    changed_files = list(changed_files_set)

    if not changed_files:
        return {"status": "success", "message": "Tidak ada perubahan file kode yang relevan."}

    # 4. Jalankan re-indexing di background task agar GitHub tidak timeout
    background_tasks.add_task(process_incremental_indexing, changed_files)

    return {
        "status": "success",
        "message": f"Webhook diterima. Re-indexing {len(changed_files)} file diproses di background.",
        "files_queued": changed_files
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)