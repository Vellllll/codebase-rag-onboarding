import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

load_dotenv()

class HybridCodeRetriever:
    def __init__(
        self, 
        vector_store_path: str = "./qdrant_storage", 
        collection_name: str = "codebase_rag"
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Gunakan QdrantVectorStore.from_existing_collection
        self.qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            path=vector_store_path,
            collection_name=collection_name
        )
        
        # Ambil client qdrant internal
        print("⚡ Memuat dokumen dari Vector DB untuk indeks BM25...")
        all_docs_response = self.qdrant.client.scroll(
            collection_name=collection_name, 
            limit=10000, 
            with_payload=True
        )
        
        self.all_docs: List[Document] = []
        for point in all_docs_response[0]:
            payload = point.payload
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            self.all_docs.append(Document(page_content=page_content, metadata=metadata))

        # Tokenisasi kode sederhana untuk BM25
        corpus = [doc.page_content.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(corpus)

        # 3. Load Reranker Model (Flashrank: Cepat, Lokal, Ringan)
        self.reranker = Ranker()

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Pencarian berdasarkan kata kunci eksak menggunakan BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Ambil top_k indeks dengan skor tertinggi
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.all_docs[i] for i in top_indices if scores[i] > 0]

    def _reciprocal_rank_fusion(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document], 
        k: int = 60
    ) -> List[Document]:
        """Menggabungkan skor dari Vector Search & BM25 Search menggunakan RRF."""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        def add_ranks(docs: List[Document]):
            for rank, doc in enumerate(docs):
                # Gunakan ID unik dari metadata jika ada, atau fallback ke konten
                doc_id = doc.metadata.get("id", doc.page_content)
                doc_map[doc_id] = doc
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (k + rank + 1)

        add_ranks(vector_docs)
        add_ranks(bm25_docs)

        # Urutkan dokumen berdasarkan skor RRF gabungan
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids]

    def get_relevant_code(self, query: str, top_k: int = 3) -> List[Document]:
        """Pipeline Utama: Vector + BM25 -> RRF -> Rerank -> Top-K Result"""
        print(f"\n🔍 Menelusuri Kode untuk Query: '{query}'")

        # Step A: Vector Search (Semantic)
        vector_results = self.qdrant.similarity_search(query, k=10)
        
        # Step B: BM25 Search (Keyword Exact Match)
        bm25_results = self._bm25_search(query, top_k=10)

        # Step C: Combine with Reciprocal Rank Fusion (RRF)
        hybrid_results = self._reciprocal_rank_fusion(vector_results, bm25_results)

        if not hybrid_results:
            return []

        # Step D: Re-ranking dengan Flashrank
        passages = [
            {
                "id": idx,
                "text": f"File: {doc.metadata.get('file_path')}\nFunction: {doc.metadata.get('name')}\nCode:\n{doc.page_content}",
                "meta": doc.metadata
            }
            for idx, doc in enumerate(hybrid_results)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)

        # Step E: Ambil top_k terbaik setelah di-rerank
        final_docs = []
        for res in reranked_results[:top_k]:
            original_doc = hybrid_results[res["id"]]
            final_docs.append(original_doc)

        return final_docs

# --- UJI COBA HYBRID RETRIEVER ---
if __name__ == "__main__":
    # Pastikan kamu sudah menjalankan indexer.py di Step 2 sebelumnya!
    try:
        retriever = HybridCodeRetriever()
        
        # Uji coba pencarian semantik & keyword
        results = retriever.get_relevant_code("Di mana fungsi autentikasi token berada?", top_k=2)
        
        print(f"\n✅ DITEMUKAN {len(results)} CHUNK KODE PALING RELEVAN:")
        for idx, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"\n--- Hasil #{idx} ---")
            print(f"📁 File     : {meta.get('file_path')} (Baris {meta.get('start_line')}-{meta.get('end_line')})")
            print(f"⚙️ Component: {meta.get('code_type')} -> {meta.get('name')}")
            print(f"📝 Snippet  :\n{doc.page_content[:150]}...")
    except Exception as e:
        print(f"❌ Gagal memuat retriever. Pastikan qdrant_storage sudah dibuat di Step 2. Error: {e}")