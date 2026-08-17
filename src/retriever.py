import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

# Import modul Filter dari qdrant_client
from qdrant_client.http import models

load_dotenv()

# Batas keamanan agar proses tidak kehabisan memori pada repo raksasa. Berbeda
# dari batas lama (scroll limit=10000 yang diam-diam memotong hasil tanpa
# pemberitahuan), di sini kita betul-betul mengambil SEMUA dokumen lewat
# paginasi, dan baru berhenti + memberi peringatan jelas kalau jumlah dokumen
# melebihi batas ini.
BM25_CORPUS_HARD_CAP = int(os.getenv("BM25_CORPUS_HARD_CAP", "100000"))
SCROLL_BATCH_SIZE = 1000


class HybridCodeRetriever:
    def __init__(
        self,
        vector_store_path: str = "./qdrant_storage",
        collection_name: str = "codebase_rag"
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.collection_name = collection_name

        connection_kwargs = self._get_qdrant_connection_kwargs(vector_store_path)

        self.qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=collection_name,
            **connection_kwargs
        )

        print("  Memuat dokumen dari Vector DB untuk indeks BM25...")
        self.all_docs: List[Document] = self._load_all_documents(collection_name)

        corpus = [doc.page_content.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(corpus) if corpus else None
        self.reranker = Ranker()

    @staticmethod
    def _get_qdrant_connection_kwargs(vector_store_path: str) -> Dict[str, Any]:
        """Sama seperti di indexer.py: pakai Qdrant server (QDRANT_URL) kalau ada,
        jika tidak fallback ke mode lokal berbasis path (single-process only)."""
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            kwargs: Dict[str, Any] = {"url": qdrant_url}
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if qdrant_api_key:
                kwargs["api_key"] = qdrant_api_key
            return kwargs
        return {"path": vector_store_path}

    def _load_all_documents(self, collection_name: str) -> List[Document]:
        """
        Memuat SELURUH dokumen dari Qdrant lewat paginasi `scroll`, bukan satu
        panggilan dengan limit tetap (limit=10000) yang dulu diam-diam memotong
        koleksi besar tanpa pemberitahuan ke siapa pun.
        """
        docs: List[Document] = []
        next_offset = None

        while True:
            points, next_offset = self.qdrant.client.scroll(
                collection_name=collection_name,
                limit=SCROLL_BATCH_SIZE,
                offset=next_offset,
                with_payload=True,
            )

            for point in points:
                payload = point.payload or {}
                page_content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                docs.append(Document(page_content=page_content, metadata=metadata))

            if len(docs) >= BM25_CORPUS_HARD_CAP:
                print(
                    f"  ⚠️ Jumlah dokumen mencapai batas aman ({BM25_CORPUS_HARD_CAP}). "
                    "Indeks BM25 dibangun dari sebagian koleksi saja. Naikkan "
                    "env var BM25_CORPUS_HARD_CAP jika perlu memuat semuanya."
                )
                break

            if next_offset is None:
                break

        print(f"  {len(docs)} dokumen berhasil dimuat untuk indeks BM25.")
        return docs

    def _is_doc_in_target_folders(self, doc: Document, target_folders: List[str]) -> bool:
        """Helper untuk memfilter dokumen BM25 berdasarkan target folder."""
        file_path = doc.metadata.get("file_path", "")
        for target in target_folders:
            normalized_target = os.path.normpath(target)
            if normalized_target in file_path:
                return True
        return False

    def _bm25_search(self, query: str, top_k: int = 10, target_folders: Optional[List[str]] = None) -> List[Document]:
        """Pencarian BM25 dengan filter target folder opsional."""
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        filtered_docs = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            doc = self.all_docs[i]

            if target_folders and not self._is_doc_in_target_folders(doc, target_folders):
                continue

            filtered_docs.append(doc)
            if len(filtered_docs) >= top_k:
                break

        return filtered_docs

    def _reciprocal_rank_fusion(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60
    ) -> List[Document]:
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        def add_ranks(docs: List[Document]):
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata.get("id", doc.page_content)
                doc_map[doc_id] = doc
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (k + rank + 1)

        add_ranks(vector_docs)
        add_ranks(bm25_docs)

        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids]

    def get_relevant_code(
        self,
        query: str,
        top_k: int = 3,
        target_folders: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Pipeline Utama dengan dukungan Folder Metadata Filtering.
        """
        print(f"\n  Menelusuri Kode untuk Query: '{query}' | Target Folders: {target_folders}")

        qdrant_filter = None
        if target_folders:
            should_conditions = []
            for folder in target_folders:
                should_conditions.append(
                    models.FieldCondition(
                        key="metadata.file_path",
                        match=models.MatchText(text=folder)
                    )
                )
            qdrant_filter = models.Filter(should=should_conditions)

        vector_results = self.qdrant.similarity_search(
            query,
            k=10,
            filter=qdrant_filter
        )

        bm25_results = self._bm25_search(query, top_k=10, target_folders=target_folders)

        hybrid_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        if not hybrid_results:
            return []

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

        final_docs = []
        for res in reranked_results[:top_k]:
            original_doc = hybrid_results[res["id"]]
            final_docs.append(original_doc)

        return final_docs