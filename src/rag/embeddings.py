# file: rag_embedding_pipeline.py

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings
from rag.extractors import extract_text_from_multiple_files, split_chunk_overlap  # your earlier code
from rag.vector_db import store_documents_in_qdrant, COLLECTION_NAME  # Import vector DB functions

# ---------------- CONFIG ----------------
DATA_DIR = "processing/downloaded_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# ---------------------------------------


def get_all_files(directory=DATA_DIR,conversation_id=None):
    """Recursively collect PDF and Excel files"""
    file_paths = []
    if not conversation_id:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".pdf", ".xlsx", ".xls")):
                    file_paths.append(os.path.join(root, file))
        return file_paths
    else:
        tracking_file = os.path.join('processing', f"{conversation_id}")
        if not os.path.exists(tracking_file):
            print(f"No tracking file found for conversation ID: {conversation_id}")
            return []
        with open(tracking_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    filename = parts[1]
                    filepath = os.path.join(directory, filename)
                    if os.path.exists(filepath):
                        file_paths.append(filepath)
                    else:
                        print(f"File listed in tracking not found: {filepath}")
        return file_paths


def _sanitize_identifier(raw: Optional[str]) -> Optional[str]:
    """Sanitize identifiers exactly like the downloader does to keep filenames in sync."""
    if not raw:
        return None
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in raw)
    return safe.strip() or None


def add_metadata_to_documents(documents, metadata_list):
    """Attach useful metadata (doi/pdf link) based on the naming convention used for downloads."""
    lookup = {}
    for item in metadata_list or []:
        for key in ("doi", "pdf_url", "pdf_link"):
            identifier = _sanitize_identifier(item.get(key))
            if not identifier:
                continue
            lookup.setdefault(identifier.lower(), item)

    for doc in documents:
        filename = os.path.basename(doc.metadata.get("filename", ""))
        base_name, _ = os.path.splitext(filename)
        meta = lookup.get(base_name.lower())
        if not meta:
            continue

        doi = meta.get("doi")
        pdf_url = meta.get("pdf_url") or meta.get("pdf_link")

        if doi:
            doc.metadata["doi"] = doi
        if pdf_url:
            doc.metadata["pdf_link"] = pdf_url

    return documents


class OllamaEmbeddings(Embeddings):
    """Minimal LangChain-compatible embedding wrapper around an Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "toshk0/nomic-embed-text-v2-moe:Q6_K",
        max_workers: Optional[int] = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_workers = max_workers
        self._dimension: Optional[int] = None

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},  # Changed 'input' to 'prompt'
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        
        # Debug: print response if embedding is missing or empty
        if "embedding" not in payload:
            raise ValueError(f"Ollama embeddings response missing 'embedding' key: {payload}")
        
        embedding = payload["embedding"]
        if not embedding or len(embedding) == 0:
            raise ValueError(f"Ollama returned empty embedding. Response: {payload}")
            
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        worker_count = self.max_workers if self.max_workers is not None else min(4, len(texts))
        worker_count = max(1, worker_count)
        if worker_count == 1:
            return [self._embed(text) for text in texts]

        # Requests are IO-bound; a modest worker pool improves throughput without overwhelming Ollama.
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(self._embed, texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def vector_size(self) -> int:
        """Infer the embedding dimensionality once so downstream stores can be configured."""
        if self._dimension is None:
            sample_vector = self._embed("dimension probe")
            self._dimension = len(sample_vector)
        return self._dimension


def embed_and_store(documents, collection_name=COLLECTION_NAME, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Create embeddings using a locally hosted Ollama model and store in an on-disk Qdrant collection.
    """
    embeddings = OllamaEmbeddings()
    
    # Split into chunks
    chunked_docs = split_chunk_overlap(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunked_docs:
        print("No document chunks produced; skipping Qdrant ingestion.")
        return
    
    # Use the vector_db module to store documents
    store_documents_in_qdrant(chunked_docs, embeddings, collection_name)


def extract_and_embed(metadata_list, data_dir=DATA_DIR, collection_name=COLLECTION_NAME):
    """Full pipeline: extract text, attach metadata, embed, store locally"""
    print(f"[1/4] Getting all files from {data_dir}...")
    file_paths = get_all_files(data_dir)
    print(f"      Found {len(file_paths)} files")
    
    print(f"[2/4] Extracting text from files...")
    all_docs = extract_text_from_multiple_files(file_paths)
    print(f"      Extracted {len(all_docs)} document pages")
    
    print(f"[3/4] Adding metadata...")
    all_docs = add_metadata_to_documents(all_docs, metadata_list)
    print(f"      Metadata added to {len(all_docs)} documents")
    
    print(f"[4/4] Embedding and storing in Qdrant...")
    embed_and_store(all_docs, collection_name=collection_name)
    
    return all_docs

def extract_and_embed_conversation(metadata_list, data_dir=DATA_DIR, collection_name=COLLECTION_NAME, conversation_id=None):
    """Full pipeline: extract text, attach metadata, embed, store locally for a specific conversation"""
    print(f"[1/4] Getting all files from {data_dir} for conversation {conversation_id}...")
    file_paths = get_all_files(data_dir, conversation_id=conversation_id)
    print(f"      Found {len(file_paths)} files")
    
    print(f"[2/4] Extracting text from files...")
    all_docs = extract_text_from_multiple_files(file_paths)
    print(f"      Extracted {len(all_docs)} document pages")
    
    print(f"[3/4] Adding metadata...")
    all_docs = add_metadata_to_documents(all_docs, metadata_list)
    print(f"      Metadata added to {len(all_docs)} documents")
    
    print(f"[4/4] Embedding and storing in Qdrant...")
    embed_and_store(all_docs, collection_name=collection_name)
    
    return all_docs
