# file: rag_embedding_pipeline.py

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rag.extractors import extract_text_from_multiple_files, split_chunk_overlap  # your earlier code

# ---------------- CONFIG ----------------
DATA_DIR = "processing/downloaded_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "papers_rag"

# Local Qdrant DB path (change if you want persistent storage)
QDRANT_DB_PATH = "processing/qdrant_local_db"  # folder will be created
# ---------------------------------------
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Lazily create a singleton Qdrant client tied to the local storage path."""
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_DB_PATH)
    return _client


def reset_qdrant_client() -> None:
    """Close and clear the cached Qdrant client so a fresh instance can be created."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_all_files(directory=DATA_DIR):
    """Recursively collect PDF and Excel files"""
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".pdf", ".xlsx", ".xls")):
                file_paths.append(os.path.join(root, file))
    return file_paths


def add_metadata_to_documents(documents, metadata_list):
    """Attach extra metadata from a list of dicts to LangChain Document objects"""
    metadata_dict = {os.path.basename(item.get("title", "")).lower(): item for item in metadata_list}
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("filename", "")).lower()
        if filename in metadata_dict:
            doc.metadata.update(metadata_dict[filename])
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


def ensure_qdrant_collection(collection_name: str, vector_size: int) -> QdrantClient:
    """
    Create the collection if missing, otherwise verify that the existing size matches the embedder.
    If size mismatch, delete and recreate the collection.
    Returns the QdrantClient (fresh instance if collection was recreated).
    """
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name)
    except Exception:
        # Collection doesn't exist, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created new collection '{collection_name}' with vector size {vector_size}")
        return client

    # Qdrant may return either a dict or an object for vector params depending on client version.
    existing_vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if existing_vectors is None and isinstance(getattr(info, "config", None), dict):
        existing_vectors = info["config"].get("params", {}).get("vectors")

    existing_size = None
    if isinstance(existing_vectors, dict):
        existing_size = existing_vectors.get("size")
    else:
        existing_size = getattr(existing_vectors, "size", None)

    if existing_size is not None and existing_size != vector_size:
        print(f"⚠️  Collection '{collection_name}' exists with vector size {existing_size}, "
              f"but embedding model returns size {vector_size}.")
        print(f"🗑️  Deleting old collection and recreating with correct size...")
        
        # Delete the old collection
        client.delete_collection(collection_name)
        
        # CRITICAL: Reset the client to clear cached collection info
        reset_qdrant_client()
        client = get_qdrant_client()
        
        # Create new collection with correct size
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Recreated collection '{collection_name}' with vector size {vector_size}")
        return client
    else:
        print(f"✅ Collection '{collection_name}' already exists with correct vector size {vector_size}")
        return client


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
    
    vector_size = embeddings.vector_size()
    client = ensure_qdrant_collection(collection_name, vector_size)  # Get fresh client

    vector_store = QdrantVectorStore(
        client=client,  # Use the client returned from ensure_qdrant_collection
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Store in a local Qdrant collection (persists to disk under QDRANT_DB_PATH)
    vector_store.add_documents(chunked_docs)

    print(f"Stored {len(chunked_docs)} chunks in local Qdrant collection '{collection_name}'.")


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
