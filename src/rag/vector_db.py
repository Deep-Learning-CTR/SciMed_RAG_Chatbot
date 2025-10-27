# file: vector_db.py

from typing import List, Optional, Tuple
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

_COLLECTION_NAME = "papers_rag"
QDRANT_DB_PATH = "processing/qdrant_local_db"


def _new_qdrant_client() -> QdrantClient:
    """Always create a new local Qdrant client (non-shared)."""
    return QdrantClient(path=QDRANT_DB_PATH)


def set_collection_name(name: str) -> None:
    global _COLLECTION_NAME
    _COLLECTION_NAME = name


def get_collection_name() -> str:
    return _COLLECTION_NAME


def ensure_qdrant_collection(collection_name: str, vector_size: int) -> None:
    """Create or recreate a Qdrant collection."""
    client = _new_qdrant_client()

    try:
        try:
            client.get_collection(collection_name)
            collection_exists = True
        except Exception:
            collection_exists = False

        if collection_exists:
            print(f"🗑️  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created fresh collection '{collection_name}' (vector size={vector_size})")
    finally:
        client.close()


def store_documents_in_qdrant(
    documents: List,
    embeddings: Embeddings,
    collection_name: Optional[str] = None,
) -> None:
    """Store documents in Qdrant and close connection after."""
    if not documents:
        print("No document chunks provided; skipping Qdrant ingestion.")
        return

    collection_name = collection_name or get_collection_name()

    # Get vector size
    if hasattr(embeddings, "vector_size"):
        vector_size = embeddings.vector_size()
    else:
        vector_size = len(embeddings.embed_query("dimension probe"))

    ensure_qdrant_collection(collection_name, vector_size)

    client = _new_qdrant_client()
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        vector_store.add_documents(documents)
        print(f"Stored {len(documents)} chunks in local Qdrant collection '{collection_name}'.")
    finally:
        client.close()


def get_vector_store(
    embeddings: Embeddings,
    collection_name: Optional[str] = None,
) -> Tuple[QdrantVectorStore, QdrantClient]:
    """
    Get a QdrantVectorStore for querying, along with its open client.
    Caller must close the client after use.
    
    Returns:
        (vector_store, client)
    """
    collection_name = collection_name or get_collection_name()
    client = _new_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store, client
