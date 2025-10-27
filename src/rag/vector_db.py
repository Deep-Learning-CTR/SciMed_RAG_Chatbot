# file: vector_db.py

from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ---------------- CONFIG ----------------
# Default collection name; can be overridden at runtime via set_collection_name().
_COLLECTION_NAME = "papers_rag"
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


def set_collection_name(name: str) -> None:
    """Set the default Qdrant collection name to use for this run."""
    global _COLLECTION_NAME
    _COLLECTION_NAME = name


def get_collection_name() -> str:
    """Get the currently configured default collection name."""
    return _COLLECTION_NAME


def __getattr__(name: str):
    """
    Backward compatibility for code importing COLLECTION_NAME as a constant.
    Returns the current default collection name dynamically.
    """
    if name == "COLLECTION_NAME":
        return get_collection_name()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_qdrant_collection(collection_name: str, vector_size: int) -> QdrantClient:
    """
    Always recreates the collection with the given name, deleting any existing one.
    This ensures a fresh collection for each embedding run.
    Returns the QdrantClient (fresh instance after recreation).
    """
    client = get_qdrant_client()
    
    # Check if collection exists
    try:
        client.get_collection(collection_name)
        collection_exists = True
    except Exception:
        collection_exists = False
    
    # If collection exists, delete it
    if collection_exists:
        print(f"🗑️  Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
        
        # CRITICAL: Reset the client to clear cached collection info
        reset_qdrant_client()
        client = get_qdrant_client()
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✅ Created fresh collection '{collection_name}' with vector size {vector_size}")
    
    return client


def store_documents_in_qdrant(
    documents: List,
    embeddings: Embeddings,
    collection_name: Optional[str] = None,
) -> None:
    """
    Store documents in a local Qdrant collection.
    
    Args:
        documents: List of document chunks to store
        embeddings: Embeddings instance to use for vectorization
        collection_name: Name of the Qdrant collection
    """
    if not documents:
        print("No document chunks provided; skipping Qdrant ingestion.")
        return
    
    # Resolve collection name
    if not collection_name:
        collection_name = get_collection_name()

    # Get vector size from embeddings
    vector_size = embeddings.vector_size() if hasattr(embeddings, 'vector_size') else None
    if vector_size is None:
        # Fallback: compute a sample embedding to determine size
        sample_vector = embeddings.embed_query("dimension probe")
        vector_size = len(sample_vector)
    
    # Ensure collection exists with correct vector size
    client = ensure_qdrant_collection(collection_name, vector_size)

    # Create vector store and add documents
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Store in a local Qdrant collection (persists to disk under QDRANT_DB_PATH)
    vector_store.add_documents(documents)

    print(f"Stored {len(documents)} chunks in local Qdrant collection '{collection_name}'.")


def get_vector_store(
    embeddings: Embeddings,
    collection_name: Optional[str] = None,
) -> QdrantVectorStore:
    """
    Get a QdrantVectorStore instance for querying.
    
    Args:
        embeddings: Embeddings instance to use for vectorization
        collection_name: Name of the Qdrant collection
        
    Returns:
        QdrantVectorStore instance
    """
    client = get_qdrant_client()
    
    # Resolve collection name
    if not collection_name:
        collection_name = get_collection_name()

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
