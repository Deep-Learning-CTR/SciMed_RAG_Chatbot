# file: rag_embedding_pipeline.py

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.extractors import extract_text_from_multiple_files, split_chunk_overlap  # your earlier code

# ---------------- CONFIG ----------------
DATA_DIR = "processing/downloaded_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "papers_rag"

# Local Qdrant DB path (change if you want persistent storage)
QDRANT_DB_PATH = "processing/qdrant_local_db"  # folder will be created
# ---------------------------------------

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


def embed_and_store(documents, collection_name=COLLECTION_NAME, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Create embeddings using nomic-embed-text-v2-moe and store in Qdrant (serverless)
    """
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe")
    
    # Split into chunks
    chunked_docs = split_chunk_overlap(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Store in Qdrant (serverless / local)
    Qdrant.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name=collection_name,
        prefer_grpc=False,
        location=QDRANT_DB_PATH  # path on disk for local DB
    )
    
    print(f"✅ Stored {len(chunked_docs)} chunks in local Qdrant collection '{collection_name}'.")


def extract_and_embed(metadata_list, data_dir=DATA_DIR, collection_name=COLLECTION_NAME):
    """Full pipeline: extract text, attach metadata, embed, store locally"""
    file_paths = get_all_files(data_dir)
    all_docs = extract_text_from_multiple_files(file_paths)
    all_docs = add_metadata_to_documents(all_docs, metadata_list)
    embed_and_store(all_docs, collection_name=collection_name)
    return all_docs
