"""CLI helper to inspect stored Qdrant chunks and their metadata."""

import argparse
import textwrap

from langchain_qdrant import QdrantVectorStore

from rag.embeddings import (
    COLLECTION_NAME,
    OllamaEmbeddings,
    QDRANT_DB_PATH,
    get_qdrant_client,
    reset_qdrant_client,
)


def format_chunk(doc, score):
    preview = doc.page_content.strip().replace("\r\n", "\n")
    preview = "\n".join(textwrap.wrap(preview, width=100))[:400]
    metadata_lines = [f"  {key}: {value}" for key, value in sorted(doc.metadata.items())]
    metadata_text = "\n".join(metadata_lines) if metadata_lines else "  (no metadata)"
    return (
        f"Score: {score:.4f}\n"
        f"Preview:\n{textwrap.indent(preview, prefix='  ')}\n"
        f"Metadata:\n{metadata_text}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        help="Natural language query to embed and search for similar chunks.",
    )
    parser.add_argument(
        "--collection",
        default="paper_embeddings",
        help=f"Qdrant collection name (default: {COLLECTION_NAME!r}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override embedding concurrency when issuing the query.",
    )
    args = parser.parse_args()

    embedder = OllamaEmbeddings(max_workers=args.max_workers)
    client = get_qdrant_client()
    try:
        if not client.collection_exists(collection_name=args.collection):
            print(
                f"Collection '{args.collection}' not found in local Qdrant store at '{QDRANT_DB_PATH}'. "
                "Run the embedding pipeline before issuing queries."
            )
            return

        store = QdrantVectorStore(
            client=client,
            collection_name=args.collection,
            embedding=embedder,
        )

        results = store.similarity_search_with_score(args.query, k=args.k)
        if not results:
            print("No matches returned for this query.")
            return

        for idx, (doc, score) in enumerate(results, start=1):
            print(f"\n=== Result {idx} ===")
            print(format_chunk(doc, score))
    finally:
        reset_qdrant_client()


if __name__ == "__main__":
    main()
