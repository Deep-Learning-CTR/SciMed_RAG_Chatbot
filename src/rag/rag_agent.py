import sys
from pathlib import Path

# ensure the repository 'src' folder is on sys.path so `import utils` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from cerebras.cloud.sdk import Cerebras

from embeddings import OllamaEmbeddings, extract_and_embed_conversation
from vector_db import get_vector_store, set_collection_name
from utils.conversations import create_new_conversation
from utils.downloader import download_papers
from utils.searcher import search_academic_papers

st.set_page_config(page_title="SciMed RAG Agent", page_icon="🧬", layout="wide")

AVAILABLE_MODELS: List[Tuple[str, str]] = [
    ("llama-4-scout-17b-16e-instruct", "LLaMA 4 Scout 17B Instruct"),
    ("llama-3.1-70b-instruct", "LLaMA 3.1 70B Instruct"),
    ("llama-3.1-8b-instruct", "LLaMA 3.1 8B Instruct"),
    ("mistral-large-2407", "Mistral Large 24.07"),
]

DOMAIN_OPTIONS: Dict[str, str] = {
    "Scientific (arXiv)": "arXiv",
    "Medical (medRxiv)": "medRxiv",
}

DEFAULT_MAX_RESULTS = 10
DEFAULT_RETRIEVAL_K = 4
SYSTEM_PROMPT = (
    "You are a research assistant specialising in scientific and medical literature. "
    "Use the supplied sources to answer the user's question clearly and concisely. "
    "Cite supporting evidence inline using clickable links in the format "
    "[Source 1](https://doi.org/DOI_HERE) — replacing DOI_HERE with the actual DOI from the metadata. "
    "For arXiv or medRxiv papers, use their corresponding DOI URLs (e.g., https://doi.org/10.48550/arXiv.2309.00252 for arXiv or https://doi.org/10.1101/2025.08.12.25333155 for medRxiv). "
    "If the sources do not contain sufficient information to answer the question, say so frankly. "
    "At the end of the response, include a 'References' section listing all sources in this format: "
    "[Source 1](https://doi.org/DOI_HERE) — filename.pdf | page X | DOI DOI_HERE. "
    "Ensure that every [Source #] in the text is a clickable DOI link."
)


@st.cache_resource(show_spinner=False)
def get_cerebras_client(api_key: str) -> Cerebras:
    """Create a cached Cerebras client."""
    return Cerebras(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_query_embeddings() -> OllamaEmbeddings:
    """Create a cached embedding model for retrieval time."""
    return OllamaEmbeddings()


def init_session_state() -> None:
    """Initialise conversation and UI state for the Streamlit session."""
    if "conversation_id" not in st.session_state:
        conv_id, conv_path = create_new_conversation()
        collection_name = f"conv_{conv_id}"
        set_collection_name(collection_name)

        st.session_state.conversation_id = conv_id
        st.session_state.conversation_path = conv_path
        st.session_state.collection_name = collection_name
        st.session_state.messages: List[Dict[str, Any]] = []
        st.session_state.papers_metadata: List[Dict[str, Any]] = []
        st.session_state.search_summary: pd.DataFrame | None = None

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = DEFAULT_RETRIEVAL_K


def reset_conversation() -> None:
    """Start a fresh conversation with a new vector collection."""
    conv_id, conv_path = create_new_conversation()
    collection_name = f"conv_{conv_id}"
    set_collection_name(collection_name)

    st.session_state.conversation_id = conv_id
    st.session_state.conversation_path = conv_path
    st.session_state.collection_name = collection_name
    st.session_state.messages = []
    st.session_state.papers_metadata = []
    st.session_state.search_summary = None


def render_sidebar() -> Dict[str, Any]:
    """Render controls in the sidebar and return the selected options."""
    with st.sidebar:
        st.header("Configuration")

        domain_label = st.selectbox(
            "Target collection",
            list(DOMAIN_OPTIONS.keys()),
            index=0,
            key="domain_label",
        )

        model_labels = [label for _, label in AVAILABLE_MODELS]
        selected_label = st.selectbox(
            "Cerebras model",
            model_labels,
            index=0,
            key="model_label",
        )
        label_to_model = {label: model_id for model_id, label in AVAILABLE_MODELS}
        model_name = label_to_model[selected_label]

        max_results = st.slider(
            "Max papers per search",
            min_value=1,
            max_value=25,
            value=DEFAULT_MAX_RESULTS,
            step=1,
            key="max_results",
        )

        retrieval_k = st.slider(
            "Chunks to retrieve per question",
            min_value=1,
            max_value=10,
            value=st.session_state.retrieval_k,
            step=1,
            key="retrieval_k",
        )

        if st.button("Start new conversation", use_container_width=True):
            reset_conversation()
            st.experimental_rerun()

    return {
        "database": DOMAIN_OPTIONS[domain_label],
        "model": model_name,
        "max_results": max_results,
        "retrieval_k": retrieval_k,
    }


def format_source_label(metadata: Dict[str, Any]) -> str:
    """Create a compact label describing the source of a document chunk."""
    parts: List[str] = []
    filename = metadata.get("filename")
    if filename:
        parts.append(str(filename))

    page = metadata.get("page")
    if isinstance(page, int):
        parts.append(f"page {page}")

    doi = metadata.get("doi")
    if doi:
        parts.append(f"DOI {doi}")

    pdf_link = metadata.get("pdf_link") or metadata.get("pdf_url")
    if pdf_link and len(parts) < 3:
        parts.append(str(pdf_link))

    return " | ".join(parts) if parts else "Unknown source"


def prepare_context(chunks: List[Tuple[Any, float]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Turn similarity search results into prompt context and UI-friendly metadata.

    Returns:
        context_str: Text block to feed the LLM.
        chunk_records: Structured metadata for UI display.
    """
    context_segments: List[str] = []
    chunk_records: List[Dict[str, Any]] = []

    for idx, (doc, score) in enumerate(chunks, start=1):
        label = format_source_label(doc.metadata)
        distance = float(score) if score is not None else float("nan")
        context_segments.append(
            f"[Source {idx}] (distance={distance:.3f}; {label})\n{doc.page_content}"
        )

        snippet = doc.page_content.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rstrip() + "..."

        chunk_records.append(
            {
                "id": idx,
                "label": label,
                "distance": distance,
                "content": snippet,
                "metadata": doc.metadata,
            }
        )

    context_str = "\n\n".join(context_segments)
    return context_str, chunk_records


def run_ingestion(query: str, database: str, max_results: int, collection_name: str, conversation_id: str) -> None:
    """Execute the search → download → embed pipeline."""
    with st.spinner("Searching academic databases..."):
        papers, summary_df = search_academic_papers(query, database=database, n_results=max_results)

    if not papers:
        st.warning("No papers found. Try refining your query or increasing the result limit.")
        return

    st.session_state.papers_metadata = papers
    st.session_state.search_summary = summary_df

    st.success(f"Found {len(papers)} papers. Downloading PDFs...")
    with st.spinner("Downloading papers..."):
        download_papers(papers, conversation_id=conversation_id)

    st.success("Download finished. Extracting and embedding content...")
    with st.spinner("Extracting chunks and updating the vector database..."):
        extract_and_embed_conversation(
            papers,
            data_dir="processing/downloaded_papers",
            collection_name=collection_name,
            conversation_id=conversation_id,
        )

    st.success("Knowledge base updated! Toggle off 'Search online' to start chatting with the new papers.")


def run_chat_flow(model_name: str, retrieval_k: int) -> None:
    """Handle the conversational experience with retrieval augmented responses."""
    if not st.session_state.papers_metadata:
        st.info("Load papers first by enabling 'Search online'.")
        return

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("chunks"):
                # Show sources summary
                source_lines = ", ".join(f"[Source {chunk['id']}]" for chunk in message["chunks"])
                st.caption(f"📚 Sources used: {source_lines}")
                
                # Show chunks in an expander
                with st.expander(f"🔍 View {len(message['chunks'])} relevant chunks", expanded=False):
                    for chunk in message["chunks"]:
                        st.markdown(
                            f"**Source {chunk['id']}** — {chunk['label']} (distance={chunk['distance']:.3f})"
                        )
                        st.markdown(f"> {chunk['content']}")
                        st.divider()

    user_prompt = st.chat_input("Ask a question about the ingested papers...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    embeddings = get_query_embeddings()
    try:
        vector_store, client = get_vector_store(
            embeddings, collection_name=st.session_state.collection_name
        )
        try:
            with st.spinner("Retrieving relevant chunks..."):
                raw_chunks = vector_store.similarity_search_with_score(user_prompt, k=retrieval_k)
        finally:
            client.close()  # ✅ always close after use
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        return

    if not raw_chunks:
        assistant_reply = ("I could not find relevant context in the current knowledge base. "
                           "Consider ingesting more papers or adjusting your question.")
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply, "chunks": []})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        return

    context_block, chunk_records = prepare_context(raw_chunks)
    source_overview = "\n".join(
        f"[Source {chunk['id']}] {chunk['label']} (distance={chunk['distance']:.3f})"
        for chunk in chunk_records
    )

    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        st.error("CEREBRAS_API_KEY is not set. Please configure it in your environment.")
        return

    client = get_cerebras_client(api_key)
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
        if msg["role"] in {"user", "assistant"}
    ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history_messages]
    messages.append(
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                f"Relevant sources:\n{source_overview}\n\n"
                f"Context:\n{context_block}\n\n"
                "Answer the question using the context and cite sources as [Source #]."
            ),
        }
    )

    try:
        with st.spinner("Calling Cerebras model..."):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

        assistant_text = response.choices[0].message.content  # type: ignore[attr-defined]
    except Exception as exc:
        st.error(f"Cerebras API call failed: {exc}")
        return

    assistant_record = {"role": "assistant", "content": assistant_text, "chunks": chunk_records}
    st.session_state.messages.append(assistant_record)

    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        # Show sources summary
        source_lines = ", ".join(f"[Source {chunk['id']}]" for chunk in chunk_records)
        st.caption(f"📚 Sources used: {source_lines}")
        
        # Show chunks in an expander
        with st.expander(f"🔍 View {len(chunk_records)} relevant chunks", expanded=False):
            for chunk in chunk_records:
                st.markdown(
                    f"**Source {chunk['id']}** — {chunk['label']} (distance={chunk['distance']:.3f})"
                )
                st.markdown(f"> {chunk['content']}")
                st.divider()


def render_search_results() -> None:
    """Display a summary of the most recent search results if available."""
    if st.session_state.search_summary is not None:
        st.subheader("Latest search results")
        st.dataframe(st.session_state.search_summary)


def main() -> None:
    init_session_state()
    options = render_sidebar()

    st.title("SciMed Research Chatbot")
    st.markdown(
        "Ingest recent scientific or medical papers, then ask targeted questions. "
        "Toggle **Search online** to control whether you're adding new material or chatting over the existing knowledge base."
    )

    search_mode = st.toggle("Search online", key="search_mode")

    if search_mode:
        st.session_state.search_query = st.text_input(
            "Paper search query",
            value=st.session_state.search_query,
            placeholder="e.g. machine learning for COVID-19 diagnosis",
        )

        if st.button("Run search and update knowledge base", type="primary", disabled=not st.session_state.search_query):
            run_ingestion(
                query=st.session_state.search_query,
                database=options["database"],
                max_results=options["max_results"],
                collection_name=st.session_state.collection_name,
                conversation_id=st.session_state.conversation_id,
            )

        render_search_results()
    else:
        render_search_results()
        run_chat_flow(model_name=options["model"], retrieval_k=options["retrieval_k"])


if __name__ == "__main__":
    main()
