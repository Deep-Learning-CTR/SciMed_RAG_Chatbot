from utils.searcher import search_academic_papers
from utils.downloader import download_papers
from utils.conversations import create_new_conversation
from rag.vector_db import set_collection_name
from rag.embeddings import extract_and_embed_conversation
from cerebras.cloud.sdk import Cerebras
import os

llm = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)
# Docs
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Why is fast inference important?",
#         }
# ],
#     model="llama-4-scout-17b-16e-instruct",
# )
# Prepare a new conversation and set a dedicated vector collection
conversation_id, conv_path = create_new_conversation(base_dir="conversations")
collection_name = f"conv_{conversation_id}"
set_collection_name(collection_name)

options=['arXiv','medRxiv']
selected_option = options[0]  # Example: selecting 'arXiv'
k=10
user_query="machine learning COVID-19"

# Correct argument order: query, database, n_results
papers, summary_df = search_academic_papers(user_query, database=selected_option, n_results=k)
if papers:
    print(f"\n[OK] Found {len(papers)} papers")
    print(f"First paper: {papers[0]['title']}")

print("\n" + "=" * 80)
print("STEP 2: Downloading papers...")
download_papers(papers, conversation_id=conversation_id)
print("[OK] Download complete")

print("\n" + "=" * 80)
print("STEP 3: Extracting and embedding...")
try:
    extract_and_embed_conversation(
        papers,
        data_dir="processing/downloaded_papers",
        collection_name=collection_name,
        conversation_id=conversation_id,
    )
    print("[OK] Embedding complete!")
    print(f"Conversation ID: {conversation_id}")
    print(f"Collection Name: {collection_name}")
    print(f"Conversation folder: {conv_path}")
except Exception as e:
    print(f"[ERROR] during embedding: {e}")
    import traceback
    traceback.print_exc()
