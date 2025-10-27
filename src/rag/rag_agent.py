from utils.searcher import search_academic_papers
from utils.downloader import download_papers
from embeddings import extract_and_embed_conversation
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
options=['arXiv','medRxiv']
selected_option = options[0]  # Example: selecting 'arXiv'
k=10
user_query="machine learning COVID-19"
papers, summary_df = search_academic_papers(selected_option, k)
if papers:
    print(f"\n[OK] Found {len(papers)} papers")
    print(f"First paper: {papers[0]['title']}")

print("\n" + "=" * 80)
print("STEP 2: Downloading papers...")
download_papers(papers)
print("[OK] Download complete")

print("\n" + "=" * 80)
print("STEP 3: Extracting and embedding...")
try:
    extract_and_embed_conversation(papers, "processing/downloaded_papers")
    print("[OK] Embedding complete!")
except Exception as e:
    print(f"[ERROR] during embedding: {e}")
    import traceback
    traceback.print_exc()
