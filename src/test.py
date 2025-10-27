from utils.searcher import search_academic_papers
from utils.downloader import download_papers

from rag.embeddings import extract_and_embed

print("=" * 80)
print("STEP 1: Searching for papers...")
papers, summary_df = search_academic_papers("machine learning COVID-19", database="arXiv", n_results=5)

# Print first paper's title
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
    extract_and_embed(papers, "processing/downloaded_papers", "paper_embeddings")
    print("[OK] Embedding complete!")
except Exception as e:
    print(f"[ERROR] during embedding: {e}")
    import traceback
    traceback.print_exc()

# Display summary
print("\n" + "=" * 80)
print("SUMMARY:")
print(summary_df)
