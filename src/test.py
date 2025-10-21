from utils.searcher import search_academic_papers
from utils.downloader import download_papers

from rag.embeddings import extract_and_embed

papers, summary_df = search_academic_papers("machine learning COVID-19", database="arXiv", n_results=5)

# Print first paper's title
if papers:
    print(papers[0]['title'])
    print(papers)

download_papers(papers)

extract_and_embed(papers, "processing/downloaded_papers", "paper_embeddings")

# Display summary
print(summary_df)
