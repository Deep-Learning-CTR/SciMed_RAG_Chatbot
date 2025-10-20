from utils.searcher import search_academic_papers
from utils.downloader import download_papers

papers, summary_df = search_academic_papers("machine learning COVID-19", database="medRxiv", n_results=5)

# Print first paper's title
if papers:
    print(papers[0]['title'])

download_papers(papers)

# Display summary
print(summary_df)
