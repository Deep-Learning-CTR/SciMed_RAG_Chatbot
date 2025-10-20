import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import json

import tempfile
from pathlib import Path
import shutil
import pandas as pd

from paperscraper.arxiv import get_and_dump_arxiv_papers

def search_medrxiv(query: str, n: int = 10, output_path: str = "medrxiv_results.jsonl"):
    """
    Scrape medRxiv search results page for a query and save top n results to a JSONL file.

    Args:
        query: Search keywords (e.g., "ai and medical imagery")
        n: Number of top results to return
        output_path: File path to save JSONL results
    """
    q = quote_plus(query)
    url = f"https://www.medrxiv.org/search/{q}%20numresults%3A{n}%20sort%3Arelevance-rank"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; MedRxivScraper/1.1)"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for article in soup.select("div.highwire-cite")[:n]:
        title_el = article.select_one(".highwire-cite-title")
        authors_el = article.select_one(".highwire-citation-authors")
        date_el = article.select_one(".highwire-cite-metadata-pages")
        doi_el = article.select_one(".highwire-cite-metadata-doi")
        journal_el = article.select_one(".highwire-cite-metadata-journal")

        if not title_el:
            continue
        
        title_text = title_el.get_text(strip=True)

        authors = authors_el.get_text(" ", strip=True) if authors_el else ""
        date = date_el.get_text(strip=True) if date_el else ""
        doi = doi_el.get_text(strip=True).strip('doi:') if doi_el else ""
        journal = journal_el.get_text(strip=True) if journal_el else ""

        result = {
            "title": title_text,
            "authors": authors,
            "date": date,
            "doi": doi,
            "journal": journal,
        }
        results.append(result)

    # Write to JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(results)} results to {output_path}")
    return output_path



def search_academic_papers(query: str, database: str = "arXiv", n_results: int = 10):
    """
    Search for academic papers on arXiv or medRxiv.

    Args:
        query (str): Search query keywords.
        database (str): "arXiv" or "medRxiv".
        n_results (int): Maximum number of papers to retrieve.

    Returns:
        list[dict]: List of paper metadata dictionaries.
        pd.DataFrame: Summary dataframe of the results.
    """
    if not query:
        raise ValueError("Query cannot be empty.")

    temp_dir = tempfile.mkdtemp()
    output_file = Path(temp_dir) / "results.jsonl"

    try:
        keywords = [query] if ' ' not in query else query.split()

        if database == "arXiv":
            get_and_dump_arxiv_papers(keywords, output_filepath=str(output_file), max_results=n_results)
        elif database == "medRxiv":
            search_medrxiv(query, n=n_results, output_path=str(output_file))
        else:
            raise ValueError("Database must be 'arXiv' or 'medRxiv'.")

        # Load results
        papers = []
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    papers.append(json.loads(line))

        if not papers:
            return [], pd.DataFrame()

        # Prepare summary dataframe
        summary_data = []
        for paper in papers:
            authors = paper.get('authors', ['Unknown'])
            if isinstance(authors, list):
                authors_summary = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
            else:
                authors_summary = authors[:100] + '...' if len(str(authors)) > 100 else str(authors)

            summary_data.append({
                'Title': paper.get('title', 'No title'),
                'Authors': authors_summary,
                'Date': paper.get('date', 'Unknown'),
                'PDF': paper.get('pdf_url', '')
            })

        df_summary = pd.DataFrame(summary_data)
        return papers, df_summary

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

