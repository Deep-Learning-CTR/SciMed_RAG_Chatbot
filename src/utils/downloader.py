from paperscraper.pdf import save_pdf

def download_papers(results, output_dir='processing/downloaded_papers'):
    """
    Downloads papers from a list of results using DOI.
    
    Args:
        results (list): List of dictionaries containing paper info, each must have 'doi'.
        output_dir (str): Directory to save the downloaded PDFs.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, paper in enumerate(results, start=1):
        doi = paper.get('doi')
        pdf_url = paper.get('pdf_url')

        if not doi and not pdf_url:
            print(f"Skipping paper {idx}: no DOI or PDF URL available.")
            continue

        identifier = doi or pdf_url
        safe_identifier = "".join(c if c.isalnum() or c in "._- " else "_" for c in identifier)

        # Construct a filename safe from special characters
        title = paper.get('title', f'paper_{idx}')
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)

        filename = f"{safe_identifier}.pdf"
        filepath = os.path.join(output_dir, filename)

        try:
            if os.path.exists(filepath):
                print(f"Skipping download, already exists: {filepath}")
                continue

            download_kwargs = {}
            if doi:
                download_kwargs["doi"] = doi
            if pdf_url:
                download_kwargs["pdf_url"] = pdf_url

            save_pdf(download_kwargs, filepath=filepath)
            print(f"Downloaded: {filepath}")
        except Exception as e:
            print(f"Failed to download {title}: {e}")
