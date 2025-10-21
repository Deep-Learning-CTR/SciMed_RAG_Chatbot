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
        if not doi:
            print(f"Skipping paper {idx}: DOI not found.")
            continue

        # Construct a filename safe from special characters
        title = paper.get('title', f'paper_{idx}')
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        filepath = os.path.join(output_dir, f"{doi}.pdf")

        try:
            save_pdf({"doi": doi}, filepath=filepath)
            print(f"Downloaded: {filepath}")
        except Exception as e:
            print(f"Failed to download {title}: {e}")
