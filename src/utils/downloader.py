from paperscraper.pdf import save_pdf

def download_papers(results, output_dir='processing/downloaded_papers',conversation_id=None):
    """
    Downloads papers from a list of results using DOI.
    
    Args:
        results (list): List of dictionaries containing paper info, each must have 'doi'.
        output_dir (str): Directory to save the downloaded PDFs.
        conversation_id (str): Optional conversation ID to track downloads.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup conversation tracking file if conversation_id is provided
    tracking_file = None
    if conversation_id:
        processing_dir = 'processing'
        if not os.path.exists(processing_dir):
            os.makedirs(processing_dir)
        tracking_file = os.path.join(processing_dir, f"{conversation_id}")

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
                # Still track it if conversation_id is provided
                if tracking_file:
                    source_link = pdf_url or doi
                    with open(tracking_file, 'a', encoding='utf-8') as f:
                        f.write(f"{source_link}:{filename}\n")
                continue

            download_kwargs = {}
            if doi:
                download_kwargs["doi"] = doi
            if pdf_url:
                download_kwargs["pdf_url"] = pdf_url

            save_pdf(download_kwargs, filepath=filepath)
            print(f"Downloaded: {filepath}")
            
            # Track the download in conversation file
            if tracking_file:
                source_link = pdf_url or doi
                with open(tracking_file, 'a', encoding='utf-8') as f:
                    f.write(f"{source_link}:{filename}\n")
                    
        except Exception as e:
            print(f"Failed to download {title}: {e}")
