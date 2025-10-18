import asyncio
from crawl4ai import AsyncWebCrawler, DefaultMarkdownGenerator, PruningContentFilter
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    config = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.6),
        options={"ignore_links": True}
    )
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://example.com",
            config=config
        )
        
        if not result.success:
            print(f"Crawl failed: {result.error_message}")
            print(f"Status code: {result.status_code}")
        # # Different content formats
        # print(result.html)         # Raw HTML
        # print(result.cleaned_html) # Cleaned HTML
        # print(result.markdown.raw_markdown) # Raw markdown from cleaned html
        # print(result.markdown.fit_markdown) # Most relevant content in markdown

        # # Check success status
        # print(result.success)      # True if crawl succeeded
        # print(result.status_code)  # HTTP status code (e.g., 200, 404)

        # # Access extracted media and links
        # print(result.media)        # Dictionary of found media (images, videos, audio)
        # print(result.links)        # Dictionary of internal and external links


if __name__ == "__main__":
    asyncio.run(main())
