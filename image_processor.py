from clip_embedder import CLIPEmbedder


class ImageProcessor:
    def __init__(self, scraper, image_embedder: CLIPEmbedder):
        self.scraper = scraper
        self.image_embedder = image_embedder

    def process(self, limit=1):
        articles = self.scraper.scrape_latest(limit=limit)
        entries = []

        for article in articles:
            page = article["page_name"]
            for block in article["content"]:
                title = block.get("title", "")
                image_urls = block.get("images", [])

                for url in image_urls:
                    vector = self.image_embedder.embed_image_from_url(url)
                    if vector:
                        entries.append((
                            page,         # page_name
                            title,        # title
                            "image",      # type
                            url,          # content (image URL)
                            vector        # embedding
                        ))

        return entries
