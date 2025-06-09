class MultimodalProcessor:
    def __init__(self, scraper, embedder, max_chars=700):
        self.scraper = scraper
        self.embedder = embedder
        self.max_chars = max_chars

    def chunk_paragraphs(self, paragraphs):
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if not para.strip():
                continue
            if len(current_chunk) + len(para) + 2 > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process(self, limit=3):
        entries = []
        full_sections = {}
        data = self.scraper.scrape_latest(limit=limit)

        for article in data:
            page_name = article["page_name"]
            for item in article["content"]:
                title = item["title"]
                paras = item["paragraphs"]

                if paras:
                    full_text = "\n\n".join(paras)
                    full_sections[(page_name, title)] = full_text

                    chunks = self.chunk_paragraphs(paras)
                    for i, chunk in enumerate(chunks):
                        emb = self.embedder.embed(chunk)
                        entries.append((page_name, title, "text", chunk, emb))

                for img_url in item["images"]:
                    emb = self.embedder.embed_image_from_url(img_url)
                    if emb:
                        entries.append((page_name, title, "image", img_url, emb))

        return entries, full_sections
