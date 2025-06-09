import streamlit as st
from dotenv import load_dotenv
import re

from the_batch_scraper import TheBatchScraper
from clip_embedder import CLIPEmbedder
from llm_generator import LLMGenerator
from milvus_vector_store import MilvusVectorStore
from multimodal_processor import MultimodalProcessor
from rag_logger import log_rag_sample

load_dotenv()

scraper = TheBatchScraper()
embedder = CLIPEmbedder()
processor = MultimodalProcessor(scraper, embedder)
milvus_store = MilvusVectorStore(dim=512, embedder=embedder)
llm = LLMGenerator()


st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG")

col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Reset storage"):
        milvus_store.reset_storage()
        st.session_state.pop("full_sections", None)
        st.success("Storage has been cleared.")

with col2:
    if st.button("Ingest articles"):
        entries, full_sections = processor.process(limit=5)
        milvus_store.insert_entries(entries)
        milvus_store.create_index()
        st.session_state["full_sections"] = full_sections
        st.success(f"Added {len(entries)} new entries.")

query = st.text_input("Ask your question:", placeholder="e.g. What is new with DeepSeek?")
if st.button("Run search") and query.strip():
    query_vec = embedder.embed(query)
    results = milvus_store.search_with_rerank(query_vec, top_k=10, rerank_k=3, image_k=3)

    full_sections = st.session_state.get("full_sections", {})
    text_articles = {}

    for hit in results:
        typ = hit.entity.get("type")
        page = hit.entity.get("page_name")
        title = hit.entity.get("title")
        base_title = re.sub(r"\s*\[chunk \d+\]$", "", title)
        key = (page, base_title)

        if typ == "text":
            text_articles[key] = full_sections.get(key, "[Not found]")

    selected_article = None
    if text_articles:
        selected_article = next(iter(text_articles.items()))

    image_urls = []
    if selected_article:
        selected_page, selected_title = selected_article[0]

        related_images = milvus_store.collection.query(
            expr=f'page_name == "{selected_page}" and title == "{selected_title}" and type == "image"',
            output_fields=["content"]
        )
        image_urls = [img["content"] for img in related_images]

    context_parts = []
    if selected_article:
        context_parts.append(f"TEXT:\n{selected_article[1]}")

    context = "\n\n".join(context_parts)

    with st.spinner("Generating answer with LLM..."):
        answer = llm.generate(context=context.strip(), question=query)

    log_rag_sample(
        question=query,
        answer=answer.strip(),
        contexts=[context.strip()],
        ground_truth=None
    )

    st.subheader("LLM answer:")
    st.markdown(f"**{answer.strip()}**")

    st.subheader("Article:")

    col1, col2 = st.columns([2, 1])
    with col1:
        if selected_article:
            st.markdown(f"**{selected_article[0][1]}**")
            st.markdown(selected_article[1])
        else:
            st.info("No text article found.")

    with col2:
        if image_urls:
            st.image(image_urls[0], use_container_width=True)
        else:
            st.info("No images found.")
