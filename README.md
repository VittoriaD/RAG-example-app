# Multimodal RAG with CLIP and Milvus

This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** pipeline that scrapes articles from [The Batch](https://www.deeplearning.ai/the-batch/), embeds both text and images using CLIP, stores them in **Milvus**, and answers user queries using an **LLM** with both text and image context.

---

## Project structure

| File | Description |
|------|-------------|
| `main.py` | Streamlit UI entry point. Enables scraping, embedding, storing, and querying. |
| `the_batch_scraper.py` | Scrapes articles, paragraphs, and image links from The Batch website. |
| `clip_embedder.py` | Embeds text and images using OpenAI's CLIP model. |
| `multimodal_processor.py` | Preprocesses article text and images into embedding entries. |
| `milvus_vector_store.py` | Connects to Milvus, handles vector storage, indexing, and similarity search. |
| `llm_generator.py` | Generates answers to user queries using OpenAI's LLMs (e.g. GPT-4). |
| `image_processor.py` | Helper class for embedding and storing images separately. |
| `rag_logger.py` | Logs RAG queries and answers into a local JSONL file. |
| `evaluate_ragas.py` | Evaluates logged RAG sessions using RAGAS metrics (faithfulness, relevancy). |
| `requirements.txt` | Python dependencies for the project. |

---

## How to run

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and update the values inside to use your own tokens:
```bash
cp example_env .env
```

### 3. Launch the app
```bash
streamlit run main.py
```

### 4. To run evaluation your RAG Sessions
Note: you may run it only after some time using the app as it will evaluate logged queries only.
```bash
python evaluate_ragas.py
```

### Main key points
1. This is Multimodal RAG, meaning that images will be also returned to user query if they associate with the request.
2. In order to support Multimodal approach CLIP-based embeddings are used.
3. When context is retrieved, GPT 4 takes care of the answer. 
4. RAGAS-based evaluation checks answer faithfulness and relevancy as it uses only the logged queries.

### UI navigation
1. Before using RAG, you need to scrape The Batch source and wait till data will be stored in the vector storage.
2. If you want to rerun everything, you can clear your vector storage from the UI.
3. After the data is collected and stored, you see the message about successful storage.
4. Run the query and enjoy.