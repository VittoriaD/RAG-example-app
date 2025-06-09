from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class MilvusVectorStore:
    MIN_SIMILARITY_THRESHOLD = 0.8

    def __init__(self, name="multimodal_rag", dim=512, embedder=None):
        self.name = name
        self.dim = dim
        self.embedder = embedder
        self._connect()
        self.collection = self._create_collection()

    def _connect(self):
        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_ENDPOINT"),
            token=os.getenv("MILVUS_TOKEN")
        )

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="page_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="Multimodal RAG content")
        return Collection(name=self.name, schema=schema)

    def insert_entries(self, entries):
        if not entries:
            return

        page_names, titles, types, contents, embeddings = zip(*entries)
        self.collection.insert([
            list(page_names),
            list(titles),
            list(types),
            list(contents),
            list(embeddings)
        ])

    def create_index(self):
        self.collection.create_index("embedding", {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        })
        self.collection.load()

    def search(self, query_vec, limit=3):
        return self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"nprobe": 10},
            limit=limit,
            output_fields=["page_name", "title", "type", "content", "embedding"]
        )

    def search_with_rerank(self, query_vec, top_k=10, rerank_k=3, image_k=3):
        hits = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"nprobe": 10},
            limit=top_k,
            output_fields=["page_name", "title", "type", "content", "embedding"]
        )[0]

        query_vec_np = np.array(query_vec).reshape(1, -1)
        text_hits = []

        for hit in hits:
            if hit.entity.get("type") != "text":
                continue
            emb = np.array(hit.entity.get("embedding")).reshape(1, -1)
            score = cosine_similarity(query_vec_np, emb)[0][0]
            text_hits.append((score, hit))

        text_hits.sort(reverse=True, key=lambda x: x[0])
        top_text_hits = [(score, hit) for score, hit in text_hits[:rerank_k] if score >= self.MIN_SIMILARITY_THRESHOLD]

        image_hits = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"nprobe": 10},
            limit=image_k,
            output_fields=["page_name", "title", "type", "content", "embedding"],
            expr='type == "image"'
        )[0]

        scored_hits = []
        for hit in image_hits:
            emb = np.array(hit.entity.get("embedding")).reshape(1, -1)
            score = cosine_similarity(query_vec_np, emb)[0][0]
            scored_hits.append((score, hit))

        scored_hits.sort(reverse=True, key=lambda x: x[0])
        reranked_image_hits = [(score, hit) for score, hit in scored_hits[:1] if score >= self.MIN_SIMILARITY_THRESHOLD]
        final_hits = [hit for score, hit in top_text_hits + reranked_image_hits]
        return final_hits if final_hits else []

    def reset_storage(self):
        if self.collection is not None:
            self.collection.drop()
            self.collection = None


    def get_existing_pairs(self):
        if not self.collection.has_index():
            return set()

        self.collection.load()
        existing_pairs = set()
        offset = 0
        batch_size = 1000

        while True:
            results = self.collection.query(
                expr="",
                output_fields=["page_name", "title", "type"],
                offset=offset,
                limit=batch_size
            )
            if not results:
                break

            for r in results:
                existing_pairs.add((r['page_name'], r['title'], r['type']))
            offset += batch_size

        return existing_pairs
