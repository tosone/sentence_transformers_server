import os
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from sentence_transformers import SentenceTransformer, util

# gpu batch_size in order of your available vram
batch_size = 32 if os.getenv('ST_BATCH_SIZE') == "" or os.getenv(
    'ST_BATCH_SIZE') == None else int(os.getenv('ST_BATCH_SIZE'))
# max context length for embeddings and passages in re-ranker
max_length = 8192 if os.getenv('ST_MAX_LENGTH') == "" or os.getenv(
    'ST_MAX_LENGTH') == None else int(os.getenv('ST_MAX_LENGTH'))
# max context length for questions in re-ranker
max_query_length = 512 if os.getenv('ST_MAX_QUERY_LENGTH') == "" or os.getenv(
    'ST_MAX_QUERY_LENGTH') == None else int(os.getenv('ST_MAX_QUERY_LENGTH'))
# model name or the model path
model_name = 'BAAI/bge-m3' if os.getenv('ST_MODEL_NAME') == "" or os.getenv(
    'ST_MODEL_NAME') == None else os.getenv('ST_MODEL_NAME')

device = util.get_device_name()


class SentenceTransformersWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(
            model_name_or_path=model_name, device=device)

    def embedding(self, sentences: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
        )
        return embeddings.tolist()

    def reranker(self, query: str, sentences: List[str]) -> List[Dict[str, int | float]]:
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
        )
        query_embedding = self.model.encode(query)
        return [{'index': item['corpus_id'], 'score': item['score']} for sublist in util.semantic_search(query_embedding, embeddings) for item in sublist]


class EmbeddingResponse(BaseModel):
    vectors: List[List[float]]


class EmbeddingRequest(BaseModel):
    sentences: List[str] = Field(
        title="The reranker sentences", max_length=8192, max_items=10
    )


class RerankerRequest(BaseModel):
    query: str = Field(
        title="The reranker query string", max_length=8192
    )
    sentences: List[str] = Field(
        title="The reranker sentences", max_length=8192, max_items=10
    )


class RerankerResponse(BaseModel):
    scores: List[Dict[str, int | float]]


model = SentenceTransformersWrapper(model_name)

app = FastAPI()


@app.post("/embedding", response_model=EmbeddingResponse)
async def embedding(request: EmbeddingRequest):
    return EmbeddingResponse(vectors=model.embedding(request.sentences))


@app.post("/reranker", response_model=RerankerResponse)
async def reranker(request: RerankerRequest):
    return RerankerResponse(scores=model.reranker(request.query, request.sentences))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
# curl -X POST -k -v http://127.0.0.1:3000/reranker -H "Content-Type: application/json" -d '{"query": "What is BGE M3?","sentences":["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", "xxx"]}'
# curl -X POST -k -v http://127.0.0.1:3000/embedding -H "Content-Type: application/json" -d '{"sentences":["xx"]}'
