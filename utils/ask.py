import requests
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import faiss, json
import tiktoken
import os

router = APIRouter()
ENC = tiktoken.get_encoding("cl100k_base")

# Preload vector and metadata
INDEX_PATH = "vector_store/chipdesign/index.faiss"
META_PATH = "vector_store/chipdesign/meta.json"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf8") as f:
    meta = json.load(f)

class Query(BaseModel):
    question: str

def embed_query(text: str) -> np.ndarray:
    # Ollama embedding call
    res = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    }).json()

    if "embedding" not in res:
        raise Exception("Embedding failed: " + str(res))
    
    return np.array(res["embedding"], dtype="float32")

@router.post("/ask")
async def ask(query: Query):
    try:
        print("🔍 Question:", query.question)

        vec = embed_query(query.question)
        _, I = index.search(vec.reshape(1, -1), 3)
        context = "\n\n".join([meta[i]["text"] for i in I[0]])

        prompt = f"Answer the question using only this context:\n\n{context}\n\nQuestion: {query.question}"
        print("📤 Prompt:", prompt[:300])

        # Final Ollama call (non-streaming)
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        })

        print("🟡 Status:", r.status_code)

        data = r.json()
        print("📥 Raw response:", data)

        return {"answer": data.get("response", "").strip()}

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}
