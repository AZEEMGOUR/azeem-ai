# ✅ pdf_ingest.py
import os, json, faiss, numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
import tiktoken

# Load env and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ENC = tiktoken.get_encoding("cl100k_base")

PDF_PATH = "data/Introduction_to_Semiconductor_Chip_Design.pdf"
PDF_ID = "chipdesign"
VECTOR_STORE = f"vector_store/{PDF_ID}"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 100

os.makedirs(VECTOR_STORE, exist_ok=True)

def get_chunks(text):
    tokens = ENC.encode(text)
    chunks = []
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunks.append(ENC.decode(chunk_tokens).strip())
    return chunks

def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embs = [np.array(obj.embedding, dtype="float32") for obj in res.data]
        embeddings.extend(embs)
    return np.array(embeddings)

print("📄 Extracting and chunking PDF...")
raw_text = extract_text(PDF_PATH)
chunks = get_chunks(raw_text)
print(f"🔹 {len(chunks)} chunks created.")

print("🧠 Embedding chunks...")
vectors = embed_texts(chunks)

print("📦 Building FAISS index...")
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)
faiss.write_index(index, f"{VECTOR_STORE}/index.faiss")

with open(f"{VECTOR_STORE}/meta.json", "w", encoding="utf8") as f:
    json.dump([{"text": c} for c in chunks], f, ensure_ascii=False, indent=2)

print("✅ FAISS and metadata saved.")
