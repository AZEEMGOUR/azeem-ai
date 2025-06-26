# 📁 main.py — Modular FastAPI with voice and vector QA
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import numpy as np
import tiktoken, os, io, json, faiss
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ✅ Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ENC = tiktoken.get_encoding("cl100k_base")

# ✅ App setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

# 🌐 Serve React index.html for all routes
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    return FileResponse("dist/index.html")
# ✅ Load vector index + metadata
INDEX_PATH = "vector_store/chipdesign/index.faiss"
META_PATH = "vector_store/chipdesign/meta.json"
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf8") as f:
    meta = json.load(f)

# ✅ Request model
class Query(BaseModel):
    question: str

# ✅ Helper: embed text
def embed_query(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding).astype("float32")

# ✅ Helper: get answer from context
def get_answer(question: str) -> str:
    vec = embed_query(question)
    _, I = index.search(vec.reshape(1, -1), 3)
    context = "\n\n".join([meta[i]["text"] for i in I[0]])

    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    system = "You are a helpful assistant. Answer only using the context."

    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return chat.choices[0].message.content.strip()

# ✅ API: Text Question
@app.post("/ask")
async def ask(query: Query):
    try:
        answer = get_answer(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# ✅ API: Voice transcription
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        print(f"📥 Received audio file: {audio.filename}")
        contents = await audio.read()
        print(f"📦 File size: {len(contents)} bytes")

        audio_file = io.BytesIO(contents)
        audio_file.name = audio.filename

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        print(f"✅ Transcription result: {transcription.strip()}")
        return {"text": transcription.strip()}

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return {"error": str(e)}

