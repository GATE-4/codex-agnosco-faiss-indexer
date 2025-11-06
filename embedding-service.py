from fastapi import FastAPI, Body
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn
import os
import pickle

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")  # small, CPU-friendly
index_file = "embeddings/index.faiss"
meta_file = "embeddings/meta.pkl"

# FAISS index
dimension = 384
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    metadata = []

class CodeChunk(BaseModel):
    file_path: str
    symbol: str
    code: str

@app.post("/index")
def add_chunks(chunks: list[CodeChunk]):
    global metadata
    texts = [c.code for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index.add(embeddings)
    for c in chunks:
        metadata.append({"file": c.file_path, "symbol": c.symbol, "code": c.code})
    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)
    return {"status": "ok", "count": len(chunks)}

@app.post("/search")
def search(query: str = Body(...), k: int = 5):
    embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(embedding, k)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
