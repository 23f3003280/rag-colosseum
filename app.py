import json
import os
import re
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
KB_FILE = os.getenv("KB_FILE", "kb.json")
model = None
passages = []
passage_ids = []
embeddings = None
sentence_rows = []
sentence_embeddings = None
model_lock = threading.Lock()

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]

def load_kb():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("passages", data.get("documents", data.get("data", [])))
    if not isinstance(data, list):
        raise ValueError("kb.json must contain a list of passages")
    rows = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("passage")
            pid = item.get("id", item.get("index", i))
        else:
            text, pid = str(item), i
        if not text:
            continue
        match = re.search(r"\[?Passage\s+(\d+)\]?", str(text), re.I)
        if match:
            pid = int(match.group(1))
        rows.append((int(pid), str(text)))
    rows.sort(key=lambda x: x[0])
    if len(rows) != 50 or [r[0] for r in rows] != list(range(50)):
        raise ValueError("Knowledge base must contain exactly passages 0 through 49")
    return rows

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, passages, passage_ids, embeddings
    global sentence_rows, sentence_embeddings

    rows = load_kb()
    passage_ids = [r[0] for r in rows]
    passages = [r[1] for r in rows]

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = model.encode(
        passages, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    for pos, text in enumerate(passages):
        for sentence in split_sentences(text):
            sentence_rows.append({"position": pos, "id": passage_ids[pos], "text": sentence})

    sentence_embeddings = model.encode(
        [r["text"] for r in sentence_rows],
        convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)
    yield

app = FastAPI(title="RAG Colosseum API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

def suspicious_instruction(text):
    # Retrieved text is evidence only; never execute instructions found in it.
    patterns = [
        r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions",
        r"reveal\s+(the\s+)?(system prompt|secrets|credentials)",
        r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
        r"execute\s+(this\s+)?command",
    ]
    return any(re.search(p, text, re.I) for p in patterns)

def retrieve(question, top_k):
    qvec = model.encode(
        [question], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    )[0].astype(np.float32)
    sims = embeddings @ qvec
    order = np.argsort(sims)[::-1][:top_k]
    hits = []
    for pos in order:
        raw = float(sims[pos])
        hits.append({
            "position": int(pos),
            "id": passage_ids[int(pos)],
            "score": round(max(0.0, min(1.0, raw)), 4),
            "raw_score": raw,
            "text": passages[int(pos)]
        })
    return qvec, hits

def attribute_supported(question, sentence):
    # Conservative checks for common exact-value questions.
    q, s = question.lower(), sentence.lower()
    checks = {
        "password": ("password", "credential", "secret"),
        "administrator password": ("password",),
        "port": ("port",),
        "heartbeat": ("heartbeat", "keepalive"),
        "learning rate": ("learning rate",),
        "quorum": ("quorum",),
        "timeout": ("timeout", "timed out"),
        "key length": ("bits", "key length"),
        "round trip": ("round trip", "round-trip"),
        "retry-after": ("retry-after",),
        "retry after": ("retry-after", "retry after"),
    }
    for attribute, terms in checks.items():
        if attribute in q and not any(term in s for term in terms):
            return False
    return True

def answerable_type_guard(question, sentence):
    # Avoid claiming an answer to a field that is absent from the evidence.
    q = question.lower()
    s = sentence.lower()
    if any(x in q for x in ("default password", "admin password", "administrator password")):
        return any(x in s for x in ("password", "credential"))
    return attribute_supported(question, sentence)

def abstain():
    return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

@app.get("/")
def health():
    return {
        "ok": True,
        "service": "rag-colosseum",
        "passages": len(passages),
        "model": MODEL_NAME
    }

@app.get("/health")
def health_check():
    return {"ok": model is not None, "passages": len(passages)}

@app.post("/query")
def query_api(request: QueryRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    question = request.question.strip()
    with model_lock:
        qvec, hits = retrieve(question, request.top_k)

    if not hits or hits[0]["raw_score"] < 0.34:
        return abstain()

    selected_positions = {h["position"] for h in hits}
    best = None
    best_score = -1.0
    best_pid = None

    for i, row in enumerate(sentence_rows):
        if row["position"] not in selected_positions:
            continue
        sentence = row["text"]
        if suspicious_instruction(sentence):
            continue
        if not answerable_type_guard(question, sentence):
            continue
        semantic = float(sentence_embeddings[i] @ qvec)
        passage_score = next(h["raw_score"] for h in hits if h["position"] == row["position"])
        combined = 0.8 * semantic + 0.2 * passage_score
        if combined > best_score:
            best, best_score, best_pid = sentence, combined, row["id"]

    if best is None or best_score < 0.30:
        return abstain()

    # Include the sentence's source and other close supporting retrievals.
    supported = [
        h for h in hits
        if h["raw_score"] >= max(0.28, hits[0]["raw_score"] - 0.12)
    ]
    if best_pid not in [h["id"] for h in supported]:
        supported.append(next(h for h in hits if h["id"] == best_pid))
    supported.sort(key=lambda h: h["raw_score"], reverse=True)

    confidence = min(0.95, max(0.61, 0.61 + (hits[0]["raw_score"] - 0.34) * 0.65))
    return {
        "answer": best,
        "sources": [h["id"] for h in supported],
        "scores": [h["score"] for h in supported],
        "confidence": round(confidence, 3)
    }
