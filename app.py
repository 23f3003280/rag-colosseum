
import json
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_FILE = "kb.json"

passages = []
passage_ids = []
vectorizer = None
passage_vectors = None
sentence_rows = []
sentence_vectors = None


def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 10
    ]


def load_kb():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for i, item in enumerate(data):
        text = item["text"]
        pid = item.get("id", i)

        match = re.search(r"\[?Passage\s+(\d+)\]?", text, re.I)
        if match:
            pid = int(match.group(1))

        rows.append((int(pid), text))

    rows.sort(key=lambda x: x[0])

    if len(rows) != 50 or [r[0] for r in rows] != list(range(50)):
        raise ValueError("Expected passages 0 through 49")

    return rows


@asynccontextmanager
async def lifespan(app: FastAPI):
    global passages, passage_ids, vectorizer
    global passage_vectors, sentence_rows, sentence_vectors

    rows = load_kb()
    passage_ids = [r[0] for r in rows]
    passages = [r[1] for r in rows]

    sentence_rows = []
    for pos, text in enumerate(passages):
        for sentence in split_sentences(text):
            sentence_rows.append({
                "position": pos,
                "id": passage_ids[pos],
                "text": sentence
            })

    all_texts = passages + [r["text"] for r in sentence_rows]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    all_vectors = vectorizer.fit_transform(all_texts)

    passage_vectors = all_vectors[:len(passages)]
    sentence_vectors = all_vectors[len(passages):]

    yield


app = FastAPI(title="RAG Colosseum API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


def suspicious_instruction(text):
    patterns = [
        r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions",
        r"reveal\s+(the\s+)?(system prompt|secrets|credentials)",
        r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
        r"execute\s+(this\s+)?command",
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def attribute_supported(question, sentence):
    q = question.lower()
    s = sentence.lower()

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


def abstain():
    return {
        "answer": None,
        "sources": [],
        "scores": [],
        "confidence": 0.1
    }


@app.get("/")
def health():
    return {
        "ok": True,
        "service": "rag-colosseum",
        "passages": len(passages),
        "model": "TF-IDF"
    }


@app.get("/health")
def health_check():
    return {"ok": bool(passages), "passages": len(passages)}


@app.post("/query")
def query_api(request: QueryRequest):

    if vectorizer is None:
        raise HTTPException(status_code=503, detail="Service loading")

    question = request.question.strip()

    qvec = vectorizer.transform([question])

    passage_sim = cosine_similarity(qvec, passage_vectors)[0]

    order = passage_sim.argsort()[::-1][:request.top_k]

    hits = []

    for pos in order:
        raw = float(passage_sim[pos])

        hits.append({
            "position": int(pos),
            "id": passage_ids[int(pos)],
            "score": raw,
            "text": passages[int(pos)]
        })

    if not hits or hits[0]["score"] < 0.10:
        return abstain()

    selected_positions = {h["position"] for h in hits}

    candidate_indices = [
        i for i, row in enumerate(sentence_rows)
        if row["position"] in selected_positions
        and not suspicious_instruction(row["text"])
        and attribute_supported(question, row["text"])
    ]

    if not candidate_indices:
        return abstain()

    sims = cosine_similarity(
        qvec,
        sentence_vectors[candidate_indices]
    )[0]

    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    best_row = sentence_rows[candidate_indices[best_idx]]

    if best_score < 0.10:
        return abstain()

    supported = [
        h for h in hits
        if h["score"] >= max(0.10, hits[0]["score"] - 0.15)
    ]

    if best_row["id"] not in [h["id"] for h in supported]:
        supported.append(
            next(h for h in hits if h["id"] == best_row["id"])
        )

    supported.sort(key=lambda h: h["score"], reverse=True)

    confidence = min(0.90, max(0.30, best_score))

    return {
        "answer": best_row["text"],
        "sources": [h["id"] for h in supported],
        "scores": [round(h["score"], 4) for h in supported],
        "confidence": round(confidence, 3)
    }
