---
title: RAG Colosseum API
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# RAG Colosseum API

A small extractive RAG service for the 50-passage technical knowledge base.

## Endpoint

`POST /query`

Example request:

```json
{
  "question": "How often does the Raft leader send heartbeats?",
  "top_k": 3
}
```

## Health

- `GET /`
- `GET /health`

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Deploy

Create a **Public Docker Space** on Hugging Face and upload all five files:
`app.py`, `kb.json`, `requirements.txt`, `Dockerfile`, and `README.md`.

The first build downloads the Sentence Transformers model. Free CPU Spaces can
have cold starts and resource limits; measure response time after the Space is
running before submitting it to the grader.

## Important

This is a baseline implementation, not a guarantee of passing all 20 probes.
Test retrieval, abstention, source attribution, prompt-injection handling, and
concurrency against the actual grader cases. The confidence value is a heuristic,
not a calibrated probability.
