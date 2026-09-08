"""
rag_core.py
============
Shared logic for searching your transcripts and generating answers via Groq.
Used by both query.py (terminal version) and app.py (web page version) so the
core logic lives in one place.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Your .env file lives one folder up (in Ravi_Tiwari_ai), not inside this project folder.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

INDEX_FILE = Path("index.pkl")
VIDEO_ID_MAP_FILE = Path("audio") / "video_ids.json"
GROQ_MODEL = "openai/gpt-oss-20b"  # current free-tier model (Groq deprecated the llama-3.3 models)
TOP_K = 3

_embedding_model = None  # loaded once, reused across queries (important for the web app)


def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError("No index.pkl found. Run build_index.py first.")
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def load_video_id_map() -> dict:
    if VIDEO_ID_MAP_FILE.exists():
        return json.loads(VIDEO_ID_MAP_FILE.read_text(encoding="utf-8"))
    return {}


def timestamp_to_seconds(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def build_youtube_link(video_name: str, start_ts: str, id_map: dict):
    video_id = id_map.get(video_name)
    if not video_id:
        return None
    seconds = timestamp_to_seconds(start_ts)
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"


def get_embedding_model():
    """Load the embedding model once and reuse it -- avoids reloading on every
    question in the web app, which would make each search slow."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def find_top_chunks(query: str, data: dict, k: int = TOP_K):
    import numpy as np

    model = get_embedding_model()
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    embeddings = data["embeddings"]
    chunks = data["chunks"]

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10
    similarities = (embeddings @ query_vec) / norms

    top_indices = np.argsort(similarities)[::-1][:k]
    return [(chunks[i], float(similarities[i])) for i in top_indices]


def build_prompt(query: str, top_chunks) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['video']} at {c['start']}]\n{c['text']}"
        for c, score in top_chunks
    )
    return f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so honestly.

Context:
{context_block}

Question: {query}

Answer:"""


def generate_answer(prompt: str) -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No GROQ_API_KEY found. Add it to your .env file:\n"
            "  GROQ_API_KEY=your_key_here\n"
            "Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def answer_question(query: str):
    """One-call convenience function: search + generate answer + build links.
    Returns a dict with everything the UI needs to display."""
    data = load_index()
    id_map = load_video_id_map()

    top_chunks = find_top_chunks(query, data)
    prompt = build_prompt(query, top_chunks)
    answer = generate_answer(prompt)

    sources = []
    for c, score in top_chunks:
        link = build_youtube_link(c["video"], c["start"], id_map)
        sources.append({
            "video": c["video"],
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "score": score,
            "link": link,
        })

    return {"answer": answer, "sources": sources, "total_chunks": len(data["chunks"])}