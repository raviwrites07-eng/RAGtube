"""
query.py
=========
Ask a question about your transcribed playlist. This:
1. Embeds your question
2. Finds the top 3 most semantically similar chunks from index.pkl
3. Feeds those chunks + your question to Groq's cloud API (fast, free tier)
4. Prints the generated answer, with sources (video + timestamp + clickable link)

REQUIREMENTS (install on your laptop):
    pip install sentence-transformers numpy groq python-dotenv

YOU ALSO NEED A FREE GROQ API KEY:
    1. Sign up at https://console.groq.com (no credit card needed for free tier)
    2. Create an API key there
    3. Add it to your .env file in this project folder:
         GROQ_API_KEY=your_key_here

USAGE:
    python query.py "what did they say about transformers?"
"""

import os
import sys
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Your .env file lives one folder up (in Ravi_Tiwari_ai), not inside this project folder,
# so point load_dotenv() there explicitly instead of relying on its default lookup.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

INDEX_FILE = Path("index.pkl")
VIDEO_ID_MAP_FILE = Path("audio") / "video_ids.json"
GROQ_MODEL = "openai/gpt-oss-20b"  # current free-tier model (Groq deprecated the llama-3.3 models)
TOP_K = 3


def load_index():
    if not INDEX_FILE.exists():
        print("No index.pkl found. Run build_index.py first.")
        sys.exit(1)
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def load_video_id_map() -> dict:
    """Maps a transcript's video name (e.g. '001 - Some Title') to its YouTube video ID,
    so we can build clickable timestamp links. Returns {} if not available (older transcripts
    made before this feature was added won't have it -- links just won't show for those)."""
    if VIDEO_ID_MAP_FILE.exists():
        return json.loads(VIDEO_ID_MAP_FILE.read_text(encoding="utf-8"))
    return {}


def timestamp_to_seconds(ts: str) -> int:
    """Convert HH:MM:SS -> total seconds, for building a YouTube ?t= link."""
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def build_youtube_link(video_name: str, start_ts: str, id_map: dict) -> str | None:
    """Returns a clickable YouTube link that jumps to the chunk's timestamp, or None
    if we don't have a video ID for this transcript (e.g. it predates video_ids.json)."""
    video_id = id_map.get(video_name)
    if not video_id:
        return None
    seconds = timestamp_to_seconds(start_ts)
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"


def find_top_chunks(query: str, data: dict, k: int = TOP_K):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    embeddings = data["embeddings"]
    chunks = data["chunks"]

    # Cosine similarity between query and every chunk
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10  # avoid divide-by-zero
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
        print("No GROQ_API_KEY found. Add it to your .env file:")
        print("  GROQ_API_KEY=your_key_here")
        print("Get a free key at https://console.groq.com")
        sys.exit(1)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,  # lower = more focused/factual, good for Q&A over transcripts
    )
    return response.choices[0].message.content.strip()


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    data = load_index()
    id_map = load_video_id_map()

    print(f"Searching {len(data['chunks'])} chunks for: {query}\n")
    top_chunks = find_top_chunks(query, data)

    print("Top 3 matching chunks:")
    for c, score in top_chunks:
        link = build_youtube_link(c["video"], c["start"], id_map)
        link_str = f" -> {link}" if link else ""
        print(f"  [{score:.3f}] {c['video']} @ {c['start']}{link_str}")
        print(f"        {c['text'][:80]}...")

    prompt = build_prompt(query, top_chunks)
    print("\nGenerating answer from Groq...\n")
    answer = generate_answer(prompt)

    print("=" * 50)
    print("ANSWER:")
    print(answer)
    print("=" * 50)
    print("\nSources (click to jump to that moment in the video):")
    for c, score in top_chunks:
        link = build_youtube_link(c["video"], c["start"], id_map)
        if link:
            print(f"  - {c['video']} [{c['start']} - {c['end']}]\n    {link}")
        else:
            print(f"  - {c['video']} [{c['start']} - {c['end']}]  (no video ID on file -- re-run transcribe_playlist.py to enable links)")


if __name__ == "__main__":
    main()