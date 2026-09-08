"""
build_index.py
===============
Reads all transcripts (produced by transcribe_playlist.py), splits them into
their timestamped chunks, embeds each chunk, and saves a local searchable index.

Run this ONCE after transcription is complete (or re-run whenever you add new transcripts).

REQUIREMENTS (install on your laptop):
    pip install sentence-transformers numpy

USAGE:
    python build_index.py
"""

import re
import json
import pickle
from pathlib import Path

TRANSCRIPT_DIR = Path("transcripts")
INDEX_FILE = Path("index.pkl")

# Matches lines like: [00:00:00 - 00:00:31] some text here
LINE_PATTERN = re.compile(r"^\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\]\s*(.*)$")


def parse_transcript(path: Path):
    """Extract (video_name, start_ts, end_ts, text) tuples from one transcript file."""
    chunks = []
    video_name = path.stem
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = LINE_PATTERN.match(line.strip())
            if match:
                start_ts, end_ts, text = match.groups()
                if text.strip():
                    chunks.append({
                        "video": video_name,
                        "start": start_ts,
                        "end": end_ts,
                        "text": text.strip(),
                    })
    return chunks


def main():
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if not TRANSCRIPT_DIR.exists():
        print("No 'transcripts' folder found. Run transcribe_playlist.py first.")
        return

    all_chunks = []
    for txt_file in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        chunks = parse_transcript(txt_file)
        all_chunks.extend(chunks)
        print(f"Parsed {len(chunks)} chunks from {txt_file.name}")

    if not all_chunks:
        print("No chunks found. Check your transcript files have the expected timestamp format.")
        return

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print("Loading embedding model (first run downloads it, ~80MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in all_chunks]
    print("Embedding all chunks (this may take a few minutes for hours of transcript)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"chunks": all_chunks, "embeddings": embeddings}, f)

    print(f"\nIndex saved to {INDEX_FILE} ({len(all_chunks)} chunks).")


if __name__ == "__main__":
    main()
