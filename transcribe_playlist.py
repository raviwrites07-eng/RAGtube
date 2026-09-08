"""
YouTube Playlist -> Timestamped Text Transcript
=================================================

WHAT THIS DOES:
1. Downloads audio-only from every video in a YouTube playlist (via yt-dlp)
2. Transcribes each audio file using faster-whisper (runs on CPU, no GPU needed)
3. Outputs plain text with timestamps, chunked into ~30 second segments
4. Saves progress after each video so you can safely stop and resume

REQUIREMENTS (install these first on your laptop, not here):
    pip install yt-dlp faster-whisper

    You also need ffmpeg installed on your system:
    - Windows: download from https://ffmpeg.org/download.html and add to PATH
      (or easier: winget install ffmpeg   -- run in Command Prompt / PowerShell)

USAGE:
    python transcribe_playlist.py "https://www.youtube.com/playlist?list=XXXXXXX"

OUTPUT:
    ./audio/               <- downloaded audio files (kept, so you can re-run without re-downloading)
    ./transcripts/         <- one .txt file per video, with timestamps
    ./progress.json        <- tracks which videos are already done (for resume)
"""

import sys
import os
import json
import subprocess
import shutil
from pathlib import Path

# ---------- CONFIG (feel free to tweak these) ----------
CHUNK_SECONDS = 30          # your requested chunking interval (for the output text)
MODEL_SIZE = "small"        # balanced choice: tiny/base/small/medium/large
                             # "small" = good balance of speed vs accuracy on CPU
COMPUTE_TYPE = "int8"       # int8 = fastest on CPU, uses less RAM
AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcripts")
PROGRESS_FILE = Path("progress.json")
SPLIT_DIR = Path("audio_splits")   # temp folder for splitting long files
SPLIT_THRESHOLD_SECONDS = 900      # files longer than 15 min get split before transcribing
SPLIT_LENGTH_SECONDS = 900         # each split piece is ~15 min -- small enough to avoid the memory error
# ---------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Convert seconds -> HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"downloaded": [], "transcribed": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def download_playlist(playlist_url: str) -> None:
    """Use yt-dlp to download audio-only, one file per video, skipping any already downloaded."""
    AUDIO_DIR.mkdir(exist_ok=True)
    print("Starting playlist download (audio only)...")

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",       # extract audio, convert to mp3
        "--audio-quality", "5",              # decent quality, smaller file size (0=best,9=worst)
        "-o", str(AUDIO_DIR / "%(playlist_index)03d - %(title)s.%(ext)s"),
        "--download-archive", str(AUDIO_DIR / "downloaded.txt"),  # skips already-downloaded videos automatically
        "--ignore-errors",                   # don't stop the whole playlist if one video fails
        playlist_url,
    ]
    subprocess.run(cmd, check=True)
    print("Download step complete.\n")

    save_video_id_map(playlist_url)


def save_video_id_map(playlist_url: str) -> None:
    """Ask yt-dlp for each video's title + YouTube video ID (no re-download, just metadata),
    and save a mapping so query.py can build clickable timestamp links later."""
    map_path = AUDIO_DIR / "video_ids.json"

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(playlist_index)03d|||%(title)s|||%(id)s",
        playlist_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    id_map = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|||")
        if len(parts) == 3:
            index, title, video_id = parts
            # This key format matches how transcript filenames are built: "001 - Title"
            key = f"{index} - {title}"
            id_map[key] = video_id

    map_path.write_text(json.dumps(id_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved video ID mapping to {map_path}\n")


def get_audio_duration(audio_path: Path) -> float:
    """Use ffprobe to get duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrapped_values=1:nokey=1", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_audio(audio_path: Path, piece_seconds: int) -> list[Path]:
    """Split a long audio file into smaller pieces using ffmpeg (avoids the memory error
    faster-whisper hits when processing very long files in one go on low-RAM machines).
    Returns list of piece file paths, in order."""
    split_subdir = SPLIT_DIR / audio_path.stem
    split_subdir.mkdir(parents=True, exist_ok=True)

    pattern = split_subdir / "piece_%03d.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-f", "segment", "-segment_time", str(piece_seconds),
        "-c", "copy", str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    pieces = sorted(split_subdir.glob("piece_*.mp3"))
    return pieces


def transcribe_all() -> None:
    """Transcribe every audio file that hasn't been transcribed yet."""
    from faster_whisper import WhisperModel

    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    progress = load_progress()

    print(f"Loading Whisper model '{MODEL_SIZE}' (this can take a minute the first time)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

    audio_files = sorted(AUDIO_DIR.glob("*.mp3"))
    if not audio_files:
        print("No audio files found. Did the download step run correctly?")
        return

    for audio_path in audio_files:
        if audio_path.name in progress["transcribed"]:
            print(f"Skipping (already done): {audio_path.name}")
            continue

        print(f"\nTranscribing: {audio_path.name}")

        duration = get_audio_duration(audio_path)
        if duration > SPLIT_THRESHOLD_SECONDS:
            print(f"  (long file, {duration/60:.0f} min -- splitting into ~{SPLIT_LENGTH_SECONDS//60} min pieces to avoid memory errors)")
            pieces = split_audio(audio_path, SPLIT_LENGTH_SECONDS)
        else:
            pieces = [audio_path]

        out_path = TRANSCRIPT_DIR / (audio_path.stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript for: {audio_path.name}\n\n")

            # time_offset tracks how far into the ORIGINAL full file we are,
            # so timestamps stay correct across pieces
            time_offset = 0.0

            for piece_idx, piece_path in enumerate(pieces):
                if len(pieces) > 1:
                    print(f"  Piece {piece_idx + 1}/{len(pieces)}: {piece_path.name}")

                segments, info = model.transcribe(
                    str(piece_path),
                    vad_filter=True,          # skips silence, speeds things up
                )

                # Group whisper's natural segments into ~30-second chunks
                chunk_start = None
                chunk_text = []
                chunk_end = 0

                for seg in segments:
                    if chunk_start is None:
                        chunk_start = seg.start + time_offset
                    chunk_text.append(seg.text.strip())
                    chunk_end = seg.end + time_offset

                    if chunk_end - chunk_start >= CHUNK_SECONDS:
                        f.write(f"[{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}] "
                                f"{' '.join(chunk_text)}\n")
                        chunk_start = None
                        chunk_text = []

                # Write any leftover partial chunk for this piece
                if chunk_text:
                    f.write(f"[{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}] "
                            f"{' '.join(chunk_text)}\n")

                # Advance the offset by this piece's actual duration for the next piece
                time_offset += get_audio_duration(piece_path)

        # Clean up split pieces for this video now that it's fully transcribed
        if len(pieces) > 1:
            shutil.rmtree(SPLIT_DIR / audio_path.stem, ignore_errors=True)

        print(f"Saved: {out_path}")
        progress["transcribed"].append(audio_path.name)
        save_progress(progress)  # save after EVERY file, so a crash never loses more than 1 video

    print("\nAll done! Check the 'transcripts' folder.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_playlist.py <playlist_url>")
        sys.exit(1)

    playlist_url = sys.argv[1]

    download_playlist(playlist_url)
    transcribe_all()


if __name__ == "__main__":
    main()