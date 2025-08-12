import os
import hashlib

from utils.downloader import download_video_audio
from utils.audio import extract_audio
from utils.transcribe import transcribe_with_groq_whisper
from utils.frames import extract_frames
from utils.embeddings import get_clip_embedding, get_text_embedding
from utils.fusion import fuse_embeddings
from utils.storage import save_embedding_to_json
from utils.vectorstore import get_or_create_collection, add_video_to_collection

def summarize_youtube_video(url):
    print("🔽 Downloading video and extracting audio...")
    video_path, _ = download_video_audio(url)
    audio_path = extract_audio(video_path)

    print("🗣️ Transcribing audio using Groq Whisper...")
    transcription_result = transcribe_with_groq_whisper(audio_path)
    if not transcription_result:
        print("❌ Transcription failed. Exiting.")
        return
    text = transcription_result['text']

    print("🖼️ Extracting frames and computing CLIP embeddings...")
    frame_paths = extract_frames(video_path, interval_sec=5)
    frame_embeddings = [get_clip_embedding(p) for p in frame_paths]

    print("✍️ Embedding transcribed text...")
    text_embedding = get_text_embedding(text)

    print("🧠 Fusing text and visual embeddings...")
    video_embedding = fuse_embeddings(text_embedding, frame_embeddings)

    os.makedirs("data/embeddings", exist_ok=True)
    video_id = hashlib.md5(url.encode()).hexdigest()
    json_filepath = f"data/embeddings/{video_id}.json"
    save_embedding_to_json(
        filepath=json_filepath,
        video_id=video_id,
        embedding=video_embedding,
        metadata={
            "url": url,
            "text": text,
            "title": "TODO: fetch video title"
        }
    )

    collection = get_or_create_collection()
    add_video_to_collection(collection, video_id, video_embedding, {
        "url": url,
        "text": text,
        "title": "TODO: fetch video title"
    })
    print("📦 Video data stored in ChromaDB.")

    print("\n--- Transcript Preview ---\n")
    print(text[:500] + "..." if len(text) > 500 else text)

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "transcription": text,
        "video_embedding": video_embedding
    }


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Ingest and embed a YouTube video")
    print("2. Query stored video summaries")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        url = input("📺 Enter YouTube video URL: ").strip()
        summarize_youtube_video(url)
    elif choice == "2":
        import utils.query as query  # assuming query.py is in the same folder or PYTHONPATH
        query.collection = get_or_create_collection()
        query_main = query.__name__  # Just to avoid unused warning

        while True:
            q = input("\n❓ Ask a question (or 'exit'): ").strip()
            if q.lower() == "exit":
                break
            ans = query.answer_question(query.collection, q)
            print("\n🤖 Answer:\n", ans)
    else:
        print("❌ Invalid choice. Exiting.")
