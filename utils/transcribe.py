import os
import requests
from dotenv import load_dotenv

load_dotenv()

def transcribe_with_assemblyai(audio_path):
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("❌ ASSEMBLYAI_API_KEY not set in .env")

    # Step 1: Upload audio file
    upload_url = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": api_key}

    with open(audio_path, "rb") as f:
        response = requests.post(upload_url, headers=headers, files={"file": f})

    if response.status_code != 200:
        print("❌ Upload failed:", response.text)
        return None

    audio_url = response.json()["upload_url"]
    print("📤 Audio uploaded")

    # Step 2: Start transcription
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    json_data = {
        "audio_url": audio_url,
        "language_code": "en",
        "auto_chapters": False,
    }
    response = requests.post(transcript_url, headers=headers, json=json_data)

    if response.status_code != 200:
        print("❌ Transcription request failed:", response.text)
        return None

    transcript_id = response.json()["id"]
    print("📝 Transcription started, ID:", transcript_id)

    # Step 3: Poll for result
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()["status"]

        if status == "completed":
            print("✅ Transcription complete!")
            return polling_response.json()["text"]
        elif status == "error":
            print("❌ Transcription failed:", polling_response.json()["error"])
            return None

        import time
        time.sleep(3)
