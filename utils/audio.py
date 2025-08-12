import os
import subprocess

def extract_audio(video_path, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, os.path.basename(video_path).rsplit('.', 1)[0] + ".mp3")
    command = [
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a", audio_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"🎧 Audio saved to {audio_path}")
    return audio_path
