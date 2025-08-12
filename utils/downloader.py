import os
from yt_dlp import YoutubeDL

def download_video_audio(url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Download best pre-merged mp4
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'noplaylist': True,
        'postprocessors': []  # Avoid using ffmpeg
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    print(f"✅ Downloaded file: {filename}")
    return filename, filename  # Return same path for video and audio
