import cv2
import os

def extract_frames(video_path, video_id, interval_sec=5):
    """
    Extract frames to a per-video folder.
    """
    frame_output_dir = os.path.join("data/videos", video_id, "frames")
    os.makedirs(frame_output_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    success, image = vidcap.read()
    count = 0
    saved_frames = []

    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(frame_output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, image)
            saved_frames.append(frame_path)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return saved_frames
