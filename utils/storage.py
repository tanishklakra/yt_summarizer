import json
import numpy as np

def save_embedding_to_json(video_id, embedding, metadata):
    import os, json
    output_dir = os.path.join("data/videos", video_id)
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, "embedding.json")
    data = {
        "video_id": video_id,
        "embedding": embedding.tolist(),
        "metadata": metadata
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Embedding saved to {filepath}")

