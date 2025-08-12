import numpy as np

def fuse_embeddings(text_embedding, frame_embeddings):
    """
    Fuse text and visual embeddings into one vector.

    Args:
        text_embedding (np.array or tensor)
        frame_embeddings (List of np.array or tensors)

    Returns:
        np.array: Unified embedding
    """
    all_embeddings = [text_embedding] + frame_embeddings
    all_embeddings = [e if isinstance(e, np.ndarray) else e.cpu().numpy() for e in all_embeddings]
    avg_embedding = np.mean(all_embeddings, axis=0)
    return avg_embedding
