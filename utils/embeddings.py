import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP model + processor once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image_path):
    """
    Get a semantic embedding of an image using CLIP.

    Args:
        image_path (str): Path to the frame image

    Returns:
        torch.Tensor: 512-d CLIP image embedding
    """
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        embedding = outputs / outputs.norm(dim=-1, keepdim=True)

    return embedding.squeeze(0)  # 512-d vector

def get_text_embedding(text):
    """
    Get CLIP-compatible text embedding, truncated to fit within 77 token limit.
    """
    # Truncate text manually to 77 tokens worth (roughly ~400 characters)
    text = text[:400]

    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
        embedding = outputs / outputs.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0)
