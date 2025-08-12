import chromadb

def get_or_create_collection():
    # Initialize Chroma client with persistence directory
    # The new way uses PersistentClient instead of Client with settings
    client = chromadb.PersistentClient(path="./chromadb_data")
    
    # Get or create collection with optional metadata (e.g. similarity metric)
    collection = client.get_or_create_collection(
        name="youtube_summarizer",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for vector search
    )
    
    return client, collection

def add_video_to_collection(client, collection, video_id, embedding, metadata):
    # Ensure video_id is part of metadata for filtering later
    metadata["video_id"] = video_id

    collection.add(
        documents=[metadata["text"]],
        embeddings=[embedding.tolist()],
        metadatas=[metadata],
        ids=[video_id]
    )

    
    # Note: PersistentClient automatically persists changes
    # No need to call client.persist() explicitly