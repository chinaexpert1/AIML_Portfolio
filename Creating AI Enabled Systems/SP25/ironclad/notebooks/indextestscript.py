

from PIL import Image
import torch
import numpy as np
from pathlib import Path
import os
import sys

# Add the modules directory to the path if needed
sys.path.insert(0, os.path.abspath(os.path.join('..', 'modules')))

from extraction.embedding import Embedding
from extraction.preprocessing import Preprocessing

# Initialize the preprocessing pipeline and embedding model once.
preprocessing = Preprocessing(image_size=160)
device = 'cpu'
embedding_model = Embedding(pretrained='casia-webface', device=device)


def compute_batch_embeddings(image_paths, batch_size=32):
    """
    Compute embeddings for a list of image paths in batches.
    """
    embeddings = []
    metadata = []
    batch_images = []
    batch_names = []
    
    for path in image_paths:
        p = Path(path)
        # Skip files starting with "._"
        if p.name.startswith("._"):
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
            continue
        
        processed_image = preprocessing.process(image)
        batch_images.append(processed_image)
        batch_names.append(p.name)
        
        if len(batch_images) == batch_size:
            batch_tensor = torch.cat(batch_images, dim=0)
            batch_embeddings = embedding_model.encode(batch_tensor)
            if batch_embeddings.ndim == 1:
                batch_embeddings = np.expand_dims(batch_embeddings, axis=0)
            for emb in batch_embeddings:
                embeddings.append(emb)
            metadata.extend(batch_names)
            batch_images = []
            batch_names = []
    
    if batch_images:
        batch_tensor = torch.cat(batch_images, dim=0)
        batch_embeddings = embedding_model.encode(batch_tensor)
        if batch_embeddings.ndim == 1:
            batch_embeddings = np.expand_dims(batch_embeddings, axis=0)
        for emb in batch_embeddings:
            embeddings.append(emb)
        metadata.extend(batch_names)
    
    return embeddings, metadata

# Set the gallery directory
gallery_dir = Path('../storage/multi_image_gallery')
image_paths = [
    str(p) for p in gallery_dir.rglob('*') 
    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and not p.name.startswith("._")
]
print(f"Found {len(image_paths)} images in gallery.")

# Compute embeddings in batches.
embeddings, metadata = compute_batch_embeddings(image_paths, batch_size=32)
embeddings = np.array(embeddings)
print(f"Processed {len(embeddings)} images with embeddings.")

# Import FAISS (will be correctly handled in the index classes)
import faiss

# Import the index classes
from retrieval.index.bruteforce import FaissBruteForce
from retrieval.index.hnsw import FaissHNSW
from retrieval.index.lsh import FaissLSH

try:
    # Create the indices on CPU
    print("Creating BruteForce index...")
    bf_index = FaissBruteForce(dim=256, metric='euclidean')
    
    print("Creating HNSW index...")
    hnsw_index = FaissHNSW(dim=256, M=32, efConstruction=40)
    
    print("Creating LSH index...")
    lsh_index = FaissLSH(dim=256, nbits=128)

    # Ensure embeddings is a NumPy array of type float32
    embeddings = embeddings.astype('float32')

    # Add embeddings to each index
    print("Adding embeddings to BruteForce index...")
    bf_index.metadata.extend(metadata)
    bf_index.index.add(embeddings)

    print("Adding embeddings to HNSW index...")
    hnsw_index.metadata.extend(metadata)
    hnsw_index.index.add(embeddings)

    print("Adding embeddings to LSH index...")
    lsh_index.metadata.extend(metadata)
    lsh_index.index.add(embeddings)

    print("Embeddings added to all indices.")
    
except Exception as e:
    print(f"Error during index creation or embedding addition: {e}")