import faiss
import numpy as np
import pickle

# Check if we're using faiss-cpu and adjust accordingly
# This is a common pattern when faiss-cpu is structured differently
if not hasattr(faiss, 'IndexFlatL2'):
    try:
        # Try different possible structures based on faiss-cpu
        import faiss.cpu as faiss
    except ImportError:
        try:
            import faiss.impl as faiss
        except ImportError:
            # Last resort - try to import faiss_cpu
            try:
                import faiss_cpu as faiss
            except ImportError:
                raise ImportError("Could not find a compatible FAISS module. Please check your installation.")


class FaissHNSW:
    """
    A FAISS HNSW index for storing embeddings and their associated metadata using
    the Hierarchical Navigable Small World (HNSW) algorithm.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes the FaissHNSW instance.

        Keyword arguments can include:
          - M (int): Number of neighbors in the HNSW graph (default: 32)
          - efConstruction (int): Construction parameter for HNSW (default: 40)
        """
        self.dim = dim
        self.metadata = []
        M = kwargs.get('M', 32)
        efConstruction = kwargs.get('efConstruction', 40)
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = efConstruction

    def add_embeddings(self, new_embeddings, new_metadata):
        """
        Adds embeddings and metadata to the HNSW index.
        """
        if len(new_embeddings) != len(new_metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")
        for emb, meta in zip(new_embeddings, new_metadata):
            emb = np.array(emb)
            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")
            self.metadata.append(meta)
            vector = emb.astype(np.float32).reshape(1, -1)
            self.index.add(vector)

    def get_metadata(self, idx):
        """
        Retrieves metadata associated with the embedding at the given index.
        """
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError("Index out of bounds.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Saves the index and metadata to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissHNSW instance from disk.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


# Example usage:
if __name__ == "__main__":
    index = FaissHNSW(dim=4, M=16, efConstruction=50)
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    identity_metadata = ["Alice", "Bob", "Charlie"]
    index.add_embeddings(embeddings, identity_metadata)
    
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    k = 2
    distances, indices = index.index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]
    
    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)
    
    filepath = "faiss_hnsw_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")
    
    loaded_index = FaissHNSW.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
