import numpy as np
import pickle
import faiss

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



class FaissLSH:
    """
    An LSH-based FAISS index for storing embeddings and their associated metadata.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes the FaissLSH index.

        Keyword arguments can include:
          - nbits (int): Number of bits for hashing (default: 128)
        """
        self.dim = dim
        self.metadata = []
        nbits = kwargs.get('nbits', 128)
        self.index = faiss.IndexLSH(dim, nbits)

    def add_embeddings(self, embeddings, metadata):
        """
        Adds embeddings and metadata to the LSH index.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")
        for emb, meta in zip(embeddings, metadata):
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
        Saves the LSH index and metadata to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissLSH instance from disk.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


# Example usage:
if __name__ == "__main__":
    index = FaissLSH(dim=4, nbits=256)
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
    
    filepath = "faiss_lsh_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")
    
    loaded_index = FaissLSH.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
