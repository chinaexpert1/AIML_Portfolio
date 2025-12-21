import numpy as np
import faiss  # Ensure faiss is imported

class FaissSearch:
    def __init__(self, faiss_index, metric='euclidean', p=3):
        """
        Initialize the search class with a FaissIndex instance and distance metric.
        """
        self.index = faiss_index.index
        self.metric = metric.lower()
        self.p = p  # parameter for Minkowski distance
        self.faiss_index = faiss_index

    def search(self, query_vector, k=5):
        """
        Perform a nearest neighbor search and retrieve the associated metadata.
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.metric == 'euclidean':
            # Perform standard L2 (Euclidean) search.
            distances, indices = self.index.search(query_vector, k)
            
        elif self.metric == 'cosine':
            # For cosine similarity, normalize the query vector.
            faiss.normalize_L2(query_vector)
            distances, indices = self.index.search(query_vector, k)
            
        elif self.metric == 'dot_product':
            # For dot product, assume the index was built accordingly.
            distances, indices = self.index.search(query_vector, k)
            
        elif self.metric == 'minkowski':
            # Candidate selection followed by re-ranking using the Minkowski distance.
            candidate_k = max(50, k * 10)
            distances_candidate, indices_candidate = self.index.search(query_vector, candidate_k)
            candidate_vectors = np.array([
                self.index.reconstruct(int(idx)) for idx in indices_candidate[0]
            ])
            minkowski_distances = self._compute_minkowski(query_vector[0], candidate_vectors, self.p)
            sorted_idx = np.argsort(minkowski_distances)[:k]
            selected_indices = indices_candidate[0][sorted_idx]
            distances = np.array(minkowski_distances)[sorted_idx].reshape(1, -1)
            indices = selected_indices.reshape(1, -1)
            
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', 'dot_product', or 'minkowski'.")

        # Retrieve metadata for the found indices.
        metadata_results = [self.faiss_index.get_metadata(int(i)) for i in indices[0]]
        return distances, indices, metadata_results

    def _compute_minkowski(self, query_vector, nearest_vectors, p):
        """
        Compute Minkowski distance between the query vector and candidate vectors.
        """
        distances = np.sum(np.abs(nearest_vectors - query_vector) ** p, axis=1) ** (1 / p)
        return distances


# Example usage:
if __name__ == "__main__":
    import numpy as np
    from index.bruteforce import FaissBruteForce  # assuming this is available

    # Create some random vectors.
    vectors = np.random.random((10000, 256)).astype('float32')
    metadata = [f"Vector_{i}" for i in range(10000)]
    query_vector = np.random.random((1, 256)).astype('float32')
    
    # Construct the brute-force index with Euclidean metric.
    faiss_index_bf = FaissBruteForce(dim=256, metric="euclidean")
    faiss_index_bf.add_embeddings(vectors, metadata=metadata)

    print("\nExample: BruteForce Search with `euclidean` measure")
    search_euclidean = FaissSearch(faiss_index_bf, metric='euclidean')
    distances, indices, meta_results = search_euclidean.search(query_vector, k=5)
    for i in range(5):
        print(f"Nearest Neighbor {i+1}: Index {indices[0][i]}, Distance {distances[0][i]}, Metadata: {meta_results[i]}")

    print("\nExample: BruteForce Search with `cosine` measure")
    search_cosine = FaissSearch(faiss_index_bf, metric='cosine')
    distances, indices, meta_results = search_cosine.search(query_vector, k=5)
    for i in range(5):
        print(f"Nearest Neighbor {i+1}: Index {indices[0][i]}, Cosine Similarity {distances[0][i]}, Metadata: {meta_results[i]}")
