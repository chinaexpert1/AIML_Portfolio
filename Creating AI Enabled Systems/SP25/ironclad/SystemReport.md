# Significant Design Decisions for IronClad Facial Recognition System

1. **Model Selection for Embedding Extraction**  
   - **Decision:** The system employs FaceNet’s InceptionResnetV1 model pretrained on the "casia-webface" dataset for feature extraction.  
   - **Significance:** This model choice directly influences the quality of the extracted embeddings, which in turn affects identification accuracy. High-quality embeddings are critical for distinguishing among faces, while computational efficiency and resource usage are also tied to the model’s architecture and pretraining dataset.  
   - **Methodology:** By reviewing `embedding.py`, I observed the use of the FaceNet model, its configuration, and its embedding generation process. This decision was reinforced by analyzing the integration of this module within the overall system workflow.

   I timed how long it took to extract an embedding from the probe image for Aaron_Sorkin. (Code Omitted)

   Embedding shape: (512,)
    Time taken for embedding extraction: 0.0205 seconds

2. **Search/Indexing Strategy**  
   - **Decision:** The system supports multiple indexing strategies for nearest neighbor search, including BruteForce, LSH, and HNSW, with the default set to a brute-force approach.  
   - **Significance:** The choice of indexing method critically affects query speed and system scalability. Brute-force methods, while accurate, can be computationally expensive for large galleries; alternatives like LSH and HNSW provide faster query responses with acceptable trade-offs in precision. This flexibility enables tuning the system according to performance requirements and dataset size.  
   - **Methodology:** Analysis of `search.py`, `lsh.py`, `bruteforce.py`, and `hnsw.py` revealed different strategies along with key parameters (e.g., `nbits` for LSH and `M`/`efConstruction` for HNSW) that allow performance tuning, highlighting how this decision impacts the overall system efficiency.

    The following tables were constructed using the probe image of Aaron_Sorkin under different indexing strategies. (Code omitted)

    Strategy: BruteForce
    Query Time: 0.0000 sec
     Retrieved Metadata: ['Person_367', 'Person_617', 'Person_846', 'Person_974', 'Person_681']
    -------------------------------------------------
    Strategy: LSH
     Query Time: 0.0000 sec
     Retrieved Metadata: ['Person_157', 'Person_50', 'Person_743', 'Person_264', 'Person_126']
    -------------------------------------------------
    Strategy: HNSW
     Query Time: 0.0010 sec
     Retrieved Metadata: ['Person_617', 'Person_846', 'Person_974', 'Person_797', 'Person_282']
    -------------------------------------------------



3. **Similarity Metric and Distance Configuration**  
   - **Decision:** The system is designed to support a range of distance metrics—including Euclidean, Cosine, Dot Product, and Minkowski—for comparing embeddings.  
   - **Significance:** The choice of similarity metric determines how distances between embedding vectors are computed, thereby directly impacting the effectiveness of identity matching. For instance, cosine similarity, with its normalization step, can provide more robust matching in high-dimensional spaces compared to plain Euclidean distance. This configurability allows the system to adapt to different data characteristics and performance requirements.  
   - **Methodology:** Examination of conditional handling for different metrics in `search.py` and `bruteforce.py` provided insights into the flexibility of the system. The use of different metrics and associated parameters (like the Minkowski parameter `p`) supports fine-tuning for optimal performance.

---

    I compared search performance using different similarity metrics (Euclidean and Cosine) on a simulated gallery using the FaissBruteForce index. It printed out query times and retrieved metadata. (Code Omitted)

    Similarity Metric: euclidean
    Query Time: 0.0005 sec
    Retrieved Metadata: ['Person_151', 'Person_838', 'Person_368', 'Person_275', 'Person_569']
    -------------------------------------------------
    Similarity Metric: cosine
    Query Time: 0.0000 sec
    Retrieved Metadata: ['Person_838', 'Person_368', 'Person_151', 'Person_275', 'Person_743']
    -------------------------------------------------


Based on this analysis, our design could be influenced by:
- Opting for a high-quality embedding model even if it incurs higher computational cost, as the improved accuracy justifies it.
- Selecting a search/indexing strategy that aligns with the size of the gallery and required query speed; for large datasets, HNSW or LSH might be preferable over brute-force.
- Carefully choosing and potentially experimenting with different similarity metrics to balance matching accuracy and computational efficiency.


