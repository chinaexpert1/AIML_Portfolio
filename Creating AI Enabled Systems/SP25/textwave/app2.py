# TODO: Add your import statements
import os
import glob
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.bruteforce import FaissBruteForce


# TODO: Add you default parameters here
# For example: 
STORAGE_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed-length' # or 'sentence'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# add more as needed...


def initialize_index():


    #######################################
    # TODO: Implement initialize()
    #######################################

    # Default parameters are defined above in the file:
    # STORAGE_DIRECTORY, CHUNKING_STRATEGY, and EMBEDDING_MODEL


    """
    1. Parse through all the documents contained in storage/corpus directory.
    2. Chunk the documents using either a 'sentence' or 'fixed-length' chunking strategy.
    3. Embed each chunk using the Embedding class.
    4. Store vector embeddings of these chunks in a BruteForce FAISS index, along with the chunks as metadata.
    5. Return the FAISS index.
    """
    processing = DocumentProcessing()
    embedding_instance = Embedding(EMBEDDING_MODEL)

    # Assuming that documents are stored in a subdirectory "corpus" within STORAGE_DIRECTORY
    file_pattern = os.path.join(STORAGE_DIRECTORY, "corpus", "*")
    document_files = glob.glob(file_pattern)

    all_embeddings = []
    all_metadata = []

    for file_path in document_files:
        # Choose chunking method based on CHUNKING_STRATEGY (default is 'fixed-length')
        if CHUNKING_STRATEGY == 'sentence':
            # Using example parameters for sentence chunking
            chunks = processing.sentence_chunking(file_path, num_sentences=5, overlap_size=3)
        else:
            # Using fixed-length chunking with a default chunk_size (e.g., 500 characters)
            chunks = processing.fixed_length_chunking(file_path, chunk_size=500, overlap_size=2)

        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                vector = embedding_instance.encode(chunk)
                all_embeddings.append(vector)
                all_metadata.append(chunk)

    if not all_embeddings:
        raise ValueError("No embeddings were generated. Check your document files and chunking parameters.")

    # Assume all embeddings have the same dimension
    dim = all_embeddings[0].shape[0]

    # Create the FAISS index using the FaissBruteForce class (defaulting to Euclidean metric)
    index = FaissBruteForce(dim, metric='euclidean')
    index.add_embeddings(all_embeddings, all_metadata)

    return index




