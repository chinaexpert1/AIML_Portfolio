import sys
import os

# Safely resolve the parent directory one level up
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_path = os.getcwd()  # Fallback for interactive mode or notebooks

# Go one level up
parent_dir = os.path.abspath(os.path.join(base_path, ".."))

# Add to sys.path if not already present
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import requests
from collections import Counter
from modules.utils.text_processing import process_text
from modules.utils.bow import Bag_of_Words
from modules.utils.tfidf import TF_IDF
import math
import requests
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize
import nltk
nltk.data.path.append("nltk_data")

# nltk.download("punkt_tab")

import nltk

nltk.download("punkt", download_dir="nltk_data")
nltk.download("wordnet", download_dir="nltk_data")
nltk.download("omw-1.4", download_dir="nltk_data")
from nltk.tokenize import sent_tokenize
print(sent_tokenize("Hello. How are you? This is a test."))


import os
os.environ["MISTRAL_API_KEY"] = "GDTId8eQPtNGoVAhqkr5hel3mKqtoD1j"




def preprocess_corpus(corpus, mode="none"):
    """
    Preprocess a list of documents using a selected mode.

    Parameters:
        corpus (list of str): The text documents to process.
        mode (str): One of 'none', 'stem', or 'lemma'.

    Returns:
        list of str: The processed corpus.
    """
    processed = []

    if mode == "none":
        processed = [text.lower() for text in corpus]
    elif mode == "stem":
        processed = [process_text(text, use_stemming=True) for text in corpus]
    elif mode == "lemma":
        processed = [process_text(text, use_lemmatization=True) for text in corpus]
    else:
        raise ValueError("Invalid mode. Use 'none', 'stem', or 'lemma'.")

    return processed


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two sparse dict vectors.
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[k] * vec2[k] for k in intersection)

    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def retrieve_relevant_docs(query, vectorizer, corpus, top_k=3):
    """
    Returns top_k documents most relevant to the query based on cosine similarity.
    """
    query_vector = vectorizer.transform(query)
    similarities = []

    for doc in corpus:
        doc_vector = vectorizer.transform(doc) 
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc, sim))

    top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [doc for doc, _ in top_docs]


def rag_pipeline(query, corpus, vectorizer_type="tfidf", top_k=3):
    """
    Full RAG pipeline: preprocess corpus, retrieve relevant documents, query Mistral.
    Includes debug logging and failure protection.
    """
    import traceback

    print("[INFO] Preprocessing corpus...")
    clean_corpus = preprocess_corpus(corpus)

    if vectorizer_type == "tfidf":
        print("[INFO] Fitting TF-IDF vectorizer...")
        vectorizer = TF_IDF().fit(clean_corpus)
    elif vectorizer_type == "bow":
        print("[INFO] Fitting Bag-of-Words vectorizer...")
        vectorizer = Bag_of_Words().fit(clean_corpus)
    else:
        raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'bow'.")

    try:
        print("[INFO] Transforming sample doc to test vectorizer...")
        test_vector = vectorizer.transform(clean_corpus[0])
        print(f"[DEBUG] Sample vector (first 5 items): {list(test_vector.items())[:5]}")
    except Exception as e:
        print("[ERROR] Vectorizer transform failed:", e)
        traceback.print_exc()
        raise

    print("[INFO] Retrieving relevant documents...")
    relevant_docs = retrieve_relevant_docs(query, vectorizer, clean_corpus, top_k)

    prompt = "Use the following documents to answer the query:\n\n"
    for i, doc in enumerate(relevant_docs):
        prompt += f"[Document {i+1}]: {doc}\n"
    prompt += f"\n[Query]: {query}"

    print("[DEBUG] Prompt to be sent to Mistral:")
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    try:
        print("[INFO] Querying Mistral API...")
        response = query_mistral(prompt)
        print("[INFO] Received response from Mistral.")
    except Exception as e:
        print("[ERROR] Mistral API call failed:", e)
        traceback.print_exc()
        raise

    return response

def query_mistral(prompt):
    """
    Sends the prompt to the Mistral API and returns the response.
    Includes detailed logging of status codes and full response.
    """
    import requests
    import os
    import traceback

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        print("[DEBUG] Sending request to Mistral API...")
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        print(f"[DEBUG] Status Code: {response.status_code}")
        print("[DEBUG] Response Text:", response.text[:500])

        # Dump full response to log file
        with open("mistral_raw_response.log", "a", encoding="utf-8") as logf:
            logf.write(f"STATUS {response.status_code}\n{response.text}\n\n")

        if response.status_code != 200:
            print(f"[ERROR] Non-200 status from Mistral: {response.status_code}")
            return f"[ERROR] Status {response.status_code}: {response.text}"

        try:
            parsed = response.json()
            print("[DEBUG] Parsed JSON keys:", list(parsed.keys()))
            return parsed["choices"][0]["message"]["content"]
        except Exception as json_err:
            print("[ERROR] Failed to parse JSON:")
            traceback.print_exc()
            return "[ERROR] Could not decode Mistral response."

    except Exception as e:
        print("[ERROR] Exception during API request:")
        traceback.print_exc()
        return "[ERROR] Mistral API call failed."



def retrieve_top_k(query, corpus, vectorizer, k=3):
    query_vec = vectorizer.transform(query)
    if not query_vec:  # Prevent None or empty dicts
        return []

    similarities = []
    for doc in corpus:
        doc_vec = vectorizer.transform(doc)
        if not doc_vec:
            sim = 0.0
        else:
            sim = cosine_similarity(query_vec, doc_vec)
        similarities.append((doc, sim))

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]


def chunk_corpus(corpus, strategy="sentence", fixed_length=50, overlap_size=1):
    chunks = []
    for doc in corpus:
        if strategy == "sentence":
            sentences = sent_tokenize(doc)
            for i in range(0, len(sentences), max(1, len(sentences) - overlap_size)):
                group = sentences[i:i+overlap_size+1]
                if group:
                    chunks.append(" ".join(group))
        elif strategy == "fixed-length":
            words = doc.split()
            for i in range(0, len(words), fixed_length - overlap_size):
                group = words[i:i+fixed_length]
                if group:
                    chunks.append(" ".join(group))
        else:
            raise ValueError("Invalid strategy: 'sentence' or 'fixed-length'")
    return chunks

def limit_vocabulary(corpus, top_n):
    all_tokens = " ".join(corpus).split()
    top_words = set(word for word, _ in Counter(all_tokens).most_common(top_n))
    return [" ".join([word for word in doc.split() if word in top_words]) for doc in corpus]


class Vectorizer:
    """
    Unified interface for BoW, TF-IDF, or raw (None).
    """
    def __init__(self, method=None):
        self.method = method
        if method == "tfidf":
            self.model = TF_IDF()
        elif method == "bow":
            self.model = Bag_of_Words()
        elif method is None:
            self.model = None
        else:
            raise ValueError("Invalid method. Use 'tfidf', 'bow', or None.")

    def fit(self, corpus):
        if self.model is not None:
            self.model.fit(corpus)
        return self

    def transform(self, document):
        if self.model is not None:
            return self.model.transform(document)
        else:
            # Fallback to raw bag-of-words (word frequency dict)
            if isinstance(document, str):
                tokens = document.lower().split()
            elif isinstance(document, list):
                tokens = document
            else:
                return {}

            # Always return a dict even if empty
            return {token: 1.0 for token in tokens if token.strip()}


import os
import pandas as pd

def read_files_in_chunks(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            try:
                yield pd.read_csv(os.path.join(directory, filename), chunksize=10000)
            except Exception as e:
                print(f"Error reading {filename}: {e}")


def run_comparison():
    corpus = [
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
        "The Louvre is a museum that houses the Mona Lisa.",
        "French cuisine includes croissants, baguettes, and escargot.",
        "The Seine River flows through the heart of Paris.",
        "Notre-Dame is a famous Gothic cathedral in Paris."
    ]
    query = "What should I visit in Paris?"

    modes = ["none", "stem", "lemma"]
    chunk_sizes = [6]
    overlaps = [2]
    top_ns = [30, 20]

    for mode in modes:
        preprocessed = preprocess_corpus(corpus, mode)
        print(f"\n--- Preprocessing Mode: {mode} ---")

        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                chunked = chunk_corpus(preprocessed, chunk_size=chunk_size, overlap=overlap)
                print(f"\nChunk Size: {chunk_size}, Overlap: {overlap}, Chunks: {len(chunked)}")

                for top_n in top_ns:
                    vocab_limited = limit_vocabulary(chunked, top_n)
                    print(f"Top-{top_n} vocabulary limited corpus...")

                    for name, Vectorizer in [("BoW", Bag_of_Words), ("TF-IDF", TF_IDF)]:
                        vectorizer = Vectorizer().fit(vocab_limited)
                        top_docs = retrieve_top_k(query, vocab_limited, vectorizer, k=3)
                        prompt = "\n".join([f"[Doc {i+1}] {doc}" for i, (doc, _) in enumerate(top_docs)])
                        prompt += f"\n\nQuestion: {query}"

                        print(f"\n→ {name} | Top-{top_n} | Mode={mode} | Chunks={len(chunked)}")
                        print(f"Querying Mistral with:\n{prompt}")
                        try:
                            response = query_mistral(prompt)
                            print("Mistral Response:\n", response)
                        except Exception as e:
                            print("API Error:", e)


import glob
import logging
import psutil
from itertools import islice

# Configure logging
logging.basicConfig(
    filename="debug_file_loading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def batched(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def load_corpus_from_files(pattern="storage/*.txt.clean", batch_size=50):
    """
    Load and concatenate contents of all .txt.clean files into a list of documents.
    Logs memory usage and file progress for troubleshooting.
    """
    corpus = []
    filepaths = glob.glob(pattern)
    total_files = len(filepaths)
    logging.info(f"Found {total_files} files matching pattern {pattern}")

    for batch_num, batch in enumerate(batched(filepaths, batch_size), start=1):
        logging.info(f"Processing batch {batch_num}: {len(batch)} files")
        print(f"Batch {batch_num}/{(total_files // batch_size)+1}...")  # Heartbeat

        for i, filepath in enumerate(batch):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        corpus.append(text)
            except Exception as e:
                logging.error(f"Error reading {filepath}: {e}")

            # Log memory every 10 files
            if i % 10 == 0:
                mem = psutil.virtual_memory()
                logging.info(f"Used RAM: {mem.percent}% | Available: {mem.available / 1e6:.2f} MB")

        # Flush log buffer to ensure we don't lose logs if it crashes
        for handler in logging.getLogger().handlers:
            handler.flush()

    logging.info(f"Completed loading. Total documents: {len(corpus)}")
    return corpus



# ------------------------ Experiment Runner ------------------------
def run_analysis():
    

    
    
    corpus = load_corpus_from_files("storage/*.txt.clean")
    questions = [
        "What are the key themes discussed?"
    ]

    preprocessing_modes = ["none", "stem", "lemma"]
    chunk_configs = [
        ("sentence", None, 2),
        ("sentence", None, 4),
        ("fixed-length", 50, 1),
        ("fixed-length", 100, 1),
        ("fixed-length", 150, 1),
    ]
    vocab_sizes = [50, 30, 10]

    for mode in preprocessing_modes:
        print(f"\n=== Preprocessing: {mode} ===")
        preprocessed = preprocess_corpus(corpus, mode)

        for strategy, length, overlap in chunk_configs:
            print(f"\n--- Chunking: {strategy}, Fixed Length: {length}, Overlap: {overlap} ---")
            chunked = chunk_corpus(preprocessed, strategy=strategy, fixed_length=length or 50, overlap_size=overlap)

            for top_n in vocab_sizes:
                limited = limit_vocabulary(chunked, top_n)
                print(f"\n→ Vocabulary Size: Top-{top_n}, Chunks: {len(limited)}")

                for vec_label in [None, "bow", "tfidf"]:
                    vectorizer = Vectorizer(method=vec_label).fit(limited)

                    for query in questions:
                        top_docs = retrieve_top_k(query, limited, vectorizer)

                        print(f"\n[Vectorizer: {vec_label}] | Query: {query}")
                        for i, (doc, score) in enumerate(top_docs, 1):
                            print(f"{i}. (Score={score:.3f}): {doc}")

                        prompt = "\n".join([f"[Doc {i+1}] {doc}" for i, (doc, _) in enumerate(top_docs)])
                        prompt += f"\n\nQuestion: {query}"
                        try:
                            response = query_mistral(prompt)
                            print("Mistral Answer:\n", response)
                        except Exception as e:
                            print("Error querying Mistral:", e)
        results.append({
        "preprocessing_mode": mode,
        "chunking_strategy": strategy,
        "overlap_size": overlap,
        "fixed_length": length,
        "vocab_top_n": top_n,
        "vectorizer": vec_label,
        "query": query,
        "avg_similarity": sum(score for _, score in top_docs) / len(top_docs),
        "mistral_response_len": len(response.strip())
        })

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("rag_results.csv", index=False)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12,6))
    sns.barplot(data=df, x="vectorizer", y="avg_similarity", hue="preprocessing_mode")
    plt.title("Average Similarity vs. Vectorizer by Preprocessing")
    plt.ylabel("Average Cosine Similarity (Top-k)")
    plt.show()

    sns.lineplot(data=df, x="vocab_top_n", y="mistral_response_len", hue="vectorizer", style="preprocessing_mode")
    plt.title("Response Length vs. Vocabulary Size")
    plt.xlabel("Top-N Vocabulary")
    plt.ylabel("Response Length (chars)")
    plt.show()

    pivot = df.pivot_table(
    values="avg_similarity",
    index="chunking_strategy",
    columns="overlap_size",
    aggfunc="mean"
    )
    
    sns.heatmap(pivot, annot=True, cmap="YlGnBu")
    plt.title("Avg Similarity by Chunking Strategy and Overlap")
    plt.show()

    print(df.sort_values("avg_similarity", ascending=False).head(5).to_markdown())
