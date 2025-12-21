import math
import re
from collections import Counter, defaultdict

class TF_IDF:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents):
        df = defaultdict(int)
        total_documents = len(documents)

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1

        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(df))}
        self.idf_ = {
            word: math.log(total_documents / (df[word] + 1)) + 1
            for word in self.vocabulary_
        }
        return self

    def transform(self, document):
        tokens = self._tokenize(document)
        total_tokens = len(tokens)
        tf_counts = Counter(tokens)

        tfidf_vector = {}
        for token in tf_counts:
            if token in self.vocabulary_:
                tf = tf_counts[token] / total_tokens
                idf = self.idf_.get(token, 0.0)
                tfidf_vector[token] = tf * idf
        return tfidf_vector


if __name__ == "__main__":
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ]

    transformer = TF_IDF()
    transformer.fit(corpus)

    test_document = "The quick dog jumps high over the lazy fox."
    tfidf_test = transformer.transform(test_document)

    print("Test Document TF-IDF:")
    for term, score in sorted(tfidf_test.items()):
        print(f"  {term}: {score:.4f}")



if __name__ == "__main__":
    # Example corpus of 9 documents to train the TF-IDF transformer.
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ]

    # Fit the transformer on the corpus.
    transformer = TF_IDF()
    transformer.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    tfidf_test = transformer.transform(test_document)
    
    # Display the TF-IDF representation of the test document.
    print("Test Document TF-IDF:")
    for term, score in sorted(tfidf_test.items()):
        print(f"  {term}: {score:.4f}")
