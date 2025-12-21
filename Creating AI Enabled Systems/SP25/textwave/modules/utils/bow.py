import re
from collections import Counter

class Bag_of_Words:
    def __init__(self):
        self.vocabulary_ = {}

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents):
        unique_words = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            unique_words.update(tokens)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(unique_words))}
        return self

    def transform(self, document):
        tokens = self._tokenize(document)
        word_counts = Counter(tokens)
        bow_vector = {token: word_counts[token] for token in word_counts if token in self.vocabulary_}
        return bow_vector


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

    transform = Bag_of_Words()
    transform.fit(corpus)

    test_document = "The quick dog jumps high over the lazy fox."
    bow_test = transform.transform(test_document)

    print("Test Document Bag-of-Words:")
    for term, count in sorted(bow_test.items()):
        print(f"  {term}: {count}")
