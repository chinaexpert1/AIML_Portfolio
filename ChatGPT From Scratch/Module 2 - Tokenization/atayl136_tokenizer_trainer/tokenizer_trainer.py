import json
from collections import Counter
from typing import List, Tuple

# Uses the provided helpers (saving.py)
from saving import save_vocab, save_merges

"""
Implements a classic Byte Pair Encoding (BPE) trainer with GPT-style "space-prefixed" words.
"""

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _load_text(txt_file: str) -> str:
    """
    Load raw text from a file. For .txt this simply returns the file content.
    If users point this at a .jsonl file, we make a best-effort attempt:
    - If a line parses as JSON and has a 'text' field, use it.
    - Otherwise, treat the raw line as text.
    In all cases, newlines are normalized to spaces so we can split on ' '.
    """
    raw = []
    if txt_file.lower().endswith(".jsonl"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "text" in obj:
                        raw.append(str(obj["text"]))
                    else:
                        raw.append(line)
                except Exception:
                    raw.append(line)
        text = " ".join(raw)
    else:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return text

def _pretokenize_space_prefixed(text: str) -> List[List[str]]:
    """
    Split on literal spaces, discard empty pieces, and represent each word as:
      [' '] + list(word_chars)
    """
    words = [w for w in text.split(" ") if w != ""]
    corpus = []
    for w in words:
        corpus.append([' '] + list(w))
    return corpus

from collections import Counter

def _count_adjacent_pairs(corpus):
    # corpus: List[List[str]] where each inner list is a word's token list
    pair_counts = Counter()

    # Collapse identical words to (word_tokens_tuple -> word_count)
    word_freq = Counter(tuple(tokens) for tokens in corpus)

    # For each unique word, add each adjacent pair weighted by its word_count
    for tokens, freq in word_freq.items():
        for a, b in zip(tokens, tokens[1:]):
            pair_counts[(a, b)] += freq

    return pair_counts


def _apply_merge(corpus: List[List[str]], left: str, right: str) -> List[List[str]]:
    """
    Apply a single merge (left, right) across the corpus.
    Replacement is greedy left-to-right with non-overlapping merges.
    """
    merged_symbol = left + right
    new_corpus = []
    for tokens in corpus:
        i = 0
        out = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                out.append(merged_symbol)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        new_corpus.append(out)
    return new_corpus


def train_tokenizer(txt_file: str, vocab_size: int, base_vocabulary: List[str]):
    '''
    param : txt_file        - path to text or jsonl data (e.g., "./data.txt" or "./train.jsonl")
    param : vocab_size      - desired final vocabulary size (>= number of base chars)
    param : base_vocabulary - list of single-character strings (must include ' ')

    saves:
      ./vocab.txt   : final vocabulary in order (one entry per line)
      ./merges.json : list of (left, right) merges in the order they were learned
    '''

    # ---- Load & pretokenize -------------------------------------------------
    text = _load_text(txt_file)
    corpus = _pretokenize_space_prefixed(text)

    # ---- Initialize vocabulary ----------------------------------------------
    vocab: List[str] = _dedupe_preserve_order(list(base_vocabulary))
    vocab_set = set(vocab)

    observed_chars = set()
    for tokens in corpus:
        for t in tokens:
            if len(t) == 1:
                observed_chars.add(t)
    for ch in sorted(observed_chars):
        if ch not in vocab_set:
            vocab.append(ch)
            vocab_set.add(ch)

    # ---- Learn merges until we hit vocab_size or run out of pairs -----------
    merges: List[Tuple[str, str]] = []

    if vocab_size <= len(vocab):
        save_vocab(vocab, "./vocab.txt")
        save_merges(merges, "./merges.json")
        return

    while len(vocab) < vocab_size:
        pair_counts = _count_adjacent_pairs(corpus)
        if not pair_counts:
            break

        max_freq = max(pair_counts.values())
        tied = [pair for pair, c in pair_counts.items() if c == max_freq]

        # Tie-break: use Python's default lexicographic tuple ordering via list.sort()
        tied.sort()
        best_left, best_right = tied[0]


        new_symbol = best_left + best_right
        merges.append((best_left, best_right))

        if new_symbol not in vocab_set:
            vocab.append(new_symbol)
            vocab_set.add(new_symbol)

        corpus = _apply_merge(corpus, best_left, best_right)

        if len(vocab) >= vocab_size:
            break

    # ---- Save results --------------------------------------------------------
    save_vocab(vocab, "./vocab.txt")
    save_merges(merges, "./merges.json")

if __name__ == "__main__":
    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'
    train_tokenizer("./data.txt", len(base) + 1000, [c for c in base])
