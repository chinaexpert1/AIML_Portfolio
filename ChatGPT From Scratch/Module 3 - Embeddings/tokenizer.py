
import json
import re
from typing import List, Dict, Tuple

class Tokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer implemented with only Python stdlib.

    Constructor expects file paths to:
      - vocab.txt : one token per line; the line index is its id
      - merges.json : list of ["token1", "token2"] pairs in merge order
    """

    def __init__(self, vocab_file: str, merges_file: str):
        # Load vocab (token -> id) and inverse (id -> token)
        with open(vocab_file, "r", encoding="utf-8") as vf:
            self.id_to_token = [line.rstrip("\n") for line in vf]
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

        # Load merges as ordered ranks: lower rank = earlier merge
        with open(merges_file, "r", encoding="utf-8") as mf:
            merges = json.load(mf)
        # Normalize to tuples and rank
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            (a, b): i for i, (a, b) in enumerate(map(tuple, merges))
        }

        # Precompile regex that splits into "runs of non-space" OR "runs of space"
        self._splitter = re.compile(r"\S+|\s+")

    # --------- Helper utilities for BPE on a single token-with-leading-space ---------

    @staticmethod
    def _get_pairs(symbols: Tuple[str, ...]):
        """Return set of adjacent symbol pairs from a tuple of symbols."""
        return {(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)}

    def _bpe(self, token: str) -> List[str]:
        """
        Apply BPE to a single token *that already includes its leading space if any*.
        We follow the standard greedy procedure: repeatedly merge the lowest-rank pair.
        """
        # Start from characters
        if not token:
            return []
        symbols = tuple(token)  # tuple of single-character strings
        pairs = self._get_pairs(symbols)
        if not pairs:
            return [token]

        while True:
            # Select best-ranked pair among existing pairs
            min_rank = None
            best_pair = None
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is not None and (min_rank is None or r < min_rank):
                    min_rank = r
                    best_pair = p

            if best_pair is None:
                break  # no more merges apply

            first, second = best_pair
            new_symbols = []
            i = 0
            L = len(symbols)
            while i < L:
                # Find next occurrence of (first, second)
                try:
                    j = symbols.index(first, i)
                except ValueError:
                    # append the remainder
                    new_symbols.extend(symbols[i:])
                    break

                # Append everything up to j
                new_symbols.extend(symbols[i:j])
                # If a pair match at j, merge it, else append single symbol
                if j < L - 1 and symbols[j] == first and symbols[j+1] == second:
                    new_symbols.append(first + second)
                    i = j + 2
                else:
                    new_symbols.append(symbols[j])
                    i = j + 1

            symbols = tuple(new_symbols)
            if len(symbols) == 1:
                break
            pairs = self._get_pairs(symbols)

        return list(symbols)

    # ------------------------------- Public API -------------------------------

    def encode(self, string: str) -> List[int]:
        """
        Encode a string into a list of token ids.
        Uses GPT-style 'space-prefix' convention: each non-space word is prefixed with a space.
        """
        ids: List[int] = []
        for chunk in self._splitter.findall(string):
            if chunk.isspace():
                # Keep spaces as-is; merge will handle if multi-space exists in vocab
                sub_tokens = self._bpe(chunk)
            else:
                # Prefix a single space so that words align with space-prefixed vocab items
                sub_tokens = self._bpe(" " + chunk if not chunk.startswith(" ") else chunk)
            for tok in sub_tokens:
                if tok not in self.token_to_id:
                    raise KeyError(f"Token '{tok}' not found in vocabulary.")
                ids.append(self.token_to_id[tok])
        return ids

    def decode(self, list_of_integers: List[int]) -> str:
        """
        Decode a sequence of token ids back into a string by simple concatenation of token strings.
        """
        out_tokens = []
        for i in list_of_integers:
            if not (0 <= i < len(self.id_to_token)):
                raise IndexError(f"Token id {i} is out of range.")
            out_tokens.append(self.id_to_token[i])
        # Tokens already carry their leading spaces where needed
        return "".join(out_tokens)



