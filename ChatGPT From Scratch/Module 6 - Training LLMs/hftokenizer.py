# hftokenizer.py
"""
Train a byte-level BPE tokenizer from a UTF-8 text file using Hugging Face.
Writes artifacts to ./hftokenizer/.
"""

from pathlib import Path
from transformers import AutoTokenizer

class HFTokenizer:
    def __init__(self):
        # start from GPT-2's byte-level tokenizer recipe
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # keep a clear EOS token string in the trained tokenizer files
        self.tokenizer.eos_token = "<|endoftext|>"

    def _line_iterator(self, datafile: str):
        # Read as UTF-8 to avoid cp1252 decode errors on Windows
        with open(datafile, "r", encoding="utf-8", errors="strict") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    def train(self, datafile: str, vocab_size: int = 10_000, limit_alphabet: int = 500):
        it = self._line_iterator(datafile)
        # train_new_from_iterator streams; no need to materialize .readlines()
        self.tokenizer = self.tokenizer.train_new_from_iterator(
            it,
            vocab_size=vocab_size,
            limit_alphabet=limit_alphabet,
        )
        outdir = Path("./hftokenizer")
        outdir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(outdir))
        print(f"Tokenizer trained and saved to {outdir.resolve()}")

    def load(self, path: str = "./hftokenizer/"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    # string to token_ids
    def encode(self, text: str):
        return self.tokenizer(text)["input_ids"]

    # token_ids to string
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

if __name__ == "__main__":
    tok = HFTokenizer()
    tok.train("./data.txt")
