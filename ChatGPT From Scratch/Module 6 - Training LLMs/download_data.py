# download_data.py
"""
Downloads WikiText-103 and writes a single UTF-8 text file (one cleaned line per example).
Requires: pip install datasets
"""

from datasets import load_dataset
from pathlib import Path

def clean(text: str) -> str:
    # collapse newlines/whitespace; keep a single space between tokens
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def main():
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1")["train"]

        out_path = Path("data.txt")
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for entry in ds:
                t = clean(entry["text"])
                if t:  # skip empty lines
                    f.write(t + "\n")

        print(f"Wrote {out_path.resolve()} (UTF-8).")
    except Exception as e:
        # minimal, explicit error reporting (your preference)
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    main()
