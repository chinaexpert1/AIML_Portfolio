# construct_dataset.py
import os
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hftokenizer import HFTokenizer

def construct_dataset(
    data_txt_file: str,
    sequence_length: int = 256,
    out_dir: str = "./dataset",
    val_frac: float = 0.05,
    seed: int = 1337,
):
    """
    convert UTF-8 text (one sample per line) into packed token-id sequences.

    Output shape: (num_sequences, sequence_length+1)
      - The +1 lets me compute next-token targets by shifting right by one.
    append an <|endoftext|> to every sample before packing.
    shuffle rows, then split into train/val and save as .npy files.

    Files written:
      {out_dir}/packed_seq{sequence_length+1}_train.npy
      {out_dir}/packed_seq{sequence_length+1}_val.npy
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # tokenizer (already trained/saved by hftokenizer.py)
    tok = HFTokenizer()
    tok.load("./hftokenizer")
    eos_id = tok.tokenizer.eos_token_id

    # read UTF-8 lines
    with open(data_txt_file, "r", encoding="utf-8", errors="strict") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # tokenize + append EOS, then flatten into one long stream
    ids_stream = []
    for ln in tqdm(lines, desc="Tokenizing"):
        ids = tok.encode(ln)
        ids.append(eos_id)
        ids_stream.extend(ids)

    ids_np = np.asarray(ids_stream, dtype=np.int32)

    # pack into chunks of length S+1
    pack_len = sequence_length + 1
    n_full = (ids_np.shape[0] // pack_len) * pack_len
    if n_full == 0:
        raise RuntimeError("Not enough tokens to form even one packed sequence.")
    ids_np = ids_np[:n_full]
    packed = ids_np.reshape(-1, pack_len)

    # shuffle rows deterministically
    rng = np.random.RandomState(seed)
    perm = rng.permutation(packed.shape[0])
    packed = packed[perm]

    # train/val split
    n_val = max(1, int(math.floor(packed.shape[0] * val_frac)))
    val = packed[:n_val]
    train = packed[n_val:]

    # save
    train_file = out_path / f"packed_seq{pack_len}_train.npy"
    val_file = out_path / f"packed_seq{pack_len}_val.npy"
    np.save(train_file, train)
    np.save(val_file, val)

    print(f"Saved train: {train.shape} -> {train_file.resolve()}")
    print(f"Saved  val : {val.shape}   -> {val_file.resolve()}")

if __name__ == "__main__":
    # I keep defaults reasonable for a first pass.
    construct_dataset("./data.txt", sequence_length=256)
