import math
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from gpt import GPTModel

# ---------------- dataset ----------------
class PackedDataset(Dataset):
    def __init__(self, npy_path: str):
        self.arr = np.load(npy_path, mmap_mode="r")
        if self.arr.ndim != 2 or self.arr.shape[1] < 2:
            raise ValueError("Expected shape (N, S+1).")
        self.S = self.arr.shape[1] - 1

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        row = self.arr[idx]
        x = torch.from_numpy(row[:-1].astype(np.int64, copy=False))
        y = torch.from_numpy(row[1:].astype(np.int64, copy=False))
        return x, y

# ---------------- sched ------------------
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            prog = float(stepnum) / float(max(1, warmup_steps))
            return 0.00001 + prog
        steps_after_peak = stepnum - warmup_steps
        tail_steps = max(1, total_steps - warmup_steps)
        prog = float(steps_after_peak) / float(tail_steps)
        return ((np.cos(np.pi * prog) + 1.0) * 0.5) * 0.9 + 0.1
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)

def train():
    # ---------- hyperparams I’m using ----------
    data_dir     = "./dataset"
    train_path   = f"{data_dir}/packed_seq257_train.npy"   # 256+1
    val_path     = f"{data_dir}/packed_seq257_val.npy"
    vocab_size   = 10_000
    max_seq_len  = 256

    d_model      = 384
    n_heads      = 6
    n_layers     = 6

    batch_size   = 16
    epochs       = 1            # quick pass; change as needed
    peak_lr      = 1e-4
    warmup_steps = 3000
    grad_clip    = 1.0
    log_every    = 1             # log EVERY step
    # ------------------------------------------

    # ---------- CUDA sanity check ----------
    print("=== CUDA sanity check ===")
    print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
        _ = torch.arange(10, device="cuda")
        _ = torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")
    print("=========================\n")

    # recommended fast-path toggles
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- data ----------
    train_ds = PackedDataset(train_path)
    val_ds   = PackedDataset(val_path)
    if train_ds.S != max_seq_len:
        raise ValueError(f"Sequence length mismatch: dataset S={train_ds.S}, model expects {max_seq_len}")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=0
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=0
    )

    # ---------- model ----------
    model = GPTModel(
        d_model=d_model,
        n_heads=n_heads,
        layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    ).to(device)

    # build GPU-side buffers once (positions, etc.)
    if hasattr(model, "prepare_buffers"):
        model.prepare_buffers(device)


    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)
    total_steps = epochs * len(train_dl)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps)
    criterion = torch.nn.CrossEntropyLoss()

    # I log EVERY step’s loss.
    losses_tokens = []
    tokens_seen = 0
    step = 0

    def evaluate(dl):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(xb)  # (B, S, V)
                    B, S, V = logits.shape
                    loss = criterion(logits.view(B * S, V), yb.view(B * S))
                total += float(loss.item())
                count += 1
        model.train()
        return total / max(1, count)

    model.train()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(xb)
                B, S, V = logits.shape
                loss = criterion(logits.view(B * S, V), yb.view(B * S))

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()

            step += 1
            tokens_seen += B * S

            # per-step logging (train loss, lr, grad-norm)
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"epoch {epoch} step {step} | tokens {tokens_seen} | "
                f"train {loss.item():.4f} | lr {lr_now:.6g} | gnorm {float(gnorm):.2f}"
            )
            losses_tokens.append((tokens_seen, float(loss.item())))

        # optional end-of-epoch validation
        val_loss = evaluate(val_dl)
        print(f"[epoch {epoch} done] val {val_loss:.4f} | elapsed {time.time() - t0:.1f}s")

    # ---------- plot (every step) ----------
    xs = [x for x, _ in losses_tokens]
    ys = [y for _, y in losses_tokens]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys)
    plt.xlabel("Tokens processed")
    plt.ylabel("Training loss (CrossEntropy)")
    plt.title("GPT Training: Loss vs Tokens (every step)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Saved loss curve to loss_curve.png")
    plt.show()

if __name__ == "__main__":
    train()
