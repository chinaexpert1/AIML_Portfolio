
# -*- coding: utf-8 -*-
# train_barrier_gpt_nexttoken12_Vanilla_prog_learning.py
"""
Sequential Fine-tuning Transformer for Futures Trading
+ Experiment Manager (scaling, context-length stress, ablations)

Adds (without removing existing functionality):
- --experiment {none,scaling,context_stress,ablations}
- --token-budget N (tokens) → early-stop when tokens_seen >= N
- Experiment subfolders & run-level summary CSVs
- Context-length stress eval at arbitrary lengths
- Ablations:
    * -ALiBi → learned absolute positions
    * -GQA → standard Multi-Head Attention (MHA)
    * -RMSNorm/Pre-LN → LayerNorm/Post-LN
    * -SwiGLU → ReLU MLP
    * -LabelSmoothing → ε=0.0

Key behavior (true next-token predictor):
- Next-token predictions CAN use the ground truth bar_label on PAST bars (tokens) in the context only;
  bar_label is treated as an input feature for past tokens.
- The target is always the NEXT bar’s bar_label, i.e., context window [t-L, …, t-1] → predict bar_label at t.
- The first K = context_len bars are discarded from the dataset AFTER loading (and scaling upstream),
  as a warm-up period so EMA-like scalers are well-formed.
- Sequential training continues across seven datasets (contracts) with shared weights.
- We are predicting the future behavior of the market with a causally-masked transformer.
- For validation, we use val_stride=1 so that (within the usable tail) we produce a prediction for
  every possible next-bar after a full context window.

Custom Logging:
        Save per-sample validation predictions:
        - bar_label_X: ground truth class
        - bar_label_y: predicted class
        - prob_*: class probabilities

"""

import math, os, sys, glob, argparse, warnings, json, zipfile, re, gc, copy, random, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from contextlib import nullcontext, contextmanager, ExitStack

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from tqdm import tqdm
import pyarrow.parquet as pq

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    f1_score, recall_score, precision_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =========================
# VERSIONING HELPER
# =========================

def _next_version_for_size(size_tag: str, context_len: int, runs_dir: Path = Path("runs")) -> str:
    """
    Computes next version number vN for a given size tag.
    Works for normal sizes (S, M, L, Vanilla) and for SCALING.

    Expected directory names:
        v3_SCALING_512_stride_8_scaling_20251130_101234
        v1_S_512_stride_32_none_20251130_090001
    """
    if not runs_dir.exists():
        return "v1"

    # Pattern matches:
    # v<digits>_<SIZE_TAG>_<CONTEXT>_stride_<digits>_<experiment>_<YYYYMMDD>_<HHMMSS>
    pat = re.compile(
        rf"^v(\d+)_({re.escape(size_tag)})_{re.escape(str(context_len))}_stride_"
        r"[0-9]+_[a-zA-Z0-9]+_[0-9]{8}_[0-9]{6}$"
    )

    max_v = 0
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            max_v = max(max_v, int(m.group(1)))

    return f"v{max_v + 1 if max_v >= 1 else 1}"


# =========================
# ENV / CUDA SETUP
# =========================

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import bitsandbytes as bnb
    HAS_8BIT = True
except ImportError:
    HAS_8BIT = False
    print("[WARN] bitsandbytes not found; standard AdamW will be used.")

# =========================
# COLUMNS
# =========================


BAR_DESIRED_COLUMN_ORDER = [
    "microprice_last",
    "bar_steps",
    "price_trend_long",
    "mean_trade_size",
    "midpoint_change",
    "bar_pace",
    "microburst_count",
    "return_per_second",
    "lull_count",
    "momentum_5",
    "bar_change",
    "slow_period_signed_volume",
    "return_reversal",
    "high_low_spread_change",

]

DROP_COL = [


]


# =========================
# GPU HELPERS
# =========================

def setup_gpu_memory(memory_fraction=0.95, device_id=0):
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available; using CPU.")
        return torch.device('cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    device = torch.device(f'cuda:{device_id}')
    gpu_name = torch.cuda.get_device_name(device_id)
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    print(f"\n[GPU] Device: {gpu_name}")
    print(f"[GPU] Total VRAM: {total_memory:.2f} GB | Alloc fraction: {memory_fraction*100:.1f}%")
    try:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)
        print("[GPU] Memory fraction set OK")
    except Exception as e:
        print(f"[GPU] set_per_process_memory_fraction failed: {e}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    return device

def print_memory_usage(device):
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# =========================
# SDPA / Flash kernels
# =========================

def prefer_sdpa_kernels(enable_flash=True, enable_mem_efficient=True, enable_math=False):
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch 2.x
        return sdpa_kernel(enable_flash=enable_flash,
                           enable_mem_efficient=enable_mem_efficient,
                           enable_math=enable_math)
    except Exception:
        class _Noop:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        return _Noop()

# =========================
# ALiBi
# =========================

def _alibi_slopes(n_heads: int) -> List[float]:
    def get_slopes(n):
        def power_of_two_slopes(n_):
            start = 2**(-2**-(math.log2(n_) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n_)]
        if math.log2(n).is_integer():
            return power_of_two_slopes(n)
        else:
            cp2 = 2 ** math.floor(math.log2(n))
            base = power_of_two_slopes(cp2)
            extra = get_slopes(2 * cp2)[0::2]
            return base + extra[:n - cp2]
    return get_slopes(n_heads)

# =========================
# Norms, FFNs, Attention
# =========================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class ReLUMlp(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        hidden = int(mult * d)
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        hidden = int(mult * d)
        self.w1 = nn.Linear(d, hidden * 2, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        gate, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x2)

def _make_alibi_bias(n_heads, seq_len, device, slopes_tensor):
    pos = torch.arange(seq_len, device=device)
    dist = torch.abs(pos[:, None] - pos[None, :]).float()
    return -slopes_tensor * dist  # [1,nH,seq,seq]

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, head_dim=64, kv_groups=2, dropout=0.2, use_alibi=True):
        super().__init__()
        assert n_heads % kv_groups == 0, "n_heads must be divisible by kv_groups"
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.kv_groups, self.use_alibi = kv_groups, use_alibi
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_groups * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_groups * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if self.use_alibi:
            slopes = _alibi_slopes(n_heads)
            self.register_buffer(
                "alibi_slopes",
                torch.tensor(slopes, dtype=torch.float32).view(1, n_heads, 1, 1),
            )
        else:
            # Keep a buffer name for state_dict compatibility, but no content
            self.register_buffer("alibi_slopes", None, persistent=False)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)     # B,H,L,D
        K = self.k_proj(x).view(B, L, self.kv_groups, self.head_dim).transpose(1, 2)   # B,G,L,D
        V = self.v_proj(x).view(B, L, self.kv_groups, self.head_dim).transpose(1, 2)   # B,G,L,D
        n_queries_per_kv = self.n_heads // self.kv_groups
        K = K.repeat_interleave(n_queries_per_kv, dim=1)  # B,H,L,D
        V = V.repeat_interleave(n_queries_per_kv, dim=1)  # B,H,L,D
        bias = None
        if self.use_alibi:
            bias = _make_alibi_bias(self.n_heads, L, x.device, self.alibi_slopes)

        with prefer_sdpa_kernels(True, True):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)

class MultiHeadAttentionStd(nn.Module):
    """Standard MHA (for -GQA ablation)."""
    def __init__(self, d_model, n_heads=8, head_dim=None, dropout=0.2, use_alibi=True):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.use_alibi = use_alibi
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        slopes = _alibi_slopes(n_heads)
        self.register_buffer('alibi_slopes', torch.tensor(slopes).view(1, n_heads, 1, 1))

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        bias = None
        if self.use_alibi:
            bias = _make_alibi_bias(self.n_heads, L, x.device, self.alibi_slopes)
        with prefer_sdpa_kernels(True, True):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)

# =========================
# Transformer Blocks
# =========================

class TransformerBlock(nn.Module):
    """
    Supports Pre-LN (RMSNorm) and Post-LN (LayerNorm) to enable ablation.
    Also switches GQA ↔ MHA, and SwiGLU ↔ ReLU MLP.
    """
    def __init__(self, d_model, n_heads, head_dim, kv_groups,
                 ffn_mult=4, dropout=0.2,
                 use_gqa=True, use_alibi=True,
                 norm_style="rms_pre", ffn_style="swiglu"):
        super().__init__()
        self.norm_style = norm_style
        if norm_style == "rms_pre":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        elif norm_style == "layer_post":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            raise ValueError("norm_style must be 'rms_pre' or 'layer_post'")

        if use_gqa:
            self.attn = GroupedQueryAttention(
                d_model, n_heads, head_dim, kv_groups, dropout, use_alibi=use_alibi
            )
        else:
            self.attn = MultiHeadAttentionStd(
                d_model, n_heads, head_dim, dropout, use_alibi=use_alibi
            )

        if ffn_style == "swiglu":
            self.ffn = SwiGLU(d_model, mult=ffn_mult)
        else:
            self.ffn = ReLUMlp(d_model, mult=ffn_mult)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.norm_style == "rms_pre":
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x
        else:
            y = self.attn(x)
            x = self.norm1(x + self.dropout(y))
            y = self.ffn(x)
            x = self.norm2(x + self.dropout(y))
            return x

# =========================
# BarrierGPTs
# =========================

class BarrierGPT(nn.Module):
    """
    Backward-compatible baseline with flexible positional encoding:

    Modes:
      1) ALiBi-only:
           use_alibi=True,  use_learned_pos=False
      2) GPT-2 learned-only:
           use_alibi=False, use_learned_pos=True
      3) Sinusoidal-only:
           use_alibi=False, use_learned_pos=False
      4) ALiBi + learned positions:
           use_alibi=True,  use_learned_pos=True
    """
    def __init__(self, n_features, max_seq_len=1024, n_classes=3,
                 d_model=512, n_layers=12, n_heads=8, kv_groups=2, ffn_mult=4, dropout=0.2, use_alibi=True, use_gqa=True, norm_style="rms_pre", ffn_style="swiglu", use_learned_pos=None):
                
        super().__init__()
        self.d_model, self.n_layers, self.max_seq_len = d_model, n_layers, max_seq_len
        self.use_alibi = use_alibi
        self.input_proj = nn.Linear(n_features, d_model)


        # ================================
        # DEFAULTS FOR POSITIONAL MODES
        # ================================
        # If user doesn't specify, we ALWAYS use learned positions:
        #   - use_alibi=True  -> ALiBi + learned (dual)
        #   - use_alibi=False -> GPT-2 learned-only
        if use_learned_pos is None:
            self.use_learned_pos = True
        else:
            self.use_learned_pos = bool(use_learned_pos)

        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding choices
        if self.use_learned_pos:
            # GPT-2 style: learned absolute positions
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.register_buffer("pos_idx", torch.arange(max_seq_len), persistent=False)
            self.pos_encoding = None
        elif not self.use_alibi:
            # No ALiBi, no learned pos -> sinusoidal baseline
            self.pos_emb = None
            self.register_buffer("pos_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        else:
            # ALiBi-only: no explicit positional embeddings
            self.pos_emb = None
            self.register_buffer("pos_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.pos_encoding = nn.Identity()

        head_dim = d_model // n_heads

        # ===== wire ablation flags into the blocks =====
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                kv_groups=kv_groups,
                ffn_mult=ffn_mult,
                dropout=dropout,
                use_gqa=use_gqa,
                use_alibi=use_alibi,
                norm_style=norm_style,
                ffn_style=ffn_style,
            )
            for _ in range(n_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self._init_weights()



    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_all_logits=False):
        # x: (B, S, n_features)
        B, S, _ = x.shape

        h = self.input_proj(x)  # (B, S, D)

        # positions
        if self.use_learned_pos and self.pos_emb is not None:
            if self.pos_idx.device != h.device:
                self.pos_idx = self.pos_idx.to(h.device)
            pos = self.pos_idx[:S].unsqueeze(0).expand(B, -1)
            h = h + self.pos_emb(pos)
        elif self.pos_encoding is not None:
            h = self.pos_encoding(h)

        for blk in self.blocks:
            h = blk(h)

        h = self.norm_out(h)
        logits = self.classifier(h)
        return logits if return_all_logits else logits[:, -1, :]

class BarrierGPTConfigurable(nn.Module):
    """
    Same interface, but supports ablations via flags.
    """
    def __init__(self, n_features, max_seq_len=1024, n_classes=3,
                 d_model=512, n_layers=12, n_heads=8, kv_groups=2, ffn_mult=4, dropout=0.2,
                 use_gqa=True, use_alibi=True, norm_style="rms_pre", ffn_style="swiglu"):
        super().__init__()
        self.d_model, self.n_layers, self.max_seq_len = d_model, n_layers, max_seq_len
        self.input_proj = nn.Linear(n_features, d_model)
        head_dim = d_model // n_heads
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, head_dim, kv_groups, ffn_mult, dropout,
                use_gqa=use_gqa, use_alibi=use_alibi,
                norm_style=norm_style, ffn_style=ffn_style
            )
            for _ in range(n_layers)
        ])
        self.norm_style = norm_style
        self.norm_out = RMSNorm(d_model) if norm_style == "rms_pre" else nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_all_logits=False):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        logits = self.classifier(x)
        return logits if return_all_logits else logits[:, -1, :]

# =========================
# Dataset (NEXT-TOKEN)
# =========================
class BarDataset(Dataset):
    """
    Next-token style dataset:

    Given context_len bars as context, predict the *next* bar's label.

    For each sample (bar index t is the target):
        - Context window: bars [t-L, ..., t-1]  (length = context_len)
        - X = ALL features for these bars, EXCLUDING bar_label
        - y = bar_label at bar t (the next bar after the context window), as class {0,1,2}.

    The first K = context_len rows of the file are discarded as a warm-up zone after loading,
    so that upstream EMA/scale state has at least a full context_len of history.
    """
    def __init__(self, parquet_path: str, context_len: int = 1024, target_col: str = "bar_labelH10", stride: int = None,
                 is_validation: bool = False, val_size: int = 3072, bar_label_scale: float = 1.0):
        super().__init__()
        self.context_len = context_len
        self.stride = stride if stride is not None else (context_len // 4)
        self.is_validation = is_validation
        self.val_size = val_size
        self.bar_label_scale = float(bar_label_scale)  # NOT USED ANYMORE but kept for compatibility

        print(f"[Dataset] Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        self.target = target_col  # passed into BarDataset from main()
        target = self.target

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in {parquet_path}")

        raw_labels = df[target].values

        # If target is continuous (like fwd_ret_h10), convert to classes:
        if target == 'fwd_ret_h10':
            # Simple 3-way sign classification (you may choose another)
            raw_labels = np.where(raw_labels > 0, 2,
                        np.where(raw_labels < 0, 0, 1))
        else:
            raw_labels = np.round(raw_labels).astype(np.int64)

        self.n_classes = int(raw_labels.max()) + 1  # usually 3
        # ============ END EXTRACT ============

        # Drop unused columns, enforce feature order
        df = df.drop(columns=[c for c in DROP_COL if c in df.columns], errors='ignore')
        

        # ============================================================
        # DROP ALL POSSIBLE TARGET COLUMNS — NONE should be features
        # ============================================================
        ALL_TARGETS = {"bar_label", "bar_labelH10", "fwd_ret_h10"}

        # remove ALL target columns (not only the selected target)
        cols_to_drop = [c for c in ALL_TARGETS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"[FEATURE FILTER] Dropped target columns from features: {cols_to_drop}")


        # ============ END DROP ============
        
        # Keep only columns in desired order (bar_label already removed)
        available_cols = [c for c in BAR_DESIRED_COLUMN_ORDER if c in df.columns]
        df = df[available_cols]

        print(f"[BarDataset] df.shape = {df.shape}")
        print(f"[BarDataset] Using {len(available_cols)} feature columns")
        # Warm-up: discard first K = context_len bars AFTER upstream scaling
        warmup = self.context_len
        if len(df) <= warmup + self.val_size + self.context_len:
            raise ValueError(
                f"[Dataset] Not enough rows in {parquet_path} after warmup and val_size.\n"
                f"Rows={len(df)}, warmup={warmup}, val_size={self.val_size}, context_len={self.context_len}"
            )
        
        # ============ APPLY WARMUP TO BOTH DATA AND LABELS ============
        df = df.iloc[warmup:].reset_index(drop=True)
        raw_labels = raw_labels[warmup:]  # Keep labels aligned
        # ============ END WARMUP ============

        # Split into train/val tails
        if is_validation:
            df = df.tail(self.val_size).reset_index(drop=True)
            raw_labels = raw_labels[-self.val_size:]
            print(f"[Dataset] Validation set: last {self.val_size} bars (after warmup)")
        else:
            df = df.head(len(df) - self.val_size).reset_index(drop=True)
            raw_labels = raw_labels[:len(df)]
            print(f"[Dataset] Training set: {len(df)} bars (excluding last {self.val_size}, after warmup)")

        # Store labels and features
        self.labels = raw_labels
        self.data = df.values.astype(np.float32)
        self.n_features = self.data.shape[1]

        # For next-token: we need context_len bars *plus* 1 target bar
        max_start = len(self.data) - (self.context_len + 1)
        if max_start < 0:
            raise ValueError(
                f"Dataset too small for next-token: {len(self.data)} "
                f"< context_len+1 ({self.context_len + 1})"
            )

        self.n_windows = (max_start // self.stride) + 1
        print(f"[Dataset] Created {self.n_windows} next-token windows (stride={self.stride}) | Features={self.n_features}")

    def effective_unique_bars(self):
        # Bars used by first window: [0, ..., context_len]
        # Each subsequent window shifts by stride
        if self.n_windows <= 0:
            return 0
        return self.context_len + (self.n_windows - 1) * self.stride

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # Start index for this window
        s = idx * self.stride
        # Context spans [s, ..., s+context_len-1]
        e = s + self.context_len

        # Context features: shape [context_len, D] (bar_label NOT included)
        window_feats = self.data[s:e]          # [L, D]
        # Target label uses the integer class at bar e
        y = int(self.labels[e])                # {0,1,2}

        X = torch.from_numpy(window_feats).float()
        y = torch.tensor(y, dtype=torch.long)
        return X, y


# =========================
# Focal Loss
# =========================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma, self.weight, self.reduction = gamma, weight, reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction=self.reduction)
        p = torch.exp(-ce)
        fl = ((1 - p) ** self.gamma) * ce
        if self.reduction == 'mean': return fl.mean()
        if self.reduction == 'sum': return fl.sum()
        return fl

# =========================
# Trainer
# =========================

class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-3, weight_decay=0.01, epochs=15, use_8bit_adam=True,
                 gradient_accumulation_steps=1, label_smoothing=0.1,
                 early_stop_patience=2, early_stop_min_delta=1e-3,
                 amp_dtype: Optional[torch.dtype]=torch.bfloat16, grad_clip=1.0,
                 class_weights: Optional[torch.Tensor]=None, use_focal_loss=False, focal_gamma=2.0,
                 lr_schedule='cosine', warmup_epochs=1,
                 use_class_weights: bool = True,   
                 token_budget: Optional[int]=None, context_len_for_tokens: Optional[int]=None,
                 use_fbeta=True, beta=0.5, confidence_threshold=0.0):
        self.model, self.train_loader, self.val_loader, self.device = model, train_loader, val_loader, device
        self.epochs, self.gradient_accumulation_steps = epochs, gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.amp_dtype = amp_dtype
        self.early_stop_patience, self.early_stop_min_delta = early_stop_patience, early_stop_min_delta
        self.lr_schedule, self.warmup_epochs = lr_schedule, warmup_epochs
        self.token_budget = token_budget
        self.context_len_for_tokens = context_len_for_tokens
        self.use_class_weights =  use_class_weights
        # ============ ADD THESE NEW ATTRIBUTES ============
        self.use_fbeta = use_fbeta
        self.beta = beta
        self.confidence_threshold = confidence_threshold
        self.best_val_metric = 0.0  # Now tracks F-beta if use_fbeta=True, else accuracy
        
        self.no_improvement_epochs = 0
        # ==========================
        # CLASS WEIGHTS
        # ==========================
        use_class_weights = self.use_class_weights

        # If train_loader and val_loader are literally the same object,
        # we're probably in an eval-only / held-out run (often balanced).
        # In that case, DO NOT compute class weights from this dataset.
        if self.train_loader is self.val_loader:
            if use_class_weights:
                print("[Trainer] Detected train_loader == val_loader "
                      "(likely eval-only or held-out). Disabling class weighting.")
            use_class_weights = False

        if not use_class_weights:
            class_weights = None
            print("[Trainer] Class weighting DISABLED (uniform loss).")
        else:
            # Always compute fresh class weights from *training* data
            print("[Trainer] Computing class weights from training data...")

            # Get all labels efficiently
            if hasattr(self.train_loader.dataset, "labels"):
                all_labels = self.train_loader.dataset.labels
            else:
                all_labels = []
                for i in range(len(self.train_loader.dataset)):
                    _, y = self.train_loader.dataset[i]
                    all_labels.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))

            counts = np.bincount(all_labels, minlength=3)
            n_samples = len(all_labels)

            print(f"[Trainer] Class distribution:")
            print(f"  Down (0):    {counts[0]:>6} ({100*counts[0]/n_samples:.1f}%)")
            print(f"  Neutral (1): {counts[1]:>6} ({100*counts[1]/n_samples:.1f}%)")
            print(f"  Up (2):      {counts[2]:>6} ({100*counts[2]/n_samples:.1f}%)")

            alpha = 1.0
            n_classes = 3
            weights = np.array(
                [n_samples / (n_classes * count) if count > 0 else 0.0 for count in counts],
                dtype=np.float32
            )
            if alpha != 1.0:
                weights = weights ** alpha

            class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

            print(f"[Trainer] Class weights (α={alpha:.1f}): {class_weights.cpu().numpy()}")
            if counts[0] > 0 and counts[1] > 0:
                print(f"  Interpretation: Neutral errors are {weights[1]/weights[0]:.2f}x "
                      f"more penalized than Down errors")

        # Initialize loss function
        if use_focal_loss:
            print(f"[Trainer] Focal loss gamma={focal_gamma}")
            self.criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        else:
            print(f"[Trainer] Cross-Entropy, label_smoothing={label_smoothing}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        if use_8bit_adam and HAS_8BIT:
            print("[Trainer] 8-bit AdamW")
            self.optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
            )
        else:
            print("[Trainer] Standard AdamW")
            self.optimizer = AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
            )
            
        # === ADD THIS DEBUG ===
        # Verify weights are actually in the loss function
        if class_weights is not None:
            if hasattr(self.criterion, 'weight'):
                if self.criterion.weight is not None:
                    print(f"[Trainer] ✓ Loss function has weights: {self.criterion.weight.cpu().numpy()}")
                else:
                    print(f"[Trainer] ❌ WARNING: Loss function weight is None!")
            else:
                print(f"[Trainer] ❌ WARNING: Loss function has no weight attribute!")
        else:
            print(f"[Trainer] Loss function using uniform weights")

        total_steps = len(train_loader) * epochs // self.gradient_accumulation_steps
        warmup_steps = len(train_loader) * warmup_epochs // self.gradient_accumulation_steps
        if lr_schedule == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr*0.1)
        elif lr_schedule == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=max(1, total_steps),
                pct_start=max(1, warmup_steps) / max(1, total_steps),
                anneal_strategy='cos'
            )
        else:
            self.scheduler = None

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [],
            'tokens_seen': [], 'tokens_per_sec': [],
        }
        self.best_val_acc, self.best_model_state = 0.0, None
        self.early_stop_counter = 0

        self.output_dir = "runs/default"
        self.temp_checkpoint_path = "checkpoint_temp.pt"
        self.final_model_path = "model_final.pt"
        self.plot_curves_path = "training_curves.png"
        self.plot_confusion_path = "confusion_matrix.png"
        self.history_csv_path = "training_history.csv"

        # In your Trainer.__init__ or loss function:
        print(f"\n[DEBUG] Using class weights: {self.use_class_weights}")
        print(f"[DEBUG] Class weights tensor: {class_weights}")
        print(f"[DEBUG] Loss function: {self.criterion}")

        # Check if CrossEntropyLoss actually has the weights:
        if hasattr(self.criterion, 'weight'):
            print(f"[DEBUG] Loss function weight parameter: {self.criterion.weight}")
        else:
            print(f"❌ Loss function has NO weight parameter!")

    def _tokens_in_batch(self, batch_size) -> int:
        return int(batch_size) * int(self.context_len_for_tokens or 1)

    def predict_with_confidence_threshold(self, X, min_confidence_down=0.7, min_confidence_up=0.7):
        """
        Only predict Down/Up if model is highly confident.
        Default to Neutral otherwise.
        
        Args:
            min_confidence_down: Minimum probability to predict Down
            min_confidence_up: Minimum probability to predict Up
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
        
        # Get raw predictions
        raw_preds = logits.argmax(dim=1)
        
        # Apply confidence thresholds
        confident_preds = []
        for i in range(len(probs)):
            prob_down, prob_neutral, prob_up = probs[i]
            
            if raw_preds[i] == 0 and prob_down >= min_confidence_down:
                confident_preds.append(0)  # Confident Down
            elif raw_preds[i] == 2 and prob_up >= min_confidence_up:
                confident_preds.append(2)  # Confident Up
            else:
                confident_preds.append(1)  # Default to Neutral (no trade)
        
        return torch.tensor(confident_preds), probs

    def validate_with_thresholds(self, confidence_threshold=0.7):
        """Evaluate with confidence-based filtering."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds, probs = self.predict_with_confidence_threshold(
                    X, 
                    min_confidence_down=confidence_threshold,
                    min_confidence_up=confidence_threshold
                )
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Now calculate metrics - Down/Up will have higher precision
        precision_per = precision_score(all_labels, all_preds, average=None, labels=[0,1,2], zero_division=0)
        recall_per = recall_score(all_labels, all_preds, average=None, labels=[0,1,2], zero_division=0)
        
        print(f"[Threshold={confidence_threshold}]")
        print(f"  Down  - P: {precision_per[0]:.3f} | R: {recall_per[0]:.3f}")
        print(f"  Up    - P: {precision_per[2]:.3f} | R: {recall_per[2]:.3f}")
        
        return all_preds, all_labels, all_probs

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        start = time.time()
        tokens_seen_this_epoch = 0
        smoothed_tps = None
        # ============ ADD GRADIENT TRACKING ============
        grad_norms = []

        for b_idx, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=(self.amp_dtype is not None)):
                logits = self.model(X)
                loss = self.criterion(logits, y) / self.gradient_accumulation_steps
            loss.backward()

            if ((b_idx + 1) % self.gradient_accumulation_steps) == 0:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5 

                grad_norms.append(total_norm)

    
                # ==========================
                # HARD GRADIENT CLIPPING
                # ==========================
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_schedule == 'onecycle' and self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            tokens_seen_this_epoch += self._tokens_in_batch(X.size(0))
            elapsed = max(1e-6, time.time() - start)
            tps = tokens_seen_this_epoch / elapsed
            smoothed_tps = tps if smoothed_tps is None else 0.9 * smoothed_tps + 0.1 * tps

            pbar.set_postfix({
                'loss': f'{total_loss / (b_idx + 1):.4f}',
                'acc': f'{100.0 * correct / max(1, total):.2f}%',
                'tok/s': f'{smoothed_tps:.0f}'
            })

            if self.token_budget is not None:
                cumulative_tokens = (self.history['tokens_seen'][-1] if self.history['tokens_seen'] else 0) + tokens_seen_this_epoch
                if cumulative_tokens >= self.token_budget:
                    # Flush accumulated gradients before breaking
                    if (b_idx + 1) % self.gradient_accumulation_steps != 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    break

        if grad_norms:
            print(f"[Gradients] Mean: {np.mean(grad_norms):.4f} | "
                f"Max: {np.max(grad_norms):.4f} | "
                f"Min: {np.min(grad_norms):.4f}")
            if np.max(grad_norms) > 100:
                print("  ⚠️ WARNING: Large gradients detected!")
            if np.mean(grad_norms) < 0.01:
                print("  ⚠️ WARNING: Tiny gradients detected!")

        avg_loss = total_loss / max(1, (b_idx + 1))
        avg_acc = correct / max(1, total)
        tokens_epoch = tokens_seen_this_epoch
        tokens_per_sec = smoothed_tps or 0.0
        return avg_loss, avg_acc, tokens_epoch, tokens_per_sec

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for X, y in tqdm(self.val_loader, desc="Validation", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                    enabled=(self.amp_dtype is not None)):
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    probs = F.softmax(logits, dim=1)
                total_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def train(self):
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"[Config] Epochs={self.epochs} | Batch={self.train_loader.batch_size} | Accum={self.gradient_accumulation_steps}")
        print(f"[Config] LR={self.optimizer.param_groups[0]['lr']} | Schedule={self.lr_schedule} | AMP={self.amp_dtype}")
        print(f"[Config] Token budget: {self.token_budget if self.token_budget else 'none'}")
        early_stop_enabled = (self.early_stop_patience is not None and self.early_stop_patience > 0)

        total_tokens_seen = 0
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}\n" + "-"*80)
            train_loss, train_acc, tokens_epoch, tokens_per_sec = self.train_epoch()
            total_tokens_seen += tokens_epoch
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate()

            f1 = f1_score(val_labels, val_preds, average='macro')
            precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
            f1_per = f1_score(val_labels, val_preds, average=None, labels=[0, 1, 2])
            recall_per = recall_score(val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0)
            # Calculate F0.5 (precision weighted 2x more than recall)
            f_beta = fbeta_score(val_labels, val_preds, beta=0.5, average='macro')
            # Also track per-class precision for monitoring
            precision_per = precision_score(val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0)
        
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            self.history['tokens_seen'].append(total_tokens_seen)
            self.history['tokens_per_sec'].append(tokens_per_sec)

            print(f"[Results] Train Loss {train_loss:.4f} | Acc {train_acc*100:.2f}% | tok/s {tokens_per_sec:.0f}")
            print(f"[Results] Val   Loss {val_loss:.4f} | Acc {val_acc*100:.2f}% | F1 {f1:.4f} | P {precision:.4f} | R {recall:.4f}")
            print(f"[Per-class] F1: Down {f1_per[0]:.3f} | Neutral {f1_per[1]:.3f} | Up {f1_per[2]:.3f}")
            print(f"[Per-class] Acc*: Down {recall_per[0]:.3f} | Neutral {recall_per[1]:.3f} | Up {recall_per[2]:.3f}  (*recall)")
            
            precision_macro = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            recall_macro    = recall_score(val_labels, val_preds, average='macro', zero_division=0)

            precision_per = precision_score(val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0)
            recall_per    = recall_score(val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0)

            print(
                f"[Results] Val Loss {val_loss:.4f} | Acc {val_acc*100:.2f}% | "
                f"P_macro {precision_macro:.4f} | R_macro {recall_macro:.4f}"
            )
            print(
                f"[Per-class] P: Down {precision_per[0]:.3f} | Neutral {precision_per[1]:.3f} | Up {precision_per[2]:.3f}"
            )
            print(
                f"[Per-class] R: Down {recall_per[0]:.3f} | Neutral {recall_per[1]:.3f} | Up {recall_per[2]:.3f}"
            )
            # ============================
            # Early stopping on F-beta
            # ============================

            # ----- Early stopping on F-beta -----
            if self.best_val_metric is None:
                # First time (or after reset): always treat as improvement
                improvement = float('inf')
                old_best_for_print = 0.0
            else:
                improvement = f_beta - self.best_val_metric
                old_best_for_print = self.best_val_metric

            if improvement > self.early_stop_min_delta:
                print(
                    f"[Checkpoint] Improved F{self.beta:.2f}: "
                    f"{old_best_for_print:.4f} → {f_beta:.4f}"
                )
                # Track best F-beta and its corresponding accuracy
                self.best_val_metric = f_beta
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
                self._save_checkpoint(epoch, val_acc, is_best=True)
            else:
                if early_stop_enabled:
                    self.early_stop_counter += 1
                    print(
                        f"[Early Stop] No improvement "
                        f"{self.early_stop_counter}/{self.early_stop_patience}"
                    )
                    if self.early_stop_counter >= self.early_stop_patience:
                        print("[Early Stop] Patience exhausted.")
                        break

            # Always save a “last epoch” checkpoint as non-best
            self._save_checkpoint(epoch, val_acc, is_best=False)




            if self.lr_schedule == 'cosine' and self.scheduler is not None:
                self.scheduler.step()

            if self.device.type == 'cuda':
                print_memory_usage(self.device)

            if self.token_budget is not None and total_tokens_seen >= self.token_budget:
                print(f"[Budget] Token budget reached: {total_tokens_seen} ≥ {self.token_budget}")
                break

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
       

        # Early stopping: load best model if we have it
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(
                f"[Early Stop] Restored best model with "
                f"F{self.beta:.2f}={self.best_val_metric:.4f}, "
                f"best_val_acc={self.best_val_acc:.4f}"
            )
        
            print(f"[Best] Val Acc (of prior epochs): {self.best_val_acc*100:.2f}% | Tokens seen: {total_tokens_seen:,}")



        print("\n[Evaluation] Final validation...")
        val_loss, val_acc, val_preds, val_labels, val_probs = self.validate()
        f1 = f1_score(val_labels, val_preds, average='macro')
        precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        f1_per = f1_score(val_labels, val_preds, average=None, labels=[0, 1, 2])
        recall_per = recall_score(val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0)
        print(f"[Final] Acc {val_acc*100:.2f}% | F1 {f1:.4f} | P {precision:.4f} | R {recall:.4f}")
        print(f"[Final] F1 per-class: {f1_per}")
        print(f"[Final] Acc* per-class: {recall_per}")

        self._save_final_model()
        self._plot_training_curves()
        self._plot_confusion_matrix(val_labels, val_preds)
        self._plot_roc_auc(val_labels, val_probs, str(Path(self.output_dir) / "roc_curves.png"))
        self._plot_pr_curves(val_labels, val_probs, str(Path(self.output_dir) / "pr_curves.png"))
        self._save_history_csv()
        self._save_val_predictions(val_labels, val_preds, val_probs)

    def _save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': {
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers,
                'max_seq_len': self.model.max_seq_len,
            }
        }
        torch.save(checkpoint, self.temp_checkpoint_path)
        if is_best:
            best_path = self.temp_checkpoint_path.replace('temp', 'best')
            torch.save(checkpoint, best_path)

    def reset_early_stopping(self):
        """
        Reset early-stopping state so that each new contract
        starts with a fresh patience window.
        """
        self.best_val_metric = None
        self.no_improvement_epochs = 0

    def _save_final_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers,
                'max_seq_len': self.model.max_seq_len,
            },
            'best_val_acc': self.best_val_acc
        }, self.final_model_path)
        print(f"[Save] Model → {self.final_model_path}")

    def _plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs, [x*100 for x in self.history['train_acc']], 'b-', label='Train Acc')
        ax2.plot(epochs, [x*100 for x in self.history['val_acc']], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Save] Curves → {self.plot_curves_path}")

    def _plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100.0
            fmt, title = ".1f", "Confusion Matrix (%)"
        else:
            fmt, title = "d", "Confusion Matrix (counts)"
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=['Down', 'Neutral', 'Up'],
            yticklabels=['Down', 'Neutral', 'Up'],
            cbar=False
        )
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); plt.tight_layout()
        plt.savefig(self.plot_confusion_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Save] Confusion → {self.plot_confusion_path}")

    def _plot_roc_auc(self, y_true, y_proba, save_path):
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(['Down', 'Neutral', 'Up']):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Save] ROC → {save_path}")

    def _plot_pr_curves(self, y_true, y_proba, save_path):
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(['Down', 'Neutral', 'Up']):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f"{label} (AUC={pr_auc:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall Curves"); plt.legend(); plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Save] PR → {save_path}")

    def _save_history_csv(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.history_csv_path, index=False)
        print(f"[Save] History CSV → {self.history_csv_path}")

    def _save_val_predictions(self, y_true, y_pred, y_proba):
        """
        Save per-sample validation predictions:
        - bar_label_y: ground truth class
        - bar_label_X: predicted class
        - prob_*: class probabilities
        """
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'bar_label_y': y_pred.astype(int),
            'bar_label_X': y_true.astype(int),
            'prob_down': y_proba[:, 0],
            'prob_neutral': y_proba[:, 1],
            'prob_up': y_proba[:, 2],
        })
        out_path = out_dir / "val_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"[Save] Val predictions → {out_path}")

# =========================
# Stratified validation
# =========================

def create_stratified_validation_subset(dataset, samples_per_class=1024):
    print(f"[Stratified Val] Building balanced set...")
    all_labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        all_labels.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
    by_class = {0: [], 1: [], 2: []}
    for idx, lab in enumerate(all_labels):
        if lab in by_class:
            by_class[lab].append(idx)
    available = {c: len(v) for c, v in by_class.items()}
    k = min(min(available.values()), samples_per_class)
    if k == 0:
        raise ValueError("Not enough samples per class.")
    sel = []
    for c in [0, 1, 2]:
        sel.extend(by_class[c][:k])
    random.Random(42).shuffle(sel)
    print(f"[Stratified Val] k={k} per class → total={len(sel)}")
    return Subset(dataset, sel)

# =========================
# Model builder
# =========================

def build_model(size_name, context_len, n_features, dropout=0.2,
                use_gqa=True, use_alibi=True, norm_style="rms_pre", ffn_style="swiglu", use_learned_pos=True):
    """
    size_name:
      - 'S', 'M', 'L'  → BarrierGPTConfigurable (your main model)
      - 'Vanilla'      → GPTModel (Custom* blocks), S-sized, no GQA/ALiBi, vanilla LayerNorm+ReLU
    """

    # ------------------------------------------------------------------
    # Special case: Vanilla GPT classifier using Custom* blocks
    # ------------------------------------------------------------------
    if size_name == "Vanilla":
        cfg = {'d_model': 384, 'n_layers': 8, 'n_heads': 6}
        model = GPTModel(
            n_features=n_features,
            max_seq_len=context_len,
            n_classes=3,
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            layers=cfg['n_layers'],
            dropout=dropout,
            use_sinusoidal_pos_emb=False,
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[Model] {size_name} (Vanilla GPT) | "
            f"params={total_params/1e6:.1f}M | trainable={trainable_params/1e6:.1f}M | "
            f"cfg={cfg}"
        )
        return model

    # ------------------------------------------------------------------
    # Original configs for BarrierGPTConfigurable (unchanged)
    # ------------------------------------------------------------------
    configs = {
        'S': {'d_model': 384, 'n_layers': 8,  'n_heads': 6,  'kv_groups': 3},
        'M': {'d_model': 512, 'n_layers': 12, 'n_heads': 8,  'kv_groups': 4},
        'L': {'d_model': 768, 'n_layers': 16, 'n_heads': 12, 'kv_groups': 4},
    }

    if size_name not in configs:
        raise ValueError(f"Unknown model size: {size_name}")

    cfg = configs[size_name]

    model = BarrierGPTConfigurable(
        n_features=n_features,
        max_seq_len=context_len,
        n_classes=3,
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        kv_groups=cfg['kv_groups'],
        ffn_mult=4,
        dropout=dropout,
        use_gqa=use_gqa,
        use_alibi=use_alibi,
        norm_style=norm_style,
        ffn_style=ffn_style,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Model] {size_name} | params={total_params/1e6:.1f}M | "
        f"trainable={trainable_params/1e6:.1f}M | cfg={cfg} | "
        f"gqa={use_gqa} | alibi={use_alibi} | norm={norm_style} | ffn={ffn_style}"
    )
    return model


# =========================
# Checkpoint loading
# =========================

def load_checkpoint_strict(model, checkpoint_path, device):
    print(f"[Checkpoint] Loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get('config', {})
    curr = {
        'd_model': model.d_model,
        'n_layers': model.n_layers,
        'max_seq_len': model.max_seq_len
    }
    mismatches = []
    for k in curr:
        if k in saved and saved[k] != curr[k]:
            mismatches.append(f"{k}: saved={saved[k]} vs current={curr[k]}")
    if mismatches:
        raise ValueError("[Checkpoint] Architecture mismatch:\n" + "\n".join(" - " + m for m in mismatches))
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', 0)
    bva = ckpt.get('best_val_acc', 0.0)
    print(f"[Checkpoint] Loaded OK | epoch={epoch} | best_val_acc={bva:.3f}")
    return epoch, bva

# =========================
# Experiment helpers
# =========================

def evaluate_context_lengths(model, device, data_file, base_output_dir, batch_size, val_size, contexts: List[int], bar_label_scale: float = 1.0):
    rows = []
    for ctx in contexts:
        print(f"\n[Context Stress] Evaluate at context_len={ctx}")
        ds = BarDataset(
            parquet_path=data_file,
            context_len=ctx,
            stride=1,
            is_validation=True,
            val_size=val_size,
            target_col=args.target_col, 
            bar_label_scale=bar_label_scale
        )

        val_subset = create_stratified_validation_subset(ds, samples_per_class=min(1024, len(ds)//3))
        loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))
        tmp_trainer = Trainer(
            model=model, train_loader=loader, val_loader=loader, device=device,
            epochs=1, amp_dtype=torch.bfloat16, grad_clip=1.0,
            token_budget=None, context_len_for_tokens=ctx,
            use_fbeta=args.use_fbeta,
            beta=args.beta,
            use_class_weights=args.use_class_weights,
            confidence_threshold=args.confidence_threshold
        )
        tmp_trainer.output_dir = str(base_output_dir / f"context_{ctx}")
        Path(tmp_trainer.output_dir).mkdir(parents=True, exist_ok=True)
        vloss, vacc, vpred, vlab, vprob = tmp_trainer.validate()
        f1 = f1_score(vlab, vpred, average='macro')
        precision = precision_score(vlab, vpred, average='macro', zero_division=0)
        recall = recall_score(vlab, vpred, average='macro', zero_division=0)
        rows.append({
            'context_len': ctx,
            'val_loss': vloss,
            'val_acc': vacc,
            'macro_f1': f1,
            'precision': precision,
            'recall': recall
        })
        tmp_trainer._plot_confusion_matrix(vlab, vpred)
        tmp_trainer._plot_roc_auc(vlab, vprob, str(Path(tmp_trainer.output_dir) / "roc.png"))
        tmp_trainer._plot_pr_curves(vlab, vprob, str(Path(tmp_trainer.output_dir) / "pr.png"))
        tmp_trainer._save_val_predictions(vlab, vpred, vprob)
    df = pd.DataFrame(rows)
    df.to_csv(base_output_dir / "context_stress_summary.csv", index=False)
    print(f"[Context Stress] Summary → {base_output_dir / 'context_stress_summary.csv'}")

def write_runrow(summary_csv, row: Dict):
    df = pd.DataFrame([row])
    if summary_csv.exists():
        df0 = pd.read_csv(summary_csv)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(summary_csv, index=False)



def diagnose_predictability(train_ds, name="Dataset"):
    """
    Comprehensive analysis of feature quality and task difficulty.
    
    Args:
        train_ds: BarDataset instance
        name: Name for logging (e.g., "Contract 1")
    """
    print("\n" + "="*80)
    print(f"PREDICTABILITY ANALYSIS: {name}")
    print("="*80)
    
    labels = train_ds.labels
    data = train_ds.data
    n_samples = len(labels)
    
    # ============================================
    # 1. LABEL STATISTICS
    # ============================================
    print("\n[1. Label Statistics]")
    counts = np.bincount(labels, minlength=3)
    pcts = counts / counts.sum() * 100
    print(f"  Class Balance:")
    print(f"    Down:    {counts[0]:6d} ({pcts[0]:5.1f}%)")
    print(f"    Neutral: {counts[1]:6d} ({pcts[1]:5.1f}%)")
    print(f"    Up:      {counts[2]:6d} ({pcts[2]:5.1f}%)")
    
    # Check for extreme imbalance
    max_pct = pcts.max()
    if max_pct > 80:
        print(f"  ⚠️  WARNING: Extreme imbalance ({max_pct:.1f}% single class)")
        print(f"     → Model will likely predict majority class only")
    elif max_pct > 60:
        print(f"  ⚠️  WARNING: Moderate imbalance ({max_pct:.1f}% single class)")
        print(f"     → Consider class weighting or resampling")
    else:
        print(f"  ✅ Balanced enough for training")
    
    # ============================================
    # 2. LABEL AUTOCORRELATION
    # ============================================
    print("\n[2. Label Autocorrelation]")
    print("  (Measures if past labels predict future labels)")
    
    autocorr_results = {}
    for lag in [1, 2, 3, 5, 10, 20]:
        if n_samples > lag:
            past = labels[:-lag]
            future = labels[lag:]
            corr = np.corrcoef(past, future)[0, 1]
            autocorr_results[lag] = corr
            print(f"    Lag {lag:2d}: {corr:+.4f}")
    
    # Interpretation
    max_autocorr = max(abs(v) for v in autocorr_results.values())
    if max_autocorr < 0.05:
        print(f"  ⚠️  WEAK SIGNAL: Max autocorr = {max_autocorr:.4f}")
        print(f"     → Labels are nearly independent (random-walk-like)")
        print(f"     → Past labels don't predict future labels")
    elif max_autocorr < 0.15:
        print(f"  ✓  MODERATE SIGNAL: Max autocorr = {max_autocorr:.4f}")
        print(f"     → Some persistence or mean reversion present")
    else:
        print(f"  ✅ STRONG SIGNAL: Max autocorr = {max_autocorr:.4f}")
        print(f"     → Labels show clear patterns")
    
    # ============================================
    # 3. TRANSITION MATRIX
    # ============================================
    print("\n[3. Transition Probabilities]")
    print("  (What typically follows each label?)")
    
    transitions = np.zeros((3, 3))
    for i in range(n_samples - 1):
        curr, next_label = labels[i], labels[i+1]
        transitions[curr, next_label] += 1
    
    # Normalize rows to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    trans_probs = transitions / np.maximum(row_sums, 1)
    
    print("             →Down  →Neutral    →Up")
    for curr in [0, 1, 2]:
        curr_name = ['Down', 'Neutral', 'Up'][curr]
        probs = [trans_probs[curr, j] * 100 for j in range(3)]
        print(f"  {curr_name:8s}  {probs[0]:5.1f}%   {probs[1]:5.1f}%  {probs[2]:5.1f}%")
    
    # Interpretation
    print("\n  Interpretation:")
    if trans_probs[0, 0] > 0.40:  # Down → Down
        print("    • Downward momentum detected (Down → Down)")
    if trans_probs[2, 2] > 0.40:  # Up → Up
        print("    • Upward momentum detected (Up → Up)")
    if trans_probs[0, 2] > trans_probs[0, 0]:  # Down → Up more than Down → Down
        print("    • Mean reversion: Down bars followed by Up")
    if trans_probs[2, 0] > trans_probs[2, 2]:  # Up → Down more than Up → Up
        print("    • Mean reversion: Up bars followed by Down")
    
    # ============================================
    # 4. PREDICTIVE FEATURE CORRELATION
    # ============================================
    print("\n[4. Feature-Target Predictive Correlation]")
    print("  (Do features at time t predict label at time t+1?)")
    
    if n_samples > 1:
        # Features at bars [0, 1, ..., N-2]
        past_features = data[:-1, :]
        # Labels at bars [1, 2, ..., N-1] (next bar)
        future_labels = labels[1:]
        
        correlations = []
        for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
            corr = np.corrcoef(past_features[:, i], future_labels)[0, 1]
            correlations.append((col, abs(corr), corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("  Top 10 most predictive features:")
        for col, abs_corr, corr in correlations[:10]:
            emoji = "✅" if abs_corr > 0.1 else "⚠️" if abs_corr > 0.05 else "❌"
            print(f"    {emoji} {col:20s}: {corr:+.4f}")
        
        # Summary
        max_corr = correlations[0][1]
        if max_corr < 0.05:
            print(f"\n  ❌ NO PREDICTIVE SIGNAL: Max correlation = {max_corr:.4f}")
            print(f"     → Features at time t don't predict label at t+1")
            print(f"     → Expected accuracy: ~33-38% (barely above random)")
        elif max_corr < 0.15:
            print(f"\n  ⚠️  WEAK SIGNAL: Max correlation = {max_corr:.4f}")
            print(f"     → Some features have slight predictive power")
            print(f"     → Expected accuracy: ~38-45%")
        elif max_corr < 0.30:
            print(f"\n  ✓  MODERATE SIGNAL: Max correlation = {max_corr:.4f}")
            print(f"     → Features provide useful information")
            print(f"     → Expected accuracy: ~45-55%")
        else:
            print(f"\n  ✅ STRONG SIGNAL: Max correlation = {max_corr:.4f}")
            print(f"     → Features are highly predictive")
            print(f"     → Expected accuracy: >55%")
    
    # ============================================
    # 5. SAME-BAR CORRELATION (LEAKAGE CHECK)
    # ============================================
    print("\n[5. Same-Bar Correlation (Leakage Detection)]")
    print("  (Do features at time t describe label at time t?)")
    
    same_bar_corrs = []
    for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
        corr = np.corrcoef(data[:, i], labels)[0, 1]
        same_bar_corrs.append((col, abs(corr), corr))
    
    same_bar_corrs.sort(key=lambda x: x[1], reverse=True)
    
    print("  Features with highest same-bar correlation:")
    has_leakage = False
    for col, abs_corr, corr in same_bar_corrs[:5]:
        if abs_corr > 0.5:
            print(f"    ⚠️  {col:20s}: {corr:+.4f}  ← POTENTIAL LEAKAGE!")
            has_leakage = True
        else:
            print(f"    ✓  {col:20s}: {corr:+.4f}")
    
    if has_leakage:
        print("\n  ⚠️  WARNING: High same-bar correlation detected!")
        print("     → Some features may describe the current bar's label")
        print("     → This is OK if features are lagged (from previous bars)")
        print("     → This is BAD if features include current bar information")
    else:
        print("\n  ✅ No obvious leakage detected")
    
    # ============================================
    # 6. FEATURE VARIANCE
    # ============================================
    print("\n[6. Feature Variance]")
    print("  (Features need variance to be useful)")
    
    low_variance_features = []
    for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
        feat_std = data[:, i].std()
        if feat_std < 0.01:
            low_variance_features.append((col, feat_std))
            print(f"    ⚠️  {col:20s}: std={feat_std:.6f}  ← NEARLY CONSTANT!")
    
    if low_variance_features:
        print(f"\n  ⚠️  {len(low_variance_features)} features have very low variance")
        print("     → Consider removing these features")
    else:
        print("  ✅ All features have adequate variance")
    
    # ============================================
    # 7. BASELINE ACCURACY ESTIMATE
    # ============================================
    print("\n[7. Estimated Baseline Accuracies]")
    
    # Random guessing
    random_acc = 1.0 / 3.0
    print(f"  Random guessing:        {random_acc*100:5.1f}%")
    
    # Majority class
    majority_acc = pcts.max() / 100
    print(f"  Always predict majority: {majority_acc*100:5.1f}%")
    
    # Persistence (predict same as previous)
    if n_samples > 1:
        persistence_correct = (labels[:-1] == labels[1:]).sum()
        persistence_acc = persistence_correct / (n_samples - 1)
        print(f"  Persistence (repeat t-1): {persistence_acc*100:5.1f}%")
    
    # With features (rough estimate based on max correlation)
    if n_samples > 1:
        max_pred_corr = correlations[0][1]
        # Very rough heuristic: acc ≈ 33% + 80% * correlation
        estimated_acc = 0.33 + 0.80 * max_pred_corr
        print(f"  Estimated with features: {estimated_acc*100:5.1f}%")
    
    # ============================================
    # 8. FINAL VERDICT
    # ============================================
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    # Score the dataset
    score = 0
    issues = []
    
    # Check balance
    if max_pct < 60:
        score += 2
    elif max_pct < 80:
        score += 1
        issues.append(f"Imbalanced classes ({max_pct:.1f}%)")
    else:
        issues.append(f"SEVERE imbalance ({max_pct:.1f}%)")
    
    # Check autocorrelation
    if max_autocorr > 0.15:
        score += 2
    elif max_autocorr > 0.05:
        score += 1
        issues.append(f"Weak label autocorrelation ({max_autocorr:.3f})")
    else:
        issues.append(f"NO label autocorrelation ({max_autocorr:.3f})")
    
    # Check feature correlation
    if n_samples > 1:
        if max_corr > 0.20:
            score += 2
        elif max_corr > 0.10:
            score += 1
            issues.append(f"Weak feature-target correlation ({max_corr:.3f})")
        else:
            issues.append(f"NO predictive features ({max_corr:.3f})")
    
    # Final assessment
    print(f"\nTrainability Score: {score}/6")
    
    if score >= 5:
        print("✅ EXCELLENT: Strong signal, should train well")
        print("   Expected accuracy: 50-70%")
    elif score >= 4:
        print("✓  GOOD: Moderate signal, training should work")
        print("   Expected accuracy: 45-55%")
    elif score >= 3:
        print("⚠️  MARGINAL: Weak signal, training will be difficult")
        print("   Expected accuracy: 38-45%")
        print("   Recommendation: Add more informative features")
    else:
        print("❌ POOR: Insufficient signal for reliable training")
        print("   Expected accuracy: 33-40% (barely above random)")
        print("   Recommendation: Redesign features or change task")
    
    if issues:
        print("\nIssues detected:")
        for issue in issues:
            print(f"  • {issue}")
    
    print("="*80 + "\n")
    
    return {
        'score': score,
        'max_autocorr': max_autocorr,
        'max_feature_corr': max_corr if n_samples > 1 else 0.0,
        'class_balance': pcts,
        'estimated_accuracy': estimated_acc if n_samples > 1 else random_acc
    }

# GPT2 style basic self attention Transformer implemented in class
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = torch.nn.Parameter(0.01 * torch.randn((output_size, input_size)))
        self.bias = torch.nn.Parameter(torch.zeros((output_size,)))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class CustomEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(0.01 * torch.randn((num_embeddings, embedding_dim)))

    def forward(self, x):
        return self.weight[x]


class CustomMHA(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = torch.nn.Parameter(0.01 * torch.randn((3 * d_model, d_model)))
        self.wo  = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))

    def forward(self, x):
        added_batch = False
        if x.dim() == 2:
            added_batch = True
            x = x[None, :, :]

        B, S, D = x.shape
        QKV = x @ self.qkv.T                      # (B, S, 3D)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)

        dh = D // self.n_heads
        # (B, S, h, dh) -> (B, h, S, dh)
        q = Q.view(B, S, self.n_heads, dh).transpose(1, 2).contiguous()
        k = K.view(B, S, self.n_heads, dh).transpose(1, 2).contiguous()
        v = V.view(B, S, self.n_heads, dh).transpose(1, 2).contiguous()

        # Fast fused attention (uses Flash/ME kernels on GPU)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, h, S, dh)

        # (B, h, S, dh) -> (B, S, D)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        y = y @ self.wo.T

        if added_batch:
            y = y[0]
        return y


class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((d_model,))
        self.mha   = CustomMHA(d_model, n_heads)
        self.norm2 = torch.nn.LayerNorm((d_model,))
        self.fc1   = CustomLinear(d_model, 4 * d_model)
        self.act   = torch.nn.ReLU()
        self.fc2   = CustomLinear(4 * d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, causal_mask_ref=None):
        # causal_mask_ref kept for compatibility; not used because SDPA is causal
        self.mha._causal_mask_ref = causal_mask_ref
        x = x + self.mha(self.norm1(x))
        x = x + self.dropout(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return x


class GPTModel(torch.nn.Module):
    """
    Vanilla decoder-only model for time-series classification:
      x: (B, S, n_features)
      output: (B, n_classes) by default (last token),
              or (B, S, n_classes) if return_all_logits=True
    """
    def __init__(self, n_features, max_seq_len, n_classes,
                 d_model, n_heads, layers, dropout=0.1, use_sinusoidal_pos_emb: bool = False, use_learned_pos=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = layers
        self.max_seq_len = max_seq_len
        self.use_sinusoidal_pos_emb = use_sinusoidal_pos_emb
        self.use_learned_pos = use_learned_pos
        # Project continuous features into d_model
        self.input_proj = CustomLinear(n_features, d_model)

        if self.use_sinusoidal_pos_emb:
            # Sinusoidal positional encoding (no learned embedding table)
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
            self.position_embeddings = None
        elif self.use_learned_pos:
            # Learned positional embeddings (GPT-2 style)
            self.position_embeddings = CustomEmbedding(max_seq_len, d_model)
            self.pos_encoding = None
        # Stack of decoder blocks
        self.layers = torch.nn.ModuleList(
            [TransformerDecoderBlock(d_model, n_heads, dropout=dropout)
             for _ in range(layers)]
        )

        # Classifier head (per time step)
        self.classifier = CustomLinear(d_model, n_classes)

        self.dropout = torch.nn.Dropout(dropout)

        # placeholders; allocate on device later if you want
        self.register_buffer("pos_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def prepare_buffers(self, device):
        # Build on device once; zero extra CPU ↔ GPU traffic in forward.
        self.pos_idx = torch.arange(self.max_seq_len, dtype=torch.long, device=device)

    def forward(self, x, return_all_logits=False):
        """
        x: (B, S, n_features)
        """
        B, S, F = x.shape

        # Lazy-create pos_idx on correct device if prepare_buffers was not called
        if self.pos_idx.numel() == 0 or self.pos_idx.device != x.device:
            self.pos_idx = torch.arange(self.max_seq_len, dtype=torch.long, device=x.device)

        positions = self.pos_idx[:S].unsqueeze(0).expand(B, -1)  # (B, S)

        # Embed: project features + add learned positions
        h = self.input_proj(x) + self.position_embeddings(positions)
        
        if self.use_sinusoidal_pos_emb:
            # Add sinusoidal PE
            h = self.pos_encoding(h)
        else:
            # Add learned positional embeddings
            positions = self.pos_idx[:S].unsqueeze(0).expand(B, -1)  # (B, S)
            h = h + self.position_embeddings(positions)
        
        h = self.dropout(h)

        # Decoder stack
        for layer in self.layers:
            h = layer(h)

        h = self.dropout(h)

        # Per-step logits
        logits = self.classifier(h)  # (B, S, n_classes)

        # Match BarrierGPT interface
        return logits if return_all_logits else logits[:, -1, :]


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Vaswani et al. (2017) sinusoidal positional encoding.

    Expects input x: (B, L, D), returns x + PE[0:L] broadcasted over batch.
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, L, D) for easy broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        return x + self.pe[:, :L, :D]


# End GPT2 style transformer created in class

# ECE ploting function
def compute_ece(probs, labels, n_bins: int = 15) -> float:
    """
    Expected Calibration Error for multi-class probabilities.

    probs: (N, C) array-like of predicted class probabilities
    labels: (N,) array-like of true integer labels
    n_bins: number of confidence bins
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    # predicted class and its confidence
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if not np.any(in_bin):
            continue
        prop = in_bin.mean()
        acc_bin = accuracies[in_bin].mean()
        conf_bin = confidences[in_bin].mean()
        ece += np.abs(acc_bin - conf_bin) * prop

    return float(ece)


# =========================
# MAIN
# =========================

def main():
    torch.cuda.empty_cache(); gc.collect()
    parser = argparse.ArgumentParser(
        description="BarrierGPT sequential training + experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--data-files", type=str, required=False, default=(
       
        r"D:\FutureFormer3B\data\MNQ_with_new_features\MNQ_0924_tick_202406130100.Last_202409221400.Last_v1000_b10_rth_scaled_bar_oldbarlabel.parquet,"
        r"D:\FutureFormer3B\data\MNQ_with_new_features\MNQ_1224_tick_202409130000.Last_202412210900.Last_v1000_b10_rth_scaled_bar_oldbarlabel.parquet,"
        r"D:\FutureFormer3B\data\MNQ_with_new_features\MNQ_0325_tick_202412130000.Last_202503220900.Last_v1000_b10_rth_scaled_bar_oldbarlabel.parquet,"
        r"D:\FutureFormer3B\data\MNQ_with_new_features\MNQ_0625_tick_202503140000.Last_202506220700.Last_v1000_b10_rth_scaled_bar_oldbarlabel.parquet,"
        r"D:\FutureFormer3B\data\MNQ_with_new_features\MNQ_0925_tick_202506121000.Last_202509190700.Last_v1000_b10_rth_scaled_bar_oldbarlabel.parquet,"
        r"D:\FutureFormer3B\data\MNQ_with_new_features\MNQ_1225_tick_202509110100.Last_202511110600.Last_v1000_b10_rth_scaled_bar_oldbarlabel.parquet"
    ), help="Comma-separated parquet paths")

    parser.add_argument(
        "--bar-label-scale",
        type=float,
        default=1.0,
        help="Scalar multiplier applied ONLY to bar_label feature in the context window (past bars)."
    )
    parser.add_argument("--val-size", type=int, default=6000)
    parser.add_argument("--val-samples-per-class", type=int, default=2000)

    # Core training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Model
    parser.add_argument("--size", default="S", choices=["S", "M", "L", "Vanilla"])
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--val-stride", type=int, default=1)

    # Opt / AMP / GPU
    parser.add_argument("--no-8bit-adam", action="store_true")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16", "auto"], default="bf16")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.95)
    parser.add_argument("--device-id", type=int, default=0)

    # Loss / reg
    parser.add_argument("--use-focal-loss", action="store_true", default=True)
    parser.add_argument("--focal-gamma", type=float, default=1.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-delta", type=float, default=0.01)

    # LR schedule
    parser.add_argument("--lr-schedule", choices=["cosine", "onecycle", "constant"], default="constant")
    parser.add_argument("--warmup-epochs", type=int, default=2)

    # Sequential
    parser.add_argument("--from-checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")

    # Experiments (default is "none" → pure next-token training with past bar_label as feature)
    parser.add_argument("--experiment", choices=["none", "scaling", "context_stress", "ablations"], default="ablations")
    parser.add_argument("--token-budget", type=int, default=400_000_000,
                        help="Max tokens to process in experiment runs (100M default)")
    parser.add_argument("--sizes", type=str, default="S,M,L,Vanilla",
                        help="For scaling: comma list of sizes, e.g. S,M,L")
    parser.add_argument("--eval-contexts", type=str, default="512,1024,2048",
                        help="For context_stress: comma list, e.g. 512,1024,2048")
    parser.add_argument("--ablate", type=str, default="full,labelsmoothing,alibi,gqa,rmsnorm", # swiglu
                        help="For ablations: comma list of components to disable vs full")


    # OR better yet, add our progressive decay as explicit argument:
    parser.add_argument("--progressive-lr-decay", type=float, default=0.5,
                    help="LR decay factor per contract for sequential training (0.5 = 50% reduction per contract)")

    # ============ ADD THESE ============
    parser.add_argument("--use-fbeta", action="store_true", default=True,
                    help="Use F-beta for early stopping instead of accuracy")
    parser.add_argument("--beta", type=float, default=0.5,
                    help="Beta for F-beta score. Beta < 1 weights precision more. "
                            "0.5 = precision 4x more important, 0.25 = 16x more important")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                    help="Minimum probability for confident Down/Up predictions (default to Neutral otherwise)")
    parser.add_argument(
        "--use-class-weights",
        action="store_true", default=True,
        help="If set, enable class weighting (use uniform class weights)."
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='bar_labelH10',
        choices=['bar_label', 'bar_labelH5', 'bar_labelH10', 'fwd_ret_h10'],
        help='Which column to use as the target label'
    )

    # Versioning / runs
    parser.add_argument("--version", type=str, default="auto")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BARRIER GPT SEQUENTIAL / EXPERIMENTS")
    print("="*80)

    total_tokens_global = 0
    total_windows_global = 0
    total_unique_bars_global = 0
    # === Global trackers (work for all experiments) ===
    cumulative_effective_bars = 0
    cumulative_train_windows = 0
    cumulative_tokens_all_contracts = 0

    data_files = [f.strip() for f in args.data_files.split(',') if f.strip()]
    if len(data_files) == 0:
        raise ValueError("No data files provided!")
    print(f"[Data] {len(data_files)} files:")
    for i, f in enumerate(data_files, 1):
        print(f"  {i}. {f}")


    if args.amp == "off":
        amp_dtype = None
    elif args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    else:
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"[AMP] {amp_dtype if amp_dtype else 'disabled'}")
    device = setup_gpu_memory(args.gpu_memory_fraction, args.device_id)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # For scaling, we train multiple sizes → root tagged as SCALING
    if args.experiment == "scaling":
        size_tag = "SCALING"
    if args.experiment == 'ablations':
        size_tag  = "ABLATIONS"   
    if args.experiment == 'context_stress':
        size_tag = "CONTEXT_STRESS"
    else:
        size_tag = args.size

    # Versioning is now per size_tag
    if args.version.strip().lower() == "auto":
        version = _next_version_for_size(size_tag, args.context_len, Path("runs") / size_tag)
        print(f"[Run] Auto version for {size_tag}_{args.context_len}: {version}")
    else:
        version = args.version
        print(f"[Run] Manual version: {version}")

    run_prefix = f"{version}_{size_tag}_{args.context_len}_stride_{args.train_stride}_{args.experiment}_{ts}"

    # ✅ Add parent folder: runs/<size_tag>/<run_prefix>
    parent_dir = Path("runs") / size_tag
    base_run_dir = parent_dir / run_prefix
    base_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Run] Root: {base_run_dir}")


    # ========================= lr decay for fine tuning
    base_lr = args.lr  # This is probably 1e-3 or 3e-4
    lr_decay = args.progressive_lr_decay
    trainer = None  # <=== IMPORTANT
    # ========= Experiment Switch =========

    if args.experiment == "none":
        model = None
        checkpoint_path = args.from_checkpoint
        cumulative_tokens_all_contracts = 0
        cumulative_effective_bars = 0
        cumulative_train_windows = 0
        global_tokens = 0
        # Track per-contract ECE 
        contract_eces = []
        optimizer_state = None


        for idx, data_file in enumerate(data_files, 1):
            remaining = args.token_budget - global_tokens if args.token_budget is not None else None


            if idx == 1:
                epochs = args.epochs  # First contract: explore fully
            else:
                epochs = max(int(0.6 * args.epochs), 5)  # Later contracts: 60% of base, minimum 5
            
            is_last = (idx == len(data_files))
            train_ds = BarDataset(
                data_file,
                context_len=args.context_len,
                stride=args.train_stride,
                is_validation=False,
                val_size=args.val_size,
                target_col=args.target_col, 
                bar_label_scale=args.bar_label_scale
            )

            # In training loop:
            if idx == 1:
                warmup_epochs = args.warmup_epochs  # Use warmup for first contract
            else:
                warmup_epochs = 0  # No warmup for subsequent contracts
            
            # Run comprehensive diagnostics
            diagnostics = diagnose_predictability(train_ds, name=f"Contract {idx}")

            # Decide whether to continue training
            #if diagnostics['score'] < 2:
            #    print(f"\n⚠️  WARNING: Contract {idx} has very poor trainability (score={diagnostics['score']}/6)")
            #    response = input("Continue training anyway? (y/n): ")
            #    if response.lower() != 'y':
            #        print(f"Skipping Contract {idx}")
            #        continue


            n_windows_epoch = len(train_ds)        # windows per epoch
            unique_bars = args.context_len + (n_windows_epoch - 1) * args.train_stride
            print(f"[Dataset] Effective unique bars (per epoch): {train_ds.effective_unique_bars()}")
            # After creating train_ds:
            all_train_labels = [int(train_ds.labels[i]) for i in range(len(train_ds.data))]
            train_counts = np.bincount(all_train_labels, minlength=3)
            train_pcts = train_counts / train_counts.sum() * 100
            print(f"[Class Balance] Down: {train_pcts[0]:.1f}% | Neutral: {train_pcts[1]:.1f}% | Up: {train_pcts[2]:.1f}%")


            print("\n[Feature Stats]")
            for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                feat_data = train_ds.data[:, i]
                print(f"  {col:20s}: mean={feat_data.mean():>8.4f}, "
                    f"std={feat_data.std():>8.4f}, "
                    f"min={feat_data.min():>8.4f}, "
                    f"max={feat_data.max():>8.4f}")

            # Check correlation between features and NEXT bar's label
            print("\n[Feature-Target Correlation] (Predictive)")
            if len(train_ds.labels) > 1:
                # For each feature at time t, check correlation with label at time t+1
                for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                    if len(train_ds.data) > 1:
                        # Features at bars [0, 1, ..., N-2]
                        feat_data = train_ds.data[:-1, i]
                        # Labels at bars [1, 2, ..., N-1] (next bar)
                        next_labels = train_ds.labels[1:]
                        
                        corr = np.corrcoef(feat_data, next_labels)[0, 1]
                        print(f"  {col:20s}: {corr:+.4f}")
                
                print("\n[Same-Bar Correlation] (For comparison - should be high for leakage detection)")
                for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                    # Features at bars [0, 1, ..., N-1]
                    feat_data = train_ds.data[:, i]
                    # Labels at bars [0, 1, ..., N-1] (same bar)
                    same_labels = train_ds.labels
                    
                    corr = np.corrcoef(feat_data, same_labels)[0, 1]
                    print(f"  {col:20s}: {corr:+.4f}")

            # ============ UPDATED AUTOCORR CHECK ============
            # Check label autocorrelation (bar_label is NO LONGER a feature)
            if len(train_ds.labels) > 1:
                past_labels = train_ds.labels[:-1]
                future_labels = train_ds.labels[1:]
                autocorr = np.corrcoef(past_labels, future_labels)[0, 1]
                print(f"\n[Label Autocorr] Correlation(bar_label_t, bar_label_t+1): {autocorr:.3f}")
                
                # Multi-lag analysis
                print("[Multi-lag Autocorr]")
                for lag in [2, 5, 10]:
                    if len(train_ds.labels) > lag:
                        past = train_ds.labels[:-lag]
                        future = train_ds.labels[lag:]
                        corr = np.corrcoef(past, future)[0, 1]
                        print(f"  Lag {lag:2d}: {corr:+.4f}")
            # ============ END UPDATE ============



            if is_last:
                raw_val = BarDataset(
                    data_file,
                    context_len=args.context_len,
                    stride=args.val_stride,
                    is_validation=True,
                    val_size=args.val_size,
                    target_col=args.target_col, 
                    bar_label_scale=args.bar_label_scale
                )

                val_ds = create_stratified_validation_subset(raw_val, samples_per_class=args.val_samples_per_class)
            else:
                val_ds = None

            if model is None:
                n_features = train_ds.n_features
                model = build_model(args.size, args.context_len, n_features, dropout=args.dropout)
                model = model.to(device)
                if checkpoint_path:
                    load_checkpoint_strict(model, checkpoint_path, device)
            else:
                print("[Model] Continuing existing model")

            num_workers = 0 if os.name == 'nt' else 2
            pin_memory = (device.type == 'cuda')
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=True, num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0 and pin_memory)
            )
           
            if val_ds is None:
                tmp_val = BarDataset(
                    data_file,
                    context_len=args.context_len,
                    stride=args.val_stride,
                    is_validation=True,
                    val_size=args.val_size,
                    target_col=args.target_col, 
                    bar_label_scale=args.bar_label_scale
                )

                tmp_val = create_stratified_validation_subset(
                    tmp_val, samples_per_class=min(args.val_samples_per_class, 512)
                )
                val_loader = DataLoader(
                    tmp_val, batch_size=args.batch_size,
                    shuffle=False, num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0 and pin_memory)
                )
            else:
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0 and pin_memory)
                )


            # Assuming dataset is your BarDataset instance for contract 6
            train_targets = np.array(train_ds.labels)  # or dataset.y / dataset.labels depending on your class

            vals, counts = np.unique(train_targets, return_counts=True)
            print(f"\n[DEBUG] TRAINING Label distribution seen by TRAINER for CONTRACT #{idx}:")
            total = counts.sum()
            for v, c in zip(vals, counts):
                print(f"  class {v}: {c} ({c/total:.2%})")



            # ==========================
            # IMPLICIT PER-CONTRACT LR DECAY
            # contract_index = idx
            # LR_i = base_lr * (0.35 ** (idx - 1))
            # ==========================
            base_lr = args.lr
            gamma = 0.35
            current_lr = base_lr * (gamma ** (idx - 1))
            print(f"[LR Schedule] Contract {idx}: lr = {current_lr:.6f}")
          
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                lr=current_lr, weight_decay=args.weight_decay, epochs=args.epochs,
                use_8bit_adam=not args.no_8bit_adam and HAS_8BIT,
                gradient_accumulation_steps=args.gradient_accumulation,
                label_smoothing=args.label_smoothing, early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_delta, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                lr_schedule=args.lr_schedule, warmup_epochs=args.warmup_epochs,
                token_budget=remaining, context_len_for_tokens=args.context_len,
                use_fbeta=args.use_fbeta,
                beta=args.beta,
                use_class_weights=args.use_class_weights,
                confidence_threshold=args.confidence_threshold
            )

            # === Reuse optimizer state from previous contract, if any ===
            if optimizer_state is not None:
                print("[Optimizer] Loading optimizer state from previous contract...")
                trainer.optimizer.load_state_dict(optimizer_state)
                # Override LR according to progressive schedule
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = current_lr
                print(f"[LR Schedule] Overrode optimizer LR to {current_lr:.6f}")

            # 🔁 Reset early-stopping state per contract
            if hasattr(trainer, "reset_early_stopping"):
                trainer.reset_early_stopping()
            else:
                # Fallback if you didn't add a helper method
                if hasattr(trainer, "best_metric"):
                    trainer.best_metric = None
                if hasattr(trainer, "no_improvement_epochs"):
                    trainer.no_improvement_epochs = 0

            if not hasattr(trainer, "optimizer"):
                raise RuntimeError("Trainer has no optimizer when attempting LR decay.")
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = current_lr
            print(f"[LR Schedule] Updated existing optimizer to lr={current_lr:.6f}")
 

            out_dir = base_run_dir / f"contract{idx}_{Path(data_file).stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            trainer.output_dir = str(out_dir)
            trainer.temp_checkpoint_path = str(out_dir / "checkpoint_temp.pt")
            trainer.final_model_path = str(out_dir / "model_final.pt")
            trainer.plot_curves_path = str(out_dir / "training_curves.png")
            trainer.plot_confusion_path = str(out_dir / "confusion_matrix.png")
            trainer.history_csv_path = str(out_dir / "training_history.csv")
            trainer.reset_early_stopping()
            trainer.train()
            # === Save optimizer state for next contract ===
            optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
            checkpoint_path = trainer.final_model_path
            # ---- Per-contract summary ----
            tokens_this_contract = trainer.history['tokens_seen'][-1]
            cumulative_tokens_all_contracts += tokens_this_contract

            # BarDataset info: only training split has these attrs
            effective_bars = getattr(train_ds, 'effective_unique_bars', None)
            train_bars = getattr(train_ds, 'n_bars', len(train_ds.data))
            train_windows = len(train_ds)

            # Correct retrieval
            effective_bars = (
                train_ds.effective_unique_bars() 
                if callable(train_ds.effective_unique_bars) 
                else train_ds.effective_unique_bars
            )
            cumulative_effective_bars += effective_bars
            cumulative_train_windows += train_windows

            print("\n[Contract Summary]")
            print(f"[Contract {idx}] File: {data_file}")
            print(f"[Contract {idx}] Train bars: {train_bars:,}")
            if effective_bars is not None:
                print(f"[Contract {idx}] Effective unique train bars (per epoch): {effective_bars:,}")
            print(f"[Contract {idx}] Train windows: {train_windows:,} | "
                  f"context_len={args.context_len} | train_stride={args.train_stride}")
            print(f"[Contract {idx}] Tokens seen this contract: {tokens_this_contract:,}")
            print(f"[Contract {idx}] Cumulative tokens seen so far: {cumulative_tokens_all_contracts:,}")

            # tokens seen on this contract = last entry of history
            tokens_this_contract = trainer.history['tokens_seen'][-1]
            epochs_run = len(trainer.history['train_loss'])
            windows_this_contract = n_windows_epoch * epochs_run

            total_tokens_global     += tokens_this_contract
            total_windows_global    += windows_this_contract
            total_unique_bars_global += unique_bars

            print(f"[Contract {idx} Summary] tokens={tokens_this_contract:,} | "
                f"windows={windows_this_contract:,} | unique_bars≈{unique_bars:,}")

            # === Per-contract ECE on this contract's validation set ===
            eval_trainer = Trainer(
                model=model, train_loader=val_loader, val_loader=val_loader, device=device,
                epochs=1, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                token_budget=None, context_len_for_tokens=args.context_len,
                use_fbeta=args.use_fbeta,
                beta=args.beta,
                use_class_weights=args.use_class_weights,
                confidence_threshold=args.confidence_threshold
            )
            _, _, _, vlab_c, vprob_c = eval_trainer.validate()
            ece_c = compute_ece(vprob_c, vlab_c, n_bins=15)
            contract_eces.append(ece_c)
            print(f"[ECE] Contract {idx}: ECE={ece_c:.4f} (experiment='none')")

            if args.token_budget is not None and global_tokens >= args.token_budget:
                print(f"[Scaling] Global token budget reached: {global_tokens} ≥ {args.token_budget}")
                break
            total_tokens_seen = trainer.history['tokens_seen'][-1]
            if total_tokens_seen >= args.token_budget:
                print("[Context Stress] Token budget reached during training")
                break

            if device.type == 'cuda':
                print(f"\n[GPU] Peak memory: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            gc.collect()
            print(f"\n[Contract {idx}] ✓ Complete")

        # === Plot ECE vs contract index for the baseline ("none") experiment ===
        if contract_eces:
            ece_dir = base_run_dir / "ece_baseline"
            ece_dir.mkdir(parents=True, exist_ok=True)
            ece_plot_path = ece_dir / "ece_by_contract.png"

            plt.figure()
            x_vals = np.arange(1, len(contract_eces) + 1)
            plt.plot(x_vals, contract_eces, marker='o')
            plt.xlabel("Contract index")
            plt.ylabel("ECE")
            plt.title("ECE by Contract - experiment='none'")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(ece_plot_path, bbox_inches="tight")
            plt.close()
            print(f"[ECE] Saved ECE-by-contract plot to {ece_plot_path}")

        print("\n" + "="*80)
        print("ALL TRAINING COMPLETE ✓")
        print("="*80)
        print(f"[Summary] Trained on {len(data_files)} contracts | runs at: {base_run_dir}")
        print(f"[Summary] Total effective unique train bars (approx): {cumulative_effective_bars:,}")
        print(f"[Summary] Total train windows across contracts: {cumulative_train_windows:,}")
        print(f"[Summary] Total tokens seen across all contracts: {cumulative_tokens_all_contracts:,}")
        print(f"[Summary] Runs root: {base_run_dir}")
        return


    # =========================
    # EXPERIMENTS
    # =========================

    first_file = data_files[0]
    last_file = data_files[-1]
    probe_train = BarDataset(
        first_file, context_len=args.context_len,
        stride=args.train_stride, is_validation=False,
        val_size=args.val_size,
        target_col=args.target_col, 
        bar_label_scale=args.bar_label_scale,
    )
    n_features = probe_train.n_features
    print(f"[Probe] n_features={n_features}, context_len={args.context_len}")
    raw_val_dataset = BarDataset(
        last_file, context_len=args.context_len,
        stride=args.val_stride, is_validation=True,
        val_size=args.val_size,
        target_col=args.target_col, 
        bar_label_scale=args.bar_label_scale
    )

    heldout_val = create_stratified_validation_subset(raw_val_dataset, samples_per_class=args.val_samples_per_class)
    num_workers = 0 if os.name == 'nt' else 0
    pin_memory = (device.type == 'cuda')
    heldout_loader = DataLoader(
        heldout_val, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory
    )

     # Per-experiment summary CSV
    summary_dir = base_run_dir / f"{args.experiment}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "experiment_summary.csv"

    if args.experiment == "scaling":
        
        # ========================= lr decay for continued pretraining
        base_lr = args.lr  # This is probably 1e-3 or 3e-4
        gamma = args.progressive_lr_decay
        current_lr = base_lr * (gamma ** (idx - 1))
        trainer = None  # <=== IMPORTANT
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        
        # To track per-size held-out metrics and ECE
        size_results = []
        # To track ECE-vs-contract curves per size
        size_ece_curves = {}
        
        for size_name in sizes:
            print("\n" + "#"*80)
            print(f"# SCALING: Size={size_name}")
            print("#"*80)
            model = build_model(size_name, args.context_len, n_features, dropout=args.dropout).to(device)
            total_tokens_seen = 0
            global_tokens = 0
            contract_eces = []   # <-- FIX: Initialize list before looping contracts
            optimizer_state = None
            for idx, data_file in enumerate(data_files, 1):

                # ========================= Progressive Learning Rate =========================
                # contract index = idx
                current_lr = base_lr * (lr_decay ** (idx - 1))

                print(f"\n[LR Schedule] Contract {idx}: lr = {current_lr:.6f} "
                    f"(base={base_lr:.6f}, decay={lr_decay})")

                # If this is NOT the first contract AND trainer already exists (same model, same optimizer)
                if idx > 1 and trainer is not None:
                    if not hasattr(trainer, "optimizer"):
                        raise RuntimeError("Trainer has no optimizer when attempting LR decay.")
                    for pg in trainer.optimizer.param_groups:
                        pg["lr"] = current_lr
                    print(f"[LR Schedule] Updated existing optimizer to lr={current_lr:.6f}")


                remaining = args.token_budget - global_tokens if args.token_budget is not None else None

                is_last = (idx == len(data_files))
                train_ds = BarDataset(
                    data_file,
                    context_len=args.context_len,
                    stride=args.train_stride,
                    is_validation=False,
                    val_size=args.val_size,
                    target_col=args.target_col, 
                    bar_label_scale=args.bar_label_scale
                )
                print(f"[Dataset] Effective unique bars (per epoch): {train_ds.effective_unique_bars()}")
                # After creating train_ds:
                all_train_labels = [int(train_ds.labels[i]) for i in range(len(train_ds.data))]
                train_counts = np.bincount(all_train_labels, minlength=3)
                train_pcts = train_counts / train_counts.sum() * 100
                print(f"[Class Balance] Down: {train_pcts[0]:.1f}% | Neutral: {train_pcts[1]:.1f}% | Up: {train_pcts[2]:.1f}%")
                
                print("\n[Feature Stats]")
                for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                    feat_data = train_ds.data[:, i]
                    print(f"  {col:20s}: mean={feat_data.mean():>8.4f}, "
                        f"std={feat_data.std():>8.4f}, "
                        f"min={feat_data.min():>8.4f}, "
                        f"max={feat_data.max():>8.4f}")

                # Check correlation between features and NEXT bar's label
                print("\n[Feature-Target Correlation] (Predictive)")
                if len(train_ds.labels) > 1:
                    # For each feature at time t, check correlation with label at time t+1
                    for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                        if len(train_ds.data) > 1:
                            # Features at bars [0, 1, ..., N-2]
                            feat_data = train_ds.data[:-1, i]
                            # Labels at bars [1, 2, ..., N-1] (next bar)
                            next_labels = train_ds.labels[1:]
                            
                            corr = np.corrcoef(feat_data, next_labels)[0, 1]
                            print(f"  {col:20s}: {corr:+.4f}")
                    
                    print("\n[Same-Bar Correlation] (For comparison - should be high for leakage detection)")
                    for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                        # Features at bars [0, 1, ..., N-1]
                        feat_data = train_ds.data[:, i]
                        # Labels at bars [0, 1, ..., N-1] (same bar)
                        same_labels = train_ds.labels
                        
                        corr = np.corrcoef(feat_data, same_labels)[0, 1]
                        print(f"  {col:20s}: {corr:+.4f}")
                # ============ UPDATED AUTOCORR CHECK ============
                # Check label autocorrelation (bar_label is NO LONGER a feature)
                if len(train_ds.labels) > 1:
                    past_labels = train_ds.labels[:-1]
                    future_labels = train_ds.labels[1:]
                    autocorr = np.corrcoef(past_labels, future_labels)[0, 1]
                    print(f"\n[Label Autocorr] Correlation(bar_label_t, bar_label_t+1): {autocorr:.3f}")
                    
                    # Multi-lag analysis
                    print("[Multi-lag Autocorr]")
                    for lag in [2, 5, 10]:
                        if len(train_ds.labels) > lag:
                            past = train_ds.labels[:-lag]
                            future = train_ds.labels[lag:]
                            corr = np.corrcoef(past, future)[0, 1]
                            print(f"  Lag {lag:2d}: {corr:+.4f}")

                #  Validation dataset
                if is_last:
                    val_ds = heldout_val
                else:
                    tmp = BarDataset(
                        data_file,
                        context_len=args.context_len,
                        stride=args.val_stride,
                        is_validation=True,
                        val_size=args.val_size,
                        target_col=args.target_col, 
                        bar_label_scale=args.bar_label_scale
                    )
                    val_ds = create_stratified_validation_subset(
                        tmp, samples_per_class=min(args.val_samples_per_class, 512)
                    )

                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=num_workers,
                    pin_memory=pin_memory
                )
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=num_workers,
                    pin_memory=pin_memory
                )

                # ========= create Trainer only once, then reuse optimizer =========

                # First contract for this size → create Trainer + optimizer

                trainer = Trainer(
                    model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                    lr=current_lr, weight_decay=args.weight_decay, epochs=args.epochs,
                    use_8bit_adam=not args.no_8bit_adam and HAS_8BIT,
                    gradient_accumulation_steps=args.gradient_accumulation,
                    label_smoothing=args.label_smoothing, early_stop_patience=args.early_stop_patience, early_stop_min_delta=args.early_stop_delta, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                    use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                    lr_schedule=args.lr_schedule, warmup_epochs=args.warmup_epochs,
                    token_budget=remaining,
                    context_len_for_tokens=args.context_len,
                    use_fbeta=args.use_fbeta,
                    beta=args.beta,
                    use_class_weights=args.use_class_weights,
                    confidence_threshold=args.confidence_threshold
                )

                # === Reuse optimizer state from previous contract, if any ===
                if optimizer_state is not None:
                    print("[Optimizer] Loading optimizer state from previous contract...")
                    trainer.optimizer.load_state_dict(optimizer_state)
                    # Override LR according to progressive schedule
                    for pg in trainer.optimizer.param_groups:
                        pg["lr"] = current_lr
                    print(f"[LR Schedule] Overrode optimizer LR to {current_lr:.6f}")

                if not hasattr(trainer, "optimizer"):
                    raise RuntimeError("Trainer has no optimizer when attempting LR decay.")
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = current_lr
                print(f"[LR Schedule] Updated existing optimizer to lr={current_lr:.6f}")
            
                    # 🔁 Reset early-stopping state per contract
                if hasattr(trainer, "reset_early_stopping"):
                    trainer.reset_early_stopping()
                else:
                    # Fallback if you didn't add a helper method
                    if hasattr(trainer, "best_metric"):
                        trainer.best_metric = None
                    if hasattr(trainer, "no_improvement_epochs"):
                        trainer.no_improvement_epochs = 0
                
                # ==================================================================



                out_dir = base_run_dir / f"scaling_{size_name}_{args.experiment}" / f"contract{idx}_{Path(data_file).stem}"
                out_dir.mkdir(parents=True, exist_ok=True)
                trainer.output_dir = str(out_dir)
                trainer.temp_checkpoint_path = str(out_dir / "checkpoint_temp.pt")
                trainer.final_model_path = str(out_dir / "model_final.pt")
                trainer.plot_curves_path = str(out_dir / "training_curves.png")
                trainer.plot_confusion_path = str(out_dir / "confusion_matrix.png")
                trainer.history_csv_path = str(out_dir / "training_history.csv")
                trainer.reset_early_stopping()
                trainer.train()
                # === Save optimizer state for next contract ===
                optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
                used_this_contract = trainer.history['tokens_seen'][-1]
                global_tokens += used_this_contract
                total_tokens_seen = trainer.history['tokens_seen'][-1]

                # === Per-contract ECE on this contract's validation set ===
                eval_trainer = Trainer(
                    model=model, train_loader=val_loader, val_loader=val_loader, device=device,
                    epochs=1, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                    token_budget=None, context_len_for_tokens=args.context_len,
                    use_fbeta=args.use_fbeta,
                    beta=args.beta,
                    use_class_weights=args.use_class_weights,
                    confidence_threshold=args.confidence_threshold
                )
                vloss_c, vacc_c, vpred_c, vlab_c, vprob_c = eval_trainer.validate()
                ece_c = compute_ece(vprob_c, vlab_c, n_bins=15)
                contract_eces.append(ece_c)
                print(f"[ECE] Contract {idx}: ECE={ece_c:.4f} (size={size_name})")

                if args.token_budget is not None and global_tokens >= args.token_budget:
                    print(f"[Scaling] Global token budget reached: {global_tokens} ≥ {args.token_budget}")
                    break
                total_tokens_seen = trainer.history['tokens_seen'][-1]
                if total_tokens_seen >= args.token_budget:
                    print(f"[Scaling] Token budget reached for size={size_name}")
                    break
                
                used_this_contract = trainer.history['tokens_seen'][-1]
                global_tokens += used_this_contract
                cumulative_tokens_all_contracts += used_this_contract

                # Bars in dataset (standard calculation)
                effective_bars = train_ds.effective_unique_bars() \
                    if callable(train_ds.effective_unique_bars) else train_ds.effective_unique_bars

                cumulative_effective_bars += effective_bars

                # Windows processed this contract
                epochs_run = len(trainer.history['train_loss'])
                windows_this_contract = len(train_ds) * epochs_run
                cumulative_train_windows += windows_this_contract


            # Store this size's ECE curve for the combined plot
            size_ece_curves[size_name] = contract_eces
        
            # === Plot ECE vs contract index for baseline ("none") experiment ===
            if contract_eces:
                ece_dir = base_run_dir / "scaling_summary"
                ece_dir.mkdir(parents=True, exist_ok=True)
                ece_plot_path = ece_dir / "ece_by_contract.png"

                plt.figure()
                x_vals = np.arange(1, len(contract_eces) + 1)
                plt.plot(x_vals, contract_eces, marker='o')
                plt.xlabel("Contract index")
                plt.ylabel("ECE")
                plt.title("ECE by Contract - experiment='scaling'")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(ece_plot_path, bbox_inches="tight")
                plt.close()
                print(f"[ECE] Saved ECE-by-contract plot to {ece_plot_path}")
           
            final_trainer = Trainer(
                model=model, train_loader=heldout_loader, val_loader=heldout_loader, device=device,
                epochs=1, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                token_budget=remaining, context_len_for_tokens=args.context_len,
                use_fbeta=args.use_fbeta,
                beta=args.beta,
                use_class_weights=args.use_class_weights,
                confidence_threshold=args.confidence_threshold
            )
            vloss, vacc, vpred, vlab, vprob = final_trainer.validate()
            f1 = f1_score(vlab, vpred, average='macro')
            prec = precision_score(vlab, vpred, average='macro', zero_division=0)
            rec = recall_score(vlab, vpred, average='macro', zero_division=0)
            
            
            # Optional: also compute held-out ECE
            heldout_ece = compute_ece(vprob, vlab, n_bins=15)
            print(f"[ECE] Held-out ECE (size={size_name}) = {heldout_ece:.4f}")

            write_runrow(summary_csv, {
                'experiment': 'scaling', 'size': size_name,
                'val_loss': vloss, 'val_acc': vacc,
                'macro_f1': f1, 'precision': prec, 'recall': rec,
                'heldout_ece': heldout_ece
            })


            size_results.append({
                'size': size_name,
                'val_loss': vloss,
                'val_acc': vacc,
                'macro_f1': f1,
                'precision': prec,
                'recall': rec,
                'heldout_ece': heldout_ece
            })
                            # === ONE combined ECE-vs-contract plot across all sizes ===
        if size_ece_curves:
            ece_plot_path = summary_dir / "ece_by_contract_all_sizes.png"
            plt.figure()
            for size_name, eces in size_ece_curves.items():
                if not eces:
                    continue
                x_vals = np.arange(1, len(eces) + 1)
                plt.plot(x_vals, eces, marker='o', label=size_name)
            plt.xlabel("Contract index")
            plt.ylabel("ECE")
            plt.title("ECE by Contract for Different Model Sizes")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(ece_plot_path, bbox_inches="tight")
            plt.close()
            print(f"[ECE] Saved combined ECE-by-contract plot to {ece_plot_path}")

    elif args.experiment == "context_stress":
        model = build_model(args.size, args.context_len, n_features, dropout=args.dropout).to(device)
        total_tokens_seen = 0
        global_tokens = 0
        contract_eces = []   # <-- FIX: Initialize list before looping contracts
        optimizer_state = None
        for idx, data_file in enumerate(data_files, 1):
            remaining = args.token_budget - global_tokens if args.token_budget is not None else None
            current_lr = base_lr * (lr_decay ** (idx - 1))
            print(f"\n[LR Schedule] context_stress Contract {idx}: lr = {current_lr:.6f}")

            train_ds = BarDataset(
                data_file, context_len=args.context_len,
                stride=args.train_stride, is_validation=False, target_col=args.target_col, 
                val_size=args.val_size, bar_label_scale=args.bar_label_scale
            )
            print(f"[Dataset] Effective unique bars (per epoch): {train_ds.effective_unique_bars()}")
            # After creating train_ds:
            all_train_labels = [int(train_ds.labels[i]) for i in range(len(train_ds.data))]
            train_counts = np.bincount(all_train_labels, minlength=3)
            train_pcts = train_counts / train_counts.sum() * 100
            print(f"[Class Balance] Down: {train_pcts[0]:.1f}% | Neutral: {train_pcts[1]:.1f}% | Up: {train_pcts[2]:.1f}%")
            
            print("\n[Feature Stats]")
            for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                feat_data = train_ds.data[:, i]
                print(f"  {col:20s}: mean={feat_data.mean():>8.4f}, "
                    f"std={feat_data.std():>8.4f}, "
                    f"min={feat_data.min():>8.4f}, "
                    f"max={feat_data.max():>8.4f}")
                
            # Check correlation between features and NEXT bar's label
            print("\n[Feature-Target Correlation] (Predictive)")
            if len(train_ds.labels) > 1:
                # For each feature at time t, check correlation with label at time t+1
                for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                    if len(train_ds.data) > 1:
                        # Features at bars [0, 1, ..., N-2]
                        feat_data = train_ds.data[:-1, i]
                        # Labels at bars [1, 2, ..., N-1] (next bar)
                        next_labels = train_ds.labels[1:]
                        
                        corr = np.corrcoef(feat_data, next_labels)[0, 1]
                        print(f"  {col:20s}: {corr:+.4f}")
                
                print("\n[Same-Bar Correlation] (For comparison - should be high for leakage detection)")
                for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                    # Features at bars [0, 1, ..., N-1]
                    feat_data = train_ds.data[:, i]
                    # Labels at bars [0, 1, ..., N-1] (same bar)
                    same_labels = train_ds.labels
                    
                    corr = np.corrcoef(feat_data, same_labels)[0, 1]
                    print(f"  {col:20s}: {corr:+.4f}")
            # ============ UPDATED AUTOCORR CHECK ============
            # Check label autocorrelation (bar_label is NO LONGER a feature)
            if len(train_ds.labels) > 1:
                past_labels = train_ds.labels[:-1]
                future_labels = train_ds.labels[1:]
                autocorr = np.corrcoef(past_labels, future_labels)[0, 1]
                print(f"\n[Label Autocorr] Correlation(bar_label_t, bar_label_t+1): {autocorr:.3f}")
                
                # Multi-lag analysis
                print("[Multi-lag Autocorr]")
                for lag in [2, 5, 10]:
                    if len(train_ds.labels) > lag:
                        past = train_ds.labels[:-lag]
                        future = train_ds.labels[lag:]
                        corr = np.corrcoef(past, future)[0, 1]
                        print(f"  Lag {lag:2d}: {corr:+.4f}")
            # ============ END UPDATE ============
            tmp_val = BarDataset(
                data_file, context_len=args.context_len,
                stride=args.val_stride, is_validation=True,
                target_col=args.target_col, 
                val_size=args.val_size, bar_label_scale=args.bar_label_scale
            )
            val_ds = create_stratified_validation_subset(
                tmp_val, samples_per_class=min(args.val_samples_per_class, 512)
            )
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=True, num_workers=num_workers,
                pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size,
                shuffle=False, num_workers=num_workers,
                pin_memory=pin_memory
            )

            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                lr=current_lr, weight_decay=args.weight_decay, epochs=epochs,
                use_8bit_adam=not args.no_8bit_adam and HAS_8BIT,
                gradient_accumulation_steps=args.gradient_accumulation,
                label_smoothing=args.label_smoothing, early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_delta, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                lr_schedule=args.lr_schedule, warmup_epochs=args.warmup_epochs,
                token_budget=args.token_budget - total_tokens_seen,
                context_len_for_tokens=args.context_len,
                use_fbeta=args.use_fbeta,
                beta=args.beta,
                use_class_weights=args.use_class_weights,
                confidence_threshold=args.confidence_threshold
            )

            # === Reuse optimizer state from previous contract, if any ===
            if optimizer_state is not None:
                print("[Optimizer] Loading optimizer state from previous contract...")
                trainer.optimizer.load_state_dict(optimizer_state)
                # Override LR according to progressive schedule
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = current_lr
                print(f"[LR Schedule] Overrode optimizer LR to {current_lr:.6f}")

            # 🔁 Reset early-stopping state per contract
            if hasattr(trainer, "reset_early_stopping"):
                trainer.reset_early_stopping()
            else:
                # Fallback if you didn't add a helper method
                if hasattr(trainer, "best_metric"):
                    trainer.best_metric = None
                if hasattr(trainer, "no_improvement_epochs"):
                    trainer.no_improvement_epochs = 0

            out_dir = base_run_dir / f"context_train" / f"contract{idx}_{Path(data_file).stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            trainer.output_dir = str(out_dir)
            trainer.temp_checkpoint_path = str(out_dir / "checkpoint_temp.pt")
            trainer.final_model_path = str(out_dir / "model_final.pt")
            trainer.plot_curves_path = str(out_dir / "training_curves.png")
            trainer.plot_confusion_path = str(out_dir / "confusion_matrix.png")
            trainer.history_csv_path = str(out_dir / "training_history.csv")
            trainer.reset_early_stopping()
            trainer.train()
            # === Save optimizer state for next contract ===
            optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
            used_this_contract = trainer.history['tokens_seen'][-1]
            global_tokens += used_this_contract

            # === Per-contract ECE on this contract's validation set ===
            eval_trainer = Trainer(
                model=model, train_loader=val_loader, val_loader=val_loader, device=device,
                epochs=1, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                token_budget=None, context_len_for_tokens=args.context_len,
                use_fbeta=args.use_fbeta,
                beta=args.beta,
                use_class_weights=args.use_class_weights,
                confidence_threshold=args.confidence_threshold
            )
            _, _, _, vlab_c, vprob_c = eval_trainer.validate()
            ece_c = compute_ece(vprob_c, vlab_c, n_bins=15)
            contract_eces.append(ece_c)
            print(f"[ECE] Contract {idx}: ECE={ece_c:.4f} (context_stress)")

            if args.token_budget is not None and global_tokens >= args.token_budget:
                print(f"[Scaling] Global token budget reached: {global_tokens} ≥ {args.token_budget}")
                break
            total_tokens_seen = trainer.history['tokens_seen'][-1]
            if total_tokens_seen >= args.token_budget:
                print("[Context Stress] Token budget reached during training")
                break
            
        # === Plot ECE vs contract index for context_stress ===
        if contract_eces:
            ctx_train_dir = base_run_dir / "context_train" 
            ctx_train_dir.mkdir(parents=True, exist_ok=True)
            ece_plot_path = ctx_train_dir / "ece_by_contract.png"

            plt.figure()
            x_vals = np.arange(1, len(contract_eces) + 1)
            plt.plot(x_vals, contract_eces, marker='o')
            plt.xlabel("Contract index")
            plt.ylabel("ECE")
            plt.title("ECE by Contract - context_stress")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(ece_plot_path, bbox_inches="tight")
            plt.close()
            print(f"[ECE] Saved ECE-by-contract plot to {ece_plot_path}")

        ctxs = [int(x.strip()) for x in args.eval_contexts.split(",") if x.strip()]
        ctx_dir = base_run_dir / "context_stress_eval"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        evaluate_context_lengths(model, device, last_file, ctx_dir, args.batch_size, args.val_size, ctxs, bar_label_scale=args.bar_label_scale)

    elif args.experiment == "ablations":
        # ========================= lr decay for continued pretraining
        base_lr = args.lr  # This is probably 1e-3 or 3e-4
        gamma = args.progressive_lr_decay

        trainer = None  # <=== IMPORTANT
        ablate_list = [a.strip().lower() for a in args.ablate.split(",") if a.strip()]
        variants = [
            ('full', dict(
                use_gqa=True, use_alibi=True, use_learned_pos=True,
                norm_style="rms_pre", ffn_style="swiglu",
                label_smoothing=args.label_smoothing
            ))
        ]
        if 'alibi' in ablate_list:
            variants.append(('minus_alibi', dict(
                use_gqa=True, use_alibi=False, use_learned_pos=True, 
                norm_style="rms_pre", ffn_style="swiglu",
                label_smoothing=args.label_smoothing
            )))
        if 'gqa' in ablate_list:
            variants.append(('minus_gqa', dict(
                use_gqa=False, use_alibi=True,
                norm_style="rms_pre", ffn_style="swiglu",
                label_smoothing=args.label_smoothing
            )))
        if 'rmsnorm' in ablate_list:
            variants.append(('minus_rmsnorm_postln', dict(
                use_gqa=True, use_alibi=True,
                norm_style="layer_post", ffn_style="swiglu",
                label_smoothing=args.label_smoothing
            )))
        if 'swiglu' in ablate_list:
            variants.append(('minus_swiglu_relu', dict(
                use_gqa=True, use_alibi=True,
                norm_style="rms_pre", ffn_style="relu",
                label_smoothing=args.label_smoothing
            )))
        if 'labelsmoothing' in ablate_list:
            variants.append(('minus_labelsmoothing', dict(
                use_gqa=True, use_alibi=True,
                norm_style="rms_pre", ffn_style="swiglu",
                label_smoothing=0.0
            )))

        # Track held-out performance per variant for summary plots
        variant_results = []

        for name, cfg in variants:
            print("\n" + "#"*80)
            print(f"# ABLATION: {name} | cfg={cfg}")
            print("#"*80)
            model = build_model(
                args.size, args.context_len, n_features, dropout=args.dropout,
                use_gqa=cfg['use_gqa'], use_alibi=cfg['use_alibi'], use_learned_pos=True,
                norm_style=cfg['norm_style'], ffn_style=cfg['ffn_style']
            ).to(device)
            total_tokens_seen = 0
            global_tokens = 0
            contract_eces = []   # <-- FIX: Initialize list before looping contracts
            optimizer_state = None
            for idx, data_file in enumerate(data_files, 1):
                current_lr = base_lr * (gamma ** (idx - 1))
                train_ds = BarDataset(
                    data_file, context_len=args.context_len,
                    stride=args.train_stride, is_validation=False,
                    target_col=args.target_col,
                    val_size=args.val_size, bar_label_scale=args.bar_label_scale
                )
                print(f"[Dataset] Effective unique bars (per epoch): {train_ds.effective_unique_bars()}")
                # After creating train_ds:
                all_train_labels = [int(train_ds.labels[i]) for i in range(len(train_ds.data))]
                train_counts = np.bincount(all_train_labels, minlength=3)
                train_pcts = train_counts / train_counts.sum() * 100
                print(f"[Class Balance] Down: {train_pcts[0]:.1f}% | Neutral: {train_pcts[1]:.1f}% | Up: {train_pcts[2]:.1f}%")
                
                print("\n[Feature Stats]")
                for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                    feat_data = train_ds.data[:, i]
                    print(f"  {col:20s}: mean={feat_data.mean():>8.4f}, "
                        f"std={feat_data.std():>8.4f}, "
                        f"min={feat_data.min():>8.4f}, "
                        f"max={feat_data.max():>8.4f}")
                    
                # Check correlation between features and NEXT bar's label
                print("\n[Feature-Target Correlation] (Predictive)")
                if len(train_ds.labels) > 1:
                    # For each feature at time t, check correlation with label at time t+1
                    for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                        if len(train_ds.data) > 1:
                            # Features at bars [0, 1, ..., N-2]
                            feat_data = train_ds.data[:-1, i]
                            # Labels at bars [1, 2, ..., N-1] (next bar)
                            next_labels = train_ds.labels[1:]
                            
                            corr = np.corrcoef(feat_data, next_labels)[0, 1]
                            print(f"  {col:20s}: {corr:+.4f}")
                    
                    print("\n[Same-Bar Correlation] (For comparison - should be high for leakage detection)")
                    for i, col in enumerate(BAR_DESIRED_COLUMN_ORDER):
                        # Features at bars [0, 1, ..., N-1]
                        feat_data = train_ds.data[:, i]
                        # Labels at bars [0, 1, ..., N-1] (same bar)
                        same_labels = train_ds.labels
                        
                        corr = np.corrcoef(feat_data, same_labels)[0, 1]
                        print(f"  {col:20s}: {corr:+.4f}")

                # ============ UPDATED AUTOCORR CHECK ============
                # Check label autocorrelation (bar_label is NO LONGER a feature)
                if len(train_ds.labels) > 1:
                    past_labels = train_ds.labels[:-1]
                    future_labels = train_ds.labels[1:]
                    autocorr = np.corrcoef(past_labels, future_labels)[0, 1]
                    print(f"\n[Label Autocorr] Correlation(bar_label_t, bar_label_t+1): {autocorr:.3f}")
                    
                    # Multi-lag analysis
                    print("[Multi-lag Autocorr]")
                    for lag in [2, 5, 10]:
                        if len(train_ds.labels) > lag:
                            past = train_ds.labels[:-lag]
                            future = train_ds.labels[lag:]
                            corr = np.corrcoef(past, future)[0, 1]
                            print(f"  Lag {lag:2d}: {corr:+.4f}")
                # ============ END UPDATE ============
                tmp_val = BarDataset(
                    data_file, context_len=args.context_len,
                    stride=args.val_stride, is_validation=True,
                    target_col=args.target_col, 
                    val_size=args.val_size, bar_label_scale=args.bar_label_scale
                )
                val_ds = create_stratified_validation_subset(
                    tmp_val, samples_per_class=min(args.val_samples_per_class, 512)
                )
                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=num_workers,
                    pin_memory=pin_memory
                )
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=num_workers,
                    pin_memory=pin_memory
                )

                trainer = Trainer(
                    model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                    lr=current_lr, weight_decay=args.weight_decay, epochs=args.epochs,
                    use_8bit_adam=not args.no_8bit_adam and HAS_8BIT,
                    gradient_accumulation_steps=args.gradient_accumulation,
                    label_smoothing=args.label_smoothing, early_stop_patience=args.early_stop_patience,
                    early_stop_min_delta=args.early_stop_delta, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                    use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                    lr_schedule=args.lr_schedule, warmup_epochs=args.warmup_epochs,
                    token_budget=args.token_budget - total_tokens_seen,
                    context_len_for_tokens=args.context_len,
                    use_fbeta=args.use_fbeta,
                    beta=args.beta,
                    use_class_weights=args.use_class_weights,
                    confidence_threshold=args.confidence_threshold
                )

                # === Reuse optimizer state from previous contract, if any ===
                if optimizer_state is not None:
                    print("[Optimizer] Loading optimizer state from previous contract...")
                    trainer.optimizer.load_state_dict(optimizer_state)
                    # Override LR according to progressive schedule
                    for pg in trainer.optimizer.param_groups:
                        pg["lr"] = current_lr
                    print(f"[LR Schedule] Overrode optimizer LR to {current_lr:.6f}")


                # 🔁 Reset early-stopping state per contract
                if hasattr(trainer, "reset_early_stopping"):
                    trainer.reset_early_stopping()
                else:
                    # Fallback if you didn't add a helper method
                    if hasattr(trainer, "best_metric"):
                        trainer.best_metric = None
                    if hasattr(trainer, "no_improvement_epochs"):
                        trainer.no_improvement_epochs = 0

                out_dir = base_run_dir / f"ablations_{name}" / f"contract{idx}_{Path(data_file).stem}"
                out_dir.mkdir(parents=True, exist_ok=True)
                trainer.output_dir = str(out_dir)
                trainer.temp_checkpoint_path = str(out_dir / "checkpoint_temp.pt")
                trainer.final_model_path = str(out_dir / "model_final.pt")
                trainer.plot_curves_path = str(out_dir / "training_curves.png")
                trainer.plot_confusion_path = str(out_dir / "confusion_matrix.png")
                trainer.history_csv_path = str(out_dir / "training_history.csv")
                trainer.reset_early_stopping()
                trainer.train()
                # === Save optimizer state for next contract ===
                optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
                used_this_contract = trainer.history['tokens_seen'][-1]
                global_tokens += used_this_contract


                # === Per-contract ECE on this contract's validation set ===
                eval_trainer = Trainer(
                    model=model, train_loader=val_loader, val_loader=val_loader, device=device,
                    epochs=1, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                    token_budget=None, context_len_for_tokens=args.context_len,
                    use_fbeta=args.use_fbeta,
                    beta=args.beta,
                    use_class_weights=args.use_class_weights,
                    confidence_threshold=args.confidence_threshold
                )
                _, _, _, vlab_c, vprob_c = eval_trainer.validate()
                ece_c = compute_ece(vprob_c, vlab_c, n_bins=15)
                contract_eces.append(ece_c)
                print(f"[ECE] Contract {idx}: ECE={ece_c:.4f} ({name})")


                if args.token_budget is not None and global_tokens >= args.token_budget:
                    print(f"[Scaling] Global token budget reached: {global_tokens} ≥ {args.token_budget}")
                    break
                total_tokens_seen = trainer.history['tokens_seen'][-1]
                if total_tokens_seen >= args.token_budget:
                    print(f"[Ablation {name}] Token budget reached.")
                    break
                    # === Plot ECE vs contract index ===

            if contract_eces:
                ctx_train_dir = base_run_dir / "context_train"
                ctx_train_dir.mkdir(parents=True, exist_ok=True)
                ece_plot_path = ctx_train_dir / "ece_by_contract.png"

                plt.figure()
                x_vals = np.arange(1, len(contract_eces) + 1)
                plt.plot(x_vals, contract_eces, marker='o')
                plt.xlabel("Contract index")
                plt.ylabel("ECE")
                plt.title(f"ECE by Contract - {args.experiment}")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(ece_plot_path, bbox_inches="tight")
                plt.close()
                print(f"[ECE] Saved ECE-by-contract plot to {ece_plot_path}")

                if not hasattr(trainer, "optimizer"):
                    raise RuntimeError("Trainer has no optimizer when attempting LR decay.")
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = current_lr
                print(f"[LR Schedule] Updated existing optimizer to lr={current_lr:.6f}")
    

            eval_tr = Trainer(
                model=model, train_loader=heldout_loader, val_loader=heldout_loader, device=device,
                epochs=1, amp_dtype=amp_dtype, grad_clip=args.grad_clip,
                token_budget=remaining, context_len_for_tokens=args.context_len,
                use_fbeta=args.use_fbeta,
                beta=args.beta,
                use_class_weights=args.use_class_weights,
                confidence_threshold=args.confidence_threshold
            )
            vloss, vacc, vpred, vlab, vprob = eval_tr.validate()
            f1 = f1_score(vlab, vpred, average='macro')
            prec = precision_score(vlab, vpred, average='macro', zero_division=0)
            rec = recall_score(vlab, vpred, average='macro', zero_division=0)
            write_runrow(summary_csv, {
                'experiment': 'ablations', 'variant': name,
                'val_loss': vloss, 'val_acc': vacc,
                'macro_f1': f1, 'precision': prec, 'recall': rec,
                'heldout_ece': heldout_ece
            })

            # Store for final plots
            variant_results.append({
                'variant': name,
                'val_loss': vloss,
                'val_acc': vacc,
                'macro_f1': f1,
                'precision': prec,
                'recall': rec,
                'heldout_ece': heldout_ece
            })

            
        # === Summary plots across variants (metrics + ECE) ===
        if variant_results:
            summary_dir = base_run_dir / "ablations_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            variant_names = [r['variant'] for r in variant_results]
            # Metrics to visualize
            metrics_to_plot = ['val_acc', 'macro_f1', 'precision', 'recall', 'heldout_ece']
            ylabels = {
                'val_acc': 'Validation Accuracy',
                'macro_f1': 'Macro F1',
                'heldout_ece': 'Held-out ECE'
            }

            for metric in metrics_to_plot:
                values = [r[metric] for r in variant_results]
                plt.figure()
                x = np.arange(len(variant_names))
                plt.bar(x, values)
                plt.xticks(x, variant_names, rotation=30, ha='right')
                plt.ylabel(ylabels[metric])
                plt.title(f"{ylabels[metric]} by Variant")
                plt.tight_layout()
                plot_path = summary_dir / f"{metric}_by_variant.png"
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
                print(f"[Summary Plot] Saved {metric} by variant to {plot_path}")

        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE ✓")
        print("="*80)
        print(f"[Summary] Results → {base_run_dir / {args.experiment} / 'experiment_summary.csv'}")
        print(f"[Summary] Root folder: {base_run_dir}_{args.experiment}")

    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE ✓")
    print("="*80)
    print(f"[Summary] Trained on {len(data_files)} contracts | runs at: {base_run_dir}")
    print(f"[Summary] Total effective unique train bars (approx): {cumulative_effective_bars:,}")
    print(f"[Summary] Total train windows across contracts: {cumulative_train_windows:,}")
    print(f"[Summary] Total tokens seen across all contracts: {cumulative_tokens_all_contracts:,}")
    print(f"[Summary] Runs root: {base_run_dir}")
    return

if __name__ == "__main__":
    main()
