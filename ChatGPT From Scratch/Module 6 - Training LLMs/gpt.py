import torch
import math
import torch, math
import torch.nn.functional as F

class CustomLinear(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((output_size, input_size)))
		self.bias = torch.nn.Parameter(torch.zeros((output_size,)))

	def forward(self, x):
		return x @ self.weight.T + self.bias


class CustomEmbedding(torch.nn.Module):
	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((num_embeddings, embedding_dim)))

	def forward(self, x):
		return self.weight[x]


class CustomMHA(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
        self.wo  = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

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

        if added_batch: y = y[0]
        return y


class TransformerDecoderBlock(torch.nn.Module):
	def __init__(self, d_model, n_heads):
		super().__init__()
		self.norm1 = torch.nn.LayerNorm((d_model,))
		self.mha   = CustomMHA(d_model, n_heads)
		self.norm2 = torch.nn.LayerNorm((d_model,))
		self.fc1   = CustomLinear(d_model, 4*d_model)
		self.act   = torch.nn.ReLU()
		self.fc2   = CustomLinear(4*d_model, d_model)
		self.dropout = torch.nn.Dropout(0.1)

	def forward(self, x, causal_mask_ref=None):
		# I hand the precomputed mask to the MHA each call.
		self.mha._causal_mask_ref = causal_mask_ref
		x = x + self.mha(self.norm1(x))
		x = x + self.dropout(self.fc2(self.act(self.fc1(self.norm2(x)))))
		return x
		
class GPTModel(torch.nn.Module):
    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.word_embeddings = CustomEmbedding(vocab_size, d_model)
        self.position_embeddings = CustomEmbedding(max_seq_len, d_model)
        self.layers = torch.nn.ModuleList([TransformerDecoderBlock(d_model, n_heads) for _ in range(layers)])
        self.fc_out = CustomLinear(d_model, vocab_size)
        # placeholders; I allocate on GPU later
        self.register_buffer("pos_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def prepare_buffers(self, device):
        # Build on GPU once; zero CPU ↔ GPU traffic in forward.
        self.pos_idx = torch.arange(self.max_seq_len, dtype=torch.long, device=device)

    def forward(self, x):
        B, S = x.shape
        positions = self.pos_idx[:S].unsqueeze(0).expand(B, -1)  # (B, S) on GPU
        x = self.word_embeddings(x) + self.position_embeddings(positions)
        for layer in self.layers:
            x = layer(x)  # SDPA handles causality; no mask passed
        return self.fc_out(x)


if __name__ == "__main__":
	model = GPTModel(256, 4, 4, 1000, 1024)
	B, S = 8, 32
	x = torch.randint(1000, (B, S))
	print("ok", model(x).shape)
