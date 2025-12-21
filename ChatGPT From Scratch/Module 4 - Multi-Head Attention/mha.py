import torch
import math

import torch
import math

class CustomMHA(torch.nn.Module):
    '''
    param d_model : (int) the length of vectors used in this model
    param n_heads : (int) the number of attention heads. You can assume that
        this evenly divides d_model.
    '''
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Parameters
        self.W_qkv = torch.nn.Parameter(torch.empty(3 * d_model, d_model))
        self.W_o   = torch.nn.Parameter(torch.empty(d_model, d_model))

        # Init weights
        torch.nn.init.xavier_uniform_(self.W_qkv)
        torch.nn.init.xavier_uniform_(self.W_o)

    '''
    param x : (tensor) an input batch, with size (B, S, D)
    returns : a tensor of the same size, which has had MHA computed for each batch entry.
    '''
    def forward(self, x):
        B, S, D = x.shape
        H, d_h = self.n_heads, self.d_head

        # (B,S,D) @ (D,3D) -> (B,S,3D)
        T = torch.matmul(x, self.W_qkv.T)

        # split into Q,K,V: each (B,S,D)
        Q, K, V = T.chunk(3, dim=-1)

        # reshape into (B,H,S,d_h)
        Q = Q.view(B, S, H, d_h).transpose(1, 2)  # (B,H,S,d_h)
        K = K.view(B, S, H, d_h).transpose(1, 2)
        V = V.view(B, S, H, d_h).transpose(1, 2)

        # attention scores: (B,H,S,S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        attn = torch.softmax(scores, dim=-1)

        # output per head: (B,H,S,d_h)
        out = torch.matmul(attn, V)

        # reshape back: (B,S,D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # final projection (B,S,D)
        Y = torch.matmul(out, self.W_o.T)
        return Y



if __name__ == "__main__":

	# example of building and running this class
	mha = CustomMHA(128,8)

	# 32 samples of length 6 each, with d_model at 128
	x = torch.randn((32,6,128))
	y = mha(x)
	print(x.shape, y.shape) # should be the same