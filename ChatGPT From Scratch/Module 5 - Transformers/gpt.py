
# gpt.py by Andrew Taylor
import torch
from linear import CustomLinear
from mha import CustomMHA

'''
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use pytorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).

For layer norm, you just need to pass in D-model: self.norm1 = torch.nn.LayerNorm((d_model,))

'''
class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # Layer normalization layers
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.mha = CustomMHA(d_model, n_heads)
        
        # Two-layer MLP 
        # typically, the hidden ddimension is 4x the model dimension as Ted said in class
        self.mlp = torch.nn.Sequential(
            CustomLinear(d_model, 4 * d_model),
            torch.nn.ReLU(),
            CustomLinear(4 * d_model, d_model)
        )
        
        # dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)

    
    def forward(self, x):
        # First layer multi-head attention with residual connection
        # apply layer norm then attention 
        attn_output = self.mha(self.norm1(x))
        x = x + self.dropout(attn_output)  # Residual connection
        
        # Second layer: feed-forward MLP with residual connection
        # Apply layer norm before MLP
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)  # Residual connection
        
        return x


'''
Create a full GPT model which has two embeddings (token and position),
and then has a series of transformer block instances (layers). Finally, the last 
layer should project outputs to size [vocab_size].
'''
class GPTModel(torch.nn.Module):

    
    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings: maps token IDs to d_model 
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        
        # position embedding: maps position indices to d_model
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        
        # Dropout applied after 
        self.dropout = torch.nn.Dropout(dropout)
        
        # Stack of transformer decoder blocks
        # I';m using ModuleList to properly register all modules
        self.blocks = torch.nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, dropout)
            for _ in range(layers)])
        
        # last layer norm
        self.final_norm = torch.nn.LayerNorm(d_model)
        
        # Output projection to vocabsize
        self.lm_head = CustomLinear(d_model, vocab_size)

    
    def forward(self, x):
        B, S = x.shape
        
        # Create position indices
        # Shape: (S,)
        positions = torch.arange(0, S, device=x.device)
        
        # Get token embeddings
        # Shape: (B, S, d_model)
        token_embeds = self.token_embedding(x)
        
        # Get position embeddings
        # Shape: (S, d_model) - will broadcast across batch dim
        pos_embeds = self.position_embedding(positions)
        
        # Combine token and position embeddings
        # Shape: (B, S, d_model)
        x = token_embeds + pos_embeds
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary size to get logits
        # Shape: (B, S, vocab_size)
        logits = self.lm_head(x)
        
        return logits


if __name__ == "__main__":

    # example of building the model and doing a forward pass
    D = 128
    H = 8
    L = 4
    model = GPTModel(D, H, L, 1000, 512)
    B = 32
    S = 48 # this can be less than 512, it just cant be more than 512
    x = torch.randint(1000, (B, S))
    y = model(x) # this should give us logits over the vocab for all positions

    # should be size (B, S, 1000)
    print(f"Output shape: {y.shape}")
    print(f"Expected shape: torch.Size([{B}, {S}, 1000])")
    print(f"Shapes match: {y.shape == torch.Size([B, S, 1000])}")