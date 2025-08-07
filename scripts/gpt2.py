import torch 
import torch.nn as nn
from einops import einsum, rearrange

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps 
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return (x_norm * self.scale) + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["d_model"], cfg["d_ff"])
        self.fc2 = nn.Linear(cfg["d_ff"], cfg["d_model"])
        self.gelu = nn.functional.gelu

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, ctx_len, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads!"

        self.head_dim = d_out // n_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape
        
        queries = self.W_q(x) # (batch_size, seq_len, d_out)
        keys = self.W_k(x)
        values = self.W_v(x)

        queries = rearrange(queries, "batch_size seq_len (n_heads head_dim) -> batch_size n_heads seq_len head_dim", head_dim=self.head_dim)
        keys = rearrange(keys, "batch_size seq_len (n_heads head_dim) -> batch_size n_heads seq_len head_dim", head_dim=self.head_dim)
        values = rearrange(values, "batch_size seq_len (n_heads head_dim) -> batch_size n_heads seq_len head_dim", head_dim=self.head_dim)

        attn_scores = einsum(queries, keys, "... s1 head_dim, ... s2 head_dim -> ... s1 s2")
        attn_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vecs = einsum(attn_weights, values, "... s1 s2, ... s2 head_dim -> ... s1 head_dim")
        context_vecs = rearrange(context_vecs, "batch_size n_heads seq_len head_dim -> batch_size seq_len (n_heads head_dim)")
        context_vecs = self.out_proj(context_vecs)
        return context_vecs

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mha = MultiHeadAttention(cfg["d_model"], cfg["d_model"], cfg["n_heads"], cfg["ctx_len"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["d_model"])
        self.norm2 = LayerNorm(cfg["d_model"])

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["d_model"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["d_model"])
        self.out_head = nn.Linear(cfg["d_model"], cfg["vocab_size"])

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len))
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits