import torch 
import torch.nn as nn 
from einops import einsum, rearrange

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.fc2 = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.fc3 = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)

    def forward(self, x):
        return self.fc3(nn.functional.silu(self.fc2(x)) * self.fc1(x))

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * self.scale

class RoPE(nn.Module):
    def __init__(self, head_dim, ctx_len, theta_base=10_000):
        super().__init__()
        self.head_dim = head_dim
        self.ctx_len = ctx_len
        self.theta_base = theta_base
        self._compute_rope()

    def _compute_rope(self):
        inv_freq = 1.0 / (self.theta_base ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        pos = torch.arange(self.ctx_len)
        angles = einsum(pos, inv_freq, "f,p -> f p") # (ctx_len, head_dim // 2)
        angles = torch.cat((angles, angles), dim=1)

        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rotate_half(self, x):
        x1 = x[...,:x.shape[-1]//2]
        x2 = x[...,x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        batch_size, n_heads, seq_len, head_dim = x.shape
        cos = self.cos[None,None,:seq_len,:]
        sin = self.sin[None,None,:seq_len,:]
        return x * cos + self._rotate_half(x) * sin

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, ctx_len):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads!"

        self.head_dim = d_out // n_heads

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        self.rope = RoPE(self.head_dim, ctx_len)

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

        queries = self.rope(queries)
        keys = self.rope(keys)

        attn_scores = einsum(queries, keys, "... s1 head_dim, ... s2 head_dim -> ... s1 s2")
        attn_scores.masked_fill_(self.mask.bool()[:seq_len,:seq_len], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vecs = einsum(attn_weights, values, "... s1 s2, ... s2 head_dim -> ... s1 head_dim")
        context_vecs = rearrange(context_vecs, "batch_size n_heads seq_len head_dim -> batch_size seq_len (n_heads head_dim)")
        context_vecs = self.out_proj(context_vecs)
        return context_vecs

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(cfg["d_model"], cfg["d_model"], cfg["n_heads"], cfg["ctx_len"])
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["d_model"])
        self.norm2 = RMSNorm(cfg["d_model"])

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["d_model"])
        self.out_head = nn.Linear(cfg["d_model"], cfg["vocab_size"])

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        for blk in self.transformer_blocks:
            x = blk(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits