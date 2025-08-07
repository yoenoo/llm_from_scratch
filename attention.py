# %%
import torch
# %%
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55], # step     (x^6)
])
inputs
# %%
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
attn_scores_2
# %%
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
attn_weights_2_tmp
# %%
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
# %%
attn_weights_2_naive = softmax_naive(attn_scores_2)
attn_weights_2_naive
# %%
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
attn_weights_2
# %%
print(inputs.shape)
attn_scores = inputs @ inputs.T
attn_scores
# %%
attn_weights = torch.softmax(attn_scores, dim=-1)
attn_weights
# %%
all_context_vecs = attn_weights @ inputs 
all_context_vecs
# %%
import torch.nn as nn 
# %%
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

# %%
d_in = inputs.shape[1]
d_out = 2
# %%
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v1(inputs)
# %%
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = q @ k.T
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ v
# %%
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2(inputs)
# %%
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False):
        super().__init__()
        self.d_out = d_out 
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        batch_size, seq_len, d_in = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = q @ k.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:seq_len,:seq_len], -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[1]**0.5, dim=-1)
        return attn_weights @ v

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length)
ca(batch)
# %%
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, qkv_bias) for _ in range(num_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, num_heads=2
)

context_vecs = mha(batch)
batch.shape, context_vecs.shape
# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be a multiple of num_heads!"

        self.head_dim = d_out // num_heads


    def forward(self, x):
        pass