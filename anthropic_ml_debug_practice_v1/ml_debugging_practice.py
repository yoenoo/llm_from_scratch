# ml_debugging_practice.py
# ML Debugging Practice Problems for Interview
# Each problem has multiple bugs - find and fix them!

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# PROBLEM 1: Transformer Multi-Head Attention (3-4 bugs)
# =============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Transform and reshape for multi-head attention
        Q = self.q_linear(query).view(*query.shape[:-1], self.num_heads, self.head_dim)
        K = self.k_linear(key).view(*key.shape[:-1], self.num_heads, self.head_dim)
        V = self.v_linear(value).view(*value.shape[:-1], self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores - BUG: Missing scaling factor
        scores = torch.matmul(Q, K.transpose(-2, -1)) / K.shape[-1]**0.5 ## fix
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_linear(context)

# =============================================================================
# PROBLEM 2: Positional Encoding (2-3 bugs)
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        # BUG: Should be register_buffer, not register_parameter
        # self.register_parameter('pe', nn.Parameter(pe))
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# =============================================================================
# PROBLEM 3: Training Loop with Memory Leaks (4-5 bugs)
# =============================================================================
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    all_losses = []
    best_model = None
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # BUG: Wrong order - backward before zero_grad
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # BUG: Storing tensor objects instead of scalars
            epoch_losses.append(loss)
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
                
            # BUG: Storing full model state dict instead of copying
            if len(all_losses) == 0 or loss < min(all_losses):
                best_model = model.state_dict()
        
        all_losses.extend(epoch_losses)
        # BUG: Computing mean of tensor objects
        print(f'Epoch {epoch} completed. Average loss: {sum(epoch_losses)/len(epoch_losses)}')
    
    return best_model, all_losses

# =============================================================================
# PROBLEM 4: KV Cache Implementation (3-4 bugs)
# =============================================================================
class KVCache:
    def __init__(self, max_seq_length, num_heads, head_dim):
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.reset()
    
    def reset(self):
        # BUG: Cache initialized on CPU, might not match input device
        self.k_cache = torch.zeros(1, self.num_heads, self.max_seq_length, self.head_dim)
        self.v_cache = torch.zeros(1, self.num_heads, self.max_seq_length, self.head_dim)
        self.current_length = 0
    
    def update_cache(self, k, v):
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        end_pos = self.current_length + seq_len
        
        # BUG: No bounds checking for cache overflow
        # Update cache
        self.k_cache[:, :, self.current_length:end_pos] = k
        self.v_cache[:, :, self.current_length:end_pos] = v
        
        self.current_length += seq_len
        
        return (self.k_cache[:, :, :self.current_length], 
                self.v_cache[:, :, :self.current_length])

# =============================================================================
# PROBLEM 5: Gradient Clipping and Numerical Stability (2-3 bugs)
# =============================================================================
def train_with_gradient_clipping(model, dataloader, optimizer, criterion, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # BUG: Using wrong function name (clip_grad_norm vs clip_grad_norm_)
        torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
        
        # Check for NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                return None
        
        optimizer.step()
        
        # BUG: Accumulating tensor objects instead of scalar values
        total_loss += loss
    
    return total_loss / len(dataloader)

# =============================================================================
# PROBLEM 6: Transformer Decoder Layer (4-5 bugs)
# =============================================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # BUG: Wrong order - should apply norm before attention (pre-norm)
        # Self attention
        attn_out = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross attention
        cross_out = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # BUG: Missing dropout in FFN residual connection
        # Feed forward
        ff_out = self.ffn(x)
        x = self.norm3(x + ff_out)
        
        return x

# =============================================================================
# PROBLEM 7: Attention Mask Creation (2-3 bugs)
# =============================================================================
def create_causal_mask(seq_len, device):
    """Create a causal (lower triangular) mask for self-attention"""
    # BUG: Wrong diagonal offset - should be diagonal=1
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=0)
    return mask == 0

def create_padding_mask(input_ids, pad_token_id=0):
    """Create a padding mask"""
    mask = (input_ids != pad_token_id)
    # BUG: Wrong unsqueeze dimensions for proper broadcasting
    return mask.unsqueeze(1).unsqueeze(1)

def combine_masks(causal_mask, padding_mask):
    """Combine causal and padding masks"""
    if padding_mask is not None:
        # BUG: Wrong broadcasting - dimensions don't align properly
        combined_mask = causal_mask * padding_mask
    else:
        combined_mask = causal_mask
    return combined_mask

# =============================================================================
# PROBLEM 8: Model Evaluation (3-4 bugs)
# =============================================================================
def evaluate_model(model, dataloader, criterion, device):
    # BUG: Missing model.eval() call
    total_loss = 0
    correct = 0
    total = 0
    
    # BUG: Missing torch.no_grad() context manager
    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}')
    return accuracy, avg_loss

# =============================================================================
# SUMMARY OF BUGS TO FIND:
# =============================================================================
"""
PROBLEM 1 (MultiHeadAttention):
- Missing scaling factor in attention scores
- Potentially wrong key sequence length handling

PROBLEM 2 (PositionalEncoding):
- register_parameter instead of register_buffer
- Should use requires_grad=False

PROBLEM 3 (Training Loop):
- Wrong order: backward() before zero_grad()
- Storing tensors instead of scalar values
- Computing mean of tensor objects

PROBLEM 4 (KV Cache):
- Device mismatch between cache and input
- No bounds checking for cache overflow
- Wrong tensor indexing

PROBLEM 5 (Gradient Clipping):
- Wrong function name: clip_grad_norm vs clip_grad_norm_
- Accumulating tensor instead of scalar

PROBLEM 6 (Decoder Layer):
- Wrong normalization order (post-norm vs pre-norm)
- Missing dropout in FFN residual

PROBLEM 7 (Attention Masks):
- Wrong diagonal offset in causal mask
- Wrong unsqueeze dimensions for broadcasting

PROBLEM 8 (Evaluation):
- Missing model.eval()
- Missing torch.no_grad()
- Model left in wrong mode
"""