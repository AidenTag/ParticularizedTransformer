import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class DALexAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.is_causal = getattr(config, 'is_causal', True)
        self.use_dalex = getattr(config, 'use_dalex', True)
        
        # DALex specific parameter
        self.pressure = getattr(config, 'dalex_pressure', 0.5)

        # Causal mask registration
        if not self.flash and self.is_causal:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def generate_dalex_weights(self, shape, device, dtype):
        # Generates random weighting vector 'w'
        # High pressure = sparse, spiky weights (Lexicase behavior)
        noise = torch.randn(shape, device=device, dtype=dtype) * self.pressure # mean 0, std = pressure

        # Softmax over the noise
        weights = F.softmax(noise, dim=-1) # shape remains after softmaxxing over the last dimension
        return weights * weights.size(-1) # Normalize so mean is 1.0 (keeping scale consistent)

        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        hs = C // self.n_head # head size (n_embd // n_head)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, 3 * C) -> 3 x (B, T, C)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if self.training and self.use_dalex: # Inject Particularity Pressure
            # Random DALex weight vector scales the FEATURE dimension of Q and K (-1, size = hs)
            # in order to "select" which features are important for this particular head in computing attention similarity
            
            # shape(q, k) == (B, nh, T, hs)
            # We want one weight vector per head, per batch, applied to hs dimension
            w = self.generate_dalex_weights((B, self.n_head, 1, hs), x.device, x.dtype)

            # We sqrt to apply appropriately to the mat mul
            # Since the attention scores are a dot product of q and we want the weights to be applied in a way that reflects their influence on the final attention scores. By taking the square root of the weights, we ensure that when q and k are multiplied together, the influence of the weights is appropriately reflected in the resulting attention scores.
            w_sqrt = torch.sqrt(w)
            
            q = q * w_sqrt
            k = k * w_sqrt

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
        else:
            # manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
