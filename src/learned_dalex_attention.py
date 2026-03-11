import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LearnedDALexAttention(nn.Module):
   def __init__(self, config):
      super().__init__()
      assert config.n_embd % config.n_head == 0

      self.n_head = config.n_head
      self.n_embd = config.n_embd
      self.hs = config.n_embd // config.n_head
      self.dropout = config.dropout
      self.bias = config.bias

      # 1. Learned Selection Parameters
      # Each head has its own "Vantage Point" (weights over the full embedding)
      # and its own "Particularity Pressure" (stubbornness/sparsity)
      self.niche_weights = nn.Parameter(torch.randn(self.n_head, self.n_embd))
      self.niche_pressure = nn.Parameter(torch.zeros(self.n_head, 1))

      # 2. The Vantage Projections
      # Instead of a single (C, 3C) projection, each head i projects the 
      # FULL embedding (~512) into its PRIVATE head space (~64).
      # We store this as one parameter of shape (3, nh, C, hs) for (Q,K,V)
      self.qkv_projection = nn.Parameter(torch.empty(3, self.n_head, self.n_embd, self.hs))
      torch.nn.init.normal_(self.qkv_projection, std=0.02 / math.sqrt(2 * getattr(config, 'n_layer', 1)))
      
      # Output projection to merge heads back to the residual stream
      self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)

      self.attn_dropout = nn.Dropout(self.dropout)
      self.resid_dropout = nn.Dropout(self.dropout)
      
      self.is_causal = getattr(config, 'is_causal', True)
      self.use_dalex_selection = getattr(config, 'use_dalex_selection', True)

      self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

      # Causal mask registration for non-Flash Attention
      if not self.flash and self.is_causal:
         self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

   def get_niche_mask(self):
      """ Generates a DALex-style weighting over the full embedding for each head. """
      # pressure = softplus(raw_pressure) ensures pressure is always > 0
      p = F.softplus(self.niche_pressure) # (nh, 1)

      # Standardize raw weights per head (mean 0, std 1) for numerical stability
      mu = self.niche_weights.mean(dim=-1, keepdim=True)
      std = self.niche_weights.std(dim=-1, keepdim=True) + 1e-6
      z = (self.niche_weights - mu) / std

      # Softmax across the full C dimension to select "Particular" features
      # Multiplying by C keeps the average weight at 1.0 (mean-1 normalization)
      mask = F.softmax(z * p, dim=-1) * self.n_embd
      return mask # (nh, C)

   def forward(self, x):
      B, T, C = x.size()
      
      # Step 1: Feature Selection
      # Every head generates its own lens (mask) over the 512 embedding dimensions
      if self.use_dalex_selection:
         w = self.get_niche_mask() # (nh, C)
      else:
         w = x.new_ones((self.n_head, C)) # Default: see everything equally

      # Step 2: Routed Projection (The "Vantage" step)
      # We use einsum to do: (x * w_head) @ W_projection_head
      # b=batch, t=time, c=input_dim, n=head, i=qkv_index, h=head_dim
      # This allows each head to 'discover' its own features from the full C
      qkv = torch.einsum('btc,nc,inch->inbth', x, w, self.qkv_projection)
      q, k, v = qkv[0], qkv[1], qkv[2] # Each is (B, nh, T, hs)

      # Step 3: Self-Attention Calculation
      if self.flash:
         y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=self.is_causal
         )
      else:
         # Standard scaled dot product attention
         scale = 1.0 / math.sqrt(self.hs)
         att = (q @ k.transpose(-2, -1)) * scale
         if self.is_causal:
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
         att = F.softmax(att, dim=-1)
         att = self.attn_dropout(att)
         y = att @ v # (B, nh, T, hs)

      # Step 4: Output Assembly
      # Re-assemble the head outputs side-by-side back into the residual stream dim
      y = y.transpose(1, 2).contiguous().view(B, T, C)
      y = self.resid_dropout(self.c_proj(y))
      return y

   # Remove old unused code below this line


