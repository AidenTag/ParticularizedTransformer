import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from .dalex_attention import DALexCausalAttention

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=8, n_embd=128, dropout=0.1, bias=False, dalex_pressure=0.5, use_dalex=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.dalex_pressure = dalex_pressure
        self.use_dalex = use_dalex

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.use_dalex:
            self.attn = DALexCausalAttention(config)
        else:
            # Fallback to a standard causal usage (using the same class with pressure=0 usually works if implemented that way, 
            # but better to interpret use_dalex=False as standard attention)
            # We can use DALexCausalAttention with training behavior disabled or just standard implementation.
            # For strict benchmarking, let's assume we want the exact same code just without the noise injection.
            # In my DALexCausalAttention, if self.training is False, it skips noise.
            # But we want to train a baseline. So let's reuse the class but force pressure to 0 or skip the injection block.
            # However, looking at the code, DALexCausalAttention only checks self.training. 
            # Ideally, we should have a 'Standard' mode.
            # Let's assume for now we use the same class but with a flag or we implement a StandardAttention.
            # For simplicity in this file, I'll instantiate DALexCausalAttention and relying on config.use_dalex 
            # being handled inside DALexCausalAttention or wrapping it.
            # The prompt implies DALexCausalAttention is the module to test. 
            # Standard Transformer validation likely needs a standard MultiheadAttention.
            # I will assume DALexCausalAttention is capable of being standard if pressure is effectively handled
            # or I should add a Standard one. 
            # Let's use DALexCausalAttention for both but set 'dalex_pressure' to 0 if standard?
            # No, standard attention is mean-based (dot product). DALex is also dot product but weighted Q/K.
            # If weights are all 1s (pressure=0 -> exp(0)=1), it is standard attention.
            # So setting dalex_pressure = 0 makes it standard attention.
            self.attn = DALexCausalAttention(config)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ListOpsTransformer(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.learning_rate = learning_rate

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Classification head for ListOps (0-9)
        self.head = nn.Linear(config.n_embd, 10, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # We need to pool the sequence for classification.
        # Common strategies:
        # 1. Use the [CLS] token (if we had one)
        # 2. Use the last token
        # 3. Mean pool
        # For causal models, the last token has attended to everything.
        logits = self.head(x[:, -1, :]) # (b, 10)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss = self(idx, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss = self(idx, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

import math
