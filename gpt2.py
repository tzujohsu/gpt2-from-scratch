from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # sets up a causal mask for attention
        # torch.tril(...) creates a lower triangular matrix
        # register_buffer: Stores this tensor in the model as a non-trainable buffer.
        # It will move with the model to GPU, and be saved in checkpoints, but won't be updated by gradients.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) \
                                            .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # (B, nh, T, hs): makeing the number of heads in to the batch dimension
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(C // self.n_head))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd) 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x) # an approximation version
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # exchange info and communicate, reduce operation
        x = x + self.mlp(self.ln_2(x)) # map operation
        return x

@dataclass
class GPTConfig:
    block_size: int = 245
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word embedding table
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding table
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Block is a custom class to be defined
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        