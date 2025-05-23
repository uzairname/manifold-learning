import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedding(nn.Module):
  def __init__(self, n_vocab, d_model, init_scale=1):
      super().__init__()
      self.W_embed = nn.Parameter(t.randn(d_model, n_vocab)/np.sqrt(d_model)*init_scale)
  
  def forward(self, x):
    return t.einsum('dbp -> bpd', self.W_embed[:, x])


class PosEmbedding(nn.Module):
  def __init__(self, d_model, seq_len, init_scale=1):
    super().__init__()
    self.W_pos = nn.Parameter(t.randn(seq_len, d_model)/np.sqrt(d_model)*init_scale)
    
  def forward(self, x):
    in_seq_len = x.shape[-2]
    return x + self.W_pos[:in_seq_len]
    
    
class Unembed(nn.Module):
  def __init__(self, n_vocab, d_model, init_scale=1):
    super().__init__()
    self.W_ue = nn.Parameter(t.randn(d_model, n_vocab)/np.sqrt(d_model)*init_scale)
  
  def forward(self, x):
    return x @ self.W_ue


class Attention(nn.Module):
  def __init__(self, d_model, n_heads, max_seq_len, init_scale=1):
    super().__init__()

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    self.head_dim = d_model // n_heads

    self.W_qkv = nn.Parameter(t.randn(n_heads, d_model, self.head_dim * 3)/np.sqrt(d_model)*init_scale)
    self.W_out = nn.Parameter(t.randn(n_heads, self.head_dim, self.head_dim * n_heads)/np.sqrt(d_model)*init_scale)

    # Matrix of 0s on the lower traingular part, and a large negative number everywhere else
    self.register_buffer('mask', - 1e10 * (1-t.tril(t.ones((max_seq_len, max_seq_len), dtype=t.float32))))

  def forward(self, x: t.Tensor):
    qkv = t.einsum('bsd,ndh -> bnsh', x, self.W_qkv) # batch, n_heads, seq_len, head_dim * 3
    q, k, v = t.chunk(qkv, 3, dim=-1) # batch, n_heads, seq_len, head_dim
    attn = t.einsum('bnqh,bnkh -> bnqk', q, k) / np.sqrt(self.head_dim) # batch, n_heads, seq_len_q, seq_len_k
    attn = attn + self.mask[:attn.shape[-2], :attn.shape[-1]]
    attn = F.softmax(attn, dim=-1)
    
    out = t.einsum('bnqk,bnkh -> bnqh', attn, v) # batch, n_heads, seq_len_q, head_dim
    out = out.reshape(out.shape[0], -1, out.shape[-2]) # batch, d_model, seq_len
    out = t.einsum('bds,nhd -> bsd', out, self.W_out) # batch, seq_len, d_model
    
    return out


class MLP(nn.Module):
  def __init__(self, d_model, hidden_dim, init_scale):
    super().__init__()
    self.fc1 = nn.Linear(d_model, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, d_model)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class TransformerBlock(nn.Module):
  def __init__(self, d_model, max_seq_len, n_heads, d_mlp, init_scale=1, ln_eps=1e-2, use_ln=True):
    super().__init__()
    self.use_ln = use_ln
    if use_ln:
      self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
      self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)

    self.attention = Attention(d_model, n_heads, max_seq_len, init_scale=init_scale)
    self.mlp = MLP(d_model, hidden_dim=d_mlp, init_scale=init_scale)

  def forward(self, x: t.Tensor):
    if self.use_ln: 
      x = self.ln1(x)
    x = x + self.attention(x)
    if self.use_ln: 
      x = self.ln2(x)
    x = x + self.mlp(x)
    return x


class Transformer(nn.Module):
  def __init__(self, d_model, n_vocab, max_seq_len, n_heads, n_layers, d_mlp, last_token=True, init_scale=1, ln_eps=1e-2, use_ln=True):
    super().__init__()
    
    self.use_ln = use_ln
    self.last_token = last_token
    self.embedding = Embedding(n_vocab=n_vocab, d_model=d_model, init_scale=init_scale)
    self.pos_embedding = PosEmbedding(d_model=d_model, seq_len=max_seq_len, init_scale=init_scale)
    
    self.transformer_blocks = nn.ModuleList([
      TransformerBlock(d_model=d_model, max_seq_len=max_seq_len, n_heads=n_heads, d_mlp=d_mlp, init_scale=init_scale, ln_eps=ln_eps, use_ln=use_ln)
      for _ in range(n_layers)
    ])
    
    if use_ln: 
      self.ln = nn.LayerNorm(d_model, eps=ln_eps)
    self.unembed = Unembed(n_vocab=n_vocab, d_model=d_model, init_scale=init_scale)

  def forward(self, x: t.Tensor):
    x = self.embedding(x)
    x = self.pos_embedding(x)
    
    for transformer in self.transformer_blocks:
      x = transformer(x)
    
    if self.use_ln:
      x = self.ln(x)
    x = self.unembed(x)
    if self.last_token:
      x = x[..., -1, :]
    
    return x
