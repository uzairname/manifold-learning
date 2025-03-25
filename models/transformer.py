import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embed(nn.Module):
    def __init__(self, n_vocab, d_model, init_scale=1):
        super().__init__()
        self.W_embed = nn.Parameter(t.randn(d_model, n_vocab)/np.sqrt(d_model)*init_scale)
    
    def forward(self, x):
      return t.einsum('dbp -> bpd', self.W_embed[:, x])


class Unembed(nn.Module):
  def __init__(self, n_vocab, d_model, init_scale=1):
    super().__init__()
    self.W_ue = nn.Parameter(t.randn(d_model, n_vocab)/np.sqrt(d_model)*init_scale)
  
  def forward(self, x):
    return x @ self.W_ue


class PosEmbed(nn.Module):
  def __init__(self, d_model, seq_len, init_scale=1):
    super().__init__()
    self.W_pos = nn.Parameter(t.randn(seq_len, d_model)/np.sqrt(d_model)*init_scale)
    
  def forward(self, x):
    in_seq_len = x.shape[-2]
    return x + self.W_pos[:in_seq_len]


class LayerNorm(nn.Module):
  def __init__(self, d_model, init_scale=1):
    super().__init__()
    self.e = 1e-4
    self.w = nn.Parameter(t.ones(d_model))
    self.b = nn.Parameter(t.zeros(d_model))
    
  def forward(self, x: t.Tensor):
    return self.w * (x - x.mean(dim=-1, keepdim=True)) / (t.sqrt(x.var(dim=-1, keepdim=True) + self.e)) + self.b
    

class Attention(nn.Module):
  def __init__(self, d_model, n_heads, max_seq_len, init_scale=1):
    super().__init__()

    assert d_model % n_heads == 0, "embed_dim must be divisible by num_heads"
    self.head_dim = d_model // n_heads

    self.W_qkv = nn.Parameter(t.randn(n_heads, d_model, self.head_dim * 3)/np.sqrt(d_model)*init_scale)
    self.W_out = nn.Parameter(t.randn(n_heads, self.head_dim, self.head_dim * n_heads)/np.sqrt(d_model)*init_scale)

    # Matrix of 0s on the lower traingular part, and a large negative number everywhere else
    self.register_buffer('mask', - 1e10 * (1-t.tril(t.ones((max_seq_len, max_seq_len), dtype=t.float32))))

  def forward(self, x: t.Tensor):
    qkv = t.einsum('bsd,ndh -> bnsh', x, self.W_qkv) # batch, n_heads, seq_len, head_dim * 3
    q, k, v = t.chunk(qkv, 3, dim=-1) # batch, n_heads, seq_len, head_dim
    attn = t.einsum('bnqh,bnkh -> bnqk', q, k) / np.sqrt(self.head_dim) # batch, n_heads, seq_len_q, seq_len_k
    attn = attn + self.mask[:attn.shape[-2], :attn.shape[-2]]
    attn = F.softmax(attn, dim=-1)
    
    out = t.einsum('bnqk,bnkh -> bnqh', attn, v) # batch, n_heads, seq_len_q, head_dim
    out = out.reshape(out.shape[0], -1, out.shape[-2]) # batch, d_model, seq_len
    out = t.einsum('bds,nhd -> bsd', out, self.W_out) # batch, seq_len, d_model
    
    return out


class MLP(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, init_scale=1):
    super().__init__()
    self.W1 = nn.Parameter(t.randn(in_dim, hidden_dim) / np.sqrt(in_dim) * init_scale)
    self.b1 = nn.Parameter(t.zeros(hidden_dim))
    self.W2 = nn.Parameter(t.randn(hidden_dim, out_dim) / np.sqrt(hidden_dim) * init_scale)
    self.b2 = nn.Parameter(t.zeros(out_dim))
    
  def forward(self, x: t.Tensor):
    x = F.gelu(x @ self.W1 + self.b1)
    x = x @ self.W2 + self.b2
    return x


class TransformerBlock(nn.Module):
  def __init__(self, d_model, max_seq_len, n_heads, d_mlp, init_scale=1):
    super().__init__()
    
    self.ln1 = LayerNorm(d_model, init_scale=init_scale)
    self.attention = Attention(d_model, n_heads, max_seq_len, init_scale=init_scale)
    self.ln2 = LayerNorm(d_model, init_scale=init_scale)
    self.mlp = MLP(d_model, d_model, hidden_dim=d_mlp, init_scale=init_scale)

  def forward(self, x: t.Tensor):
    x = x + self.attention(self.ln1(x))
    # x = x + self.attention(x)
    x = x + self.mlp(self.ln2(x))
    # x = x + self.mlp(x)
    return x


class Transformer(nn.Module):
  def __init__(self, d_model, n_vocab, max_seq_len, n_heads, n_layers, d_mlp, last_token=True, init_scale=1):
    super().__init__()
    
    self.last_token = last_token
    self.embedding = Embed(n_vocab=n_vocab, d_model=d_model, init_scale=init_scale)
    self.pos_embedding = PosEmbed(d_model=d_model, seq_len=max_seq_len, init_scale=init_scale)
    
    self.transformer_blocks = nn.ModuleList([
      TransformerBlock(d_model=d_model, max_seq_len=max_seq_len, n_heads=n_heads, d_mlp=d_mlp, init_scale=init_scale)
      for _ in range(n_layers)
    ])
    
    self.ln = LayerNorm(d_model=d_model, init_scale=init_scale)
    self.unembed = Unembed(n_vocab=n_vocab, d_model=d_model, init_scale=init_scale)

  def forward(self, x: t.Tensor):
    x = self.embedding(x)
    x = self.pos_embedding(x)
    
    for transformer in self.transformer_blocks:
      x = transformer(x)
    
    x = self.ln(x)
    x = self.unembed(x)
    if self.last_token:
      x = x[..., -1, :]
    
    return x
