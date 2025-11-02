import torch
from torch import nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    """
    A basic implementation of the transformer encoder architecture using self-attention.
    """
    
    def __init__(self, embed_dim, num_heads, layers=1, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, 
                 embed_dropout=0.0, attn_mask=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.attn_mask = attn_mask
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        
        # Stack multiple layers if specified
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout, embed_dropout, attn_mask)
            for _ in range(layers)
        ])
        
    def forward(self, q, k=None, v=None, mask=None):
        """
        Forward pass with cross-attention support
        q: query tensor
        k: key tensor (if None, use q)
        v: value tensor (if None, use k)
        """
        if k is None:
            k = q
        if v is None:
            v = k
            
        # Apply each transformer layer in sequence
        for layer in self.transformer_layers:
            q = layer(q, k, v, mask)
        return q

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, 
                 embed_dropout=0.0, attn_mask=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attention_dropout = nn.Dropout(attn_dropout)
        self.relu_dropout = nn.Dropout(relu_dropout)
        self.res_dropout = nn.Dropout(res_dropout)
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, q, k, v, mask=None):
        # Multi-head attention
        residual = q
        q = self.layer_norm1(q)
        k = self.layer_norm1(k)
        v = self.layer_norm1(v)
        
        batch_size, seq_len_q, embed_dim = q.size()
        seq_len_k = k.size(1)
        scaling = float(self.head_dim) ** -0.5
        
        # Project q, k, v
        q = self.q_proj(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, embed_dim)
        out = self.out_proj(out)
        out = self.res_dropout(out)
        out = residual + out
        
        # Feed-forward network
        residual = out
        out = self.layer_norm2(out)
        out = self.feed_forward(out)
        out = self.res_dropout(out)
        out = residual + out
        
        return out
