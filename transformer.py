import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        q = self.q_linear(x).reshape(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.k_linear(x).reshape(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v_linear(x).reshape(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(batch_size, seq_length, self.d_model)
        return self.out_linear(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, tgt_mask)))  # Masked self-attention
        x = self.norm2(x + self.dropout(self.cross_attn(enc_output, x, src_mask)))  # Cross-attention
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_ff, max_len=100, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, num_classes)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc_out(tgt[:, 0, :])  # Use first token for classification

# Example usage
if __name__ == "__main__":
    model = Transformer(input_dim=1000, d_model=512, num_heads=8, num_layers=6, dim_ff=2048)
    src = torch.randint(0, 1000, (8, 20))  # Batch of 8, sequence length of 20
    tgt = torch.randint(0, 1000, (8, 20))
    output = model(src, tgt)
    print(output.shape)  # Should be (8, 10)
