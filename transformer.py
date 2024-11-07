import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Create query, key, value projections for all heads
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        # Initialize weights properly
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        
        # Project and reshape to allow for multi-head attention
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, dim
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, dim
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, nh, T, dim
        
        # Compute scaled attention scores
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B, nh, T, T
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads: [1, 1, T, T] -> [B, nh, T, T]
            expanded_mask = mask.expand(B, self.n_head, T, T)
            att = att.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Apply softmax along the key dimension
        att = F.softmax(att, dim=-1)  # B, nh, T, T
        
        # For debugging - verify normalization
        sums = att.sum(dim=-1)  # Should be 1.0 for all positions
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
            print(f"Warning: Attention weights not properly normalized. Sums: {sums[0, 0]}")
        
        # Apply attention to values
        y = torch.matmul(att, v)  # B, nh, T, dim
        
        # Reshape and project back
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # B, T, C
        out = self.out_proj(y)
        
        return out, att

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # As specified in the assignment, hidden dim is 100 and activation is ReLU
        self.net = nn.Sequential(
            nn.Linear(n_embd, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_norm = self.ln1(x)
        attn_out, att_weights = self.attn(attn_norm)
        x = x + attn_out
        
        # Feedforward with residual connection and layer norm
        ff_norm = self.ln2(x)
        x = x + self.ff(ff_norm)
        
        return x, att_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store block_size
        self.block_size = block_size
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, idx):
        B, T = idx.size()
        
        # Assert sequence length is not greater than block_size
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} is greater than block_size {self.block_size}")
            
        # Get token and position embeddings
        token_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Combine embeddings and apply dropout
        x = self.dropout(token_emb + pos_emb.unsqueeze(0))
        
        # Store attention maps
        attention_maps = []
        
        # Apply transformer blocks
        for block in self.blocks:
            x, att_weights = block(x)
            # Store mean attention weights across batch dimension
            mean_att_weights = att_weights.mean(dim=0)  # [n_heads, seq_len, seq_len]
            for head_idx in range(mean_att_weights.size(0)):
                attention_maps.append(mean_att_weights[head_idx])
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Take mean of sequence dimension to get fixed-size representation
        x = x.mean(dim=1)  # [B, n_embd]
        
        return x, attention_maps

class Classifier(nn.Module):
    def __init__(self, n_embd, n_classes=3, hidden_dim=100, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class EncoderWithClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_classes=3, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
        self.classifier = Classifier(n_embd, n_classes, dropout=dropout)
        
    def forward(self, idx):
        # Get encoder outputs
        encoded_output, attention_maps = self.encoder(idx)
        
        # Pass through classifier
        logits = self.classifier(encoded_output)
        
        return logits, attention_maps

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_norm = self.ln1(x)
        attn_out, att_weights = self.attn(attn_norm, mask)
        x = x + attn_out
        
        # Feedforward with residual connection and layer norm
        ff_norm = self.ln2(x)
        x = x + self.ff(ff_norm)
        
        # Return both the output and attention weights
        return x, att_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_proj = nn.Linear(n_embd, vocab_size)
        
        # Other attributes
        self.block_size = block_size
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def get_causal_mask(self, size):
        # Create causal mask for decoder self-attention
        mask = torch.ones(1, 1, size, size)
        mask = torch.triu(mask, diagonal=1).eq(0)
        return mask
            
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Get token and position embeddings
        token_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Combine embeddings and apply dropout
        x = self.dropout(token_emb + pos_emb.unsqueeze(0))
        
        # Get causal mask for self-attention
        mask = self.get_causal_mask(T).to(idx.device)
        
        # Store attention maps from the first head only
        attention_maps = []
        
        # Apply transformer blocks
        for block in self.blocks:
            x, att_weights = block(x, mask)
            # att_weights shape: [batch_size, n_heads, seq_len, seq_len]
            
            # Take mean across batch dimension and add each head's attention map to flat list
            mean_att_weights = att_weights.mean(dim=0)  # [n_heads, seq_len, seq_len]
            
            for head_idx in range(mean_att_weights.size(0)):
                # Get attention map for single head: [seq_len, seq_len]
                head_att = mean_att_weights[head_idx]  # Shape [32, 32]
                attention_maps.append(head_att.detach())
            
        # Apply final layer norm and project to vocabulary
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return loss, attention_maps