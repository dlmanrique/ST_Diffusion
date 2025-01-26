import torch
import torch.nn as nn
import math


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, latent_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.encoder_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU())

        self.positional_encoding = PositionalEncoding(embedding_dim)
        #self.positional_encoding = nn.Parameter(torch.randn(1, 7, embedding_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward= 2 * embedding_dim,
            dropout=dropout,
        )
        
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.to_latent = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, latent_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        # Input shape: [Batch, 7, 1024]
        x = self.encoder_projection(x) # Shape: [Batch, 7, Embedding Dim]
        x = self.positional_encoding(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x)  # Shape: [Batch, 7, Embedding Dim]
        x = x.permute(1, 0, 2)
        
        x = self.to_latent(x)  # Shape: [Batch, 7, Latent Dim]
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, embedding_dim, input_dim, dropout=0.1):
        super().__init__()
 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, input_dim)
        )

    def forward(self, z):
        # Input shape: [Batch, 7, 128]
        #batch_size, num_spots, _ = z.size()
        #z = z.view(-1, z.size(-1))  # Flatten to [Batch * 7, Latent Dim]
        z = self.decoder(z)  # Shape: [Batch * 7, Output Dim]
        return z
        #return z.view(batch_size, num_spots, -1)  # Reshape back to [Batch, 7, Output Dim]


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.encoding = self.encoding.unsqueeze(0)  # Shape: [1, max_len, embedding_dim]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, Seq_len, Embedding_dim]
        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)
