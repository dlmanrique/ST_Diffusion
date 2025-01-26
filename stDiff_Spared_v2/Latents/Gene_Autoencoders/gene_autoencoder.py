# This file has all the possible Gene autoencoders
import random
from Transformer_encoder_decoder import TransformerEncoder, TransformerDecoder

class Transformer_encoder_mlp_decoder():
    def __init__(self,
                 input_dim: int, 
                 latent_dim: int,
                 embedding_dim: int,
                 num_layers: int,
                 num_heads: int) -> None:
        super().__init__()
        
        
        self.Encoder = TransformerEncoder(input_dim=input_dim, 
                               embedding_dim=embedding_dim, 
                               latent_dim=latent_dim, 
                               num_heads=num_heads, 
                               num_layers=num_layers, 
                               dropout=0.1)
        
        self.Decoder = TransformerDecoder(input_dim=input_dim, 
                               embedding_dim=embedding_dim, 
                               latent_dim=latent_dim, 
                               num_heads=num_heads, 
                               num_layers=4, 
                               dropout=0.1)

    def encoder(self, x):
        x = self.Encoder(x)
        return x
    
    def decoder(self, encoder_output):
        x = self.Decoder(encoder_output)
        return x
        
