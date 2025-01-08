# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from spared.datasets import get_dataset
import torch
import torch.nn as nn
from activation_function import TanhWithAlpha

class Encoder_LSTM(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size,
                 num_layers
                 ) -> None:
        """
        Initializes Encoder object.

        Parameters
        ----------
        sizes : tuple
            Tuple of sizes of linear layers.
        """
        super(Encoder_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define biLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )
        
        self.hidden_projection = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the biLSTM encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        ----------
        torch.Tensor
            Compressed latent representation of shape (batch_size, 2 * hidden_size).
        """
        # Expand input to match LSTM expectations: (seq_len, batch_size, input_size)
        x = x.unsqueeze(1)

        # Pass through biLSTM
        lstm_out, hidden = self.lstm(x)

        # Combine final forward and backward hidden states
        #forward_hidden = hidden[0][0]  # Forward hidden state
        #backward_hidden = hidden[0][1]  # Backward hidden state
        #combined_hidden = (backward_hidden + forward_hidden)/2
        #combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # Shape: (batch_size, 2 * hidden_size)

        # Project to hidden_size
        #latent_representation = self.hidden_projection(combined_hidden)
        latent_representation = hidden[0].squeeze(0)
        #latent_representation = combined_hidden
        return latent_representation  

    def init_hidden(self, batch_size: int, device: torch.device = torch.device("cpu")):
        """
        Initialize hidden state with zeros.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        device : torch.device
            Device for the tensors (CPU or GPU).

        Returns
        ----------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of zero-initialized hidden and cell states.
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )