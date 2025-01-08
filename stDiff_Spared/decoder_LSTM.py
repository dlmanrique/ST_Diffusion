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

class Decoder_LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers
                 ) -> None:
        """
        Initializes Decoder object.

        Parameters
        ----------
        sizes : tuple
            Tuple of sizes of linear layers.
        """
        super(Decoder_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Define uni-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected layer to project back to original space
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        encoder_hidden_states : tuple[torch.Tensor, torch.Tensor]
            Hidden and cell states from the encoder.
        z : torch.Tensor
            Input tensor in the latent space, shape (batch_size, 2 * hidden_size).

        Returns
        ----------
        torch.Tensor
            Reconstructed output in the original space.
        """
        # Expand z to match LSTM input expectations: (1, batch_size, input_size)
        z = z.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
        # Pass through LSTM using encoder's final hidden state
        initial_cell_state = torch.zeros_like(encoder_hidden_states)
        lstm_out, self.hidden = self.lstm(z, (encoder_hidden_states, initial_cell_state))

        # Map the hidden state output back to the original input size
        output = self.linear(lstm_out.squeeze(1))  # Shape: (batch_size, output_size)

        return output, self.hidden