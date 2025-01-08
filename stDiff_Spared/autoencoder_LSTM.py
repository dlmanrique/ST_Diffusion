import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from encoder_LSTM import Encoder_LSTM
from decoder_LSTM import Decoder_LSTM
import torch.optim as optim
from utils import *
import random

class Autoencoder_LSTM(pl.LightningModule):
    def __init__(self,
                 input_size, 
                 hidden_size,
                 lr=1e-5,
                 gene_weights = None):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.decoder = Decoder_LSTM(input_size=hidden_size, hidden_size=hidden_size, output_size=input_size, num_layers=1)

        self.lr = lr
        self.weights = gene_weights
        
    def forward(self, x, batch_size):
        """The forward function takes in an image and returns the reconstructed image."""
        encoder_hidden = self.encoder.init_hidden(batch_size, device=self.device)
        # Encode input
        encoder_hidden = self.encoder(x)
        # Use last input as the decoder's initial input
        decoder_input = torch.zeros(batch_size, self.decoder.input_size).to(self.device)
        decoder_hidden = encoder_hidden
        # Decode the latent representation
        decoder_output, _ = self.decoder(decoder_input, decoder_hidden)
        return decoder_output
        
    def _get_reconstruction_loss(self, x, x_hat, alpha=0.9):
        weights_test = self.weights.unsqueeze(0).repeat(x.size(0), 1)
        important_mask = (weights_test == 1).bool()
        auxiliary_mask = (weights_test == 0).bool()

        important_loss = F.mse_loss(x[important_mask], x_hat[important_mask])
        auxiliary_loss = F.mse_loss(x[auxiliary_mask], x_hat[auxiliary_mask])
        
        total_loss = alpha * important_loss + (1 - alpha) * auxiliary_loss 
        #total_loss = F.mse_loss(x[important_mask], x_hat.flatten())
        
        self.log("important_loss", important_loss, prog_bar=True, logger=True)
        self.log("auxiliary_loss", auxiliary_loss, prog_bar=True, logger=True)
        self.log("total_loss", total_loss, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-10)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x = batch[0].float()
        # Randomly decide to add noise
        if random.random() < 0.5:  # 50% chance to add noise
            noisy_x = add_noise(x, noise_factor=0.1)
        else:
            noisy_x = x  # Use clean data
        #noisy_x = x
        batch_size = x.shape[0]
        x_hat = self.forward(noisy_x, batch_size)
        loss = self._get_reconstruction_loss(x, x_hat)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0].float()
        batch_size = x.shape[0]
        x_hat = self.forward(x, batch_size)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log("val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        x = batch[0].float()
        batch_size = x.shape[0]
        x_hat = self.forward(x, batch_size)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log("test_loss", loss, logger=True)
