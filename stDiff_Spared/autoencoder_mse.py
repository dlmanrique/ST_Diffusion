import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
import torch.optim as optim

class Autoencoder(pl.LightningModule):
    def __init__(self,
                 num_res_blocks=2,
                 ch=256,
                 ch_mult=(1,2,4),
                 lr=1e-5):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        self.num_res_blocks = num_res_blocks
        self.ch = ch
        self.ch_mult = ch_mult
        self.lr = lr
        # Creating encoder and decoder
        self.encoder = Encoder(in_channels=1, num_res_blocks=self.num_res_blocks, ch=self.ch, ch_mult=self.ch_mult,
                               z_channels=1, double_z=False, resolution=512,
                               attn_resolutions=[], dropout=0.0, resamp_with_conv=True,
                               out_ch=1)
        
        self.scale_factor = [1, 7/3, 3][:len(self.ch_mult)]
        
        self.decoder = Decoder(out_ch=1, z_channels=1, attn_resolutions=[], dropout=0.0,
                               resamp_with_conv=True, in_channels=1, num_res_blocks=self.num_res_blocks,
                               ch_mult=self.ch_mult, resolution=512, ch=self.ch, scale_factor=self.scale_factor)
        # Example input array needed for visualizing the graph of the network
        #self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x = batch[0]  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat)
        #loss = F.mse_loss(x, x_hat, reduction="none")
        #loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        
        #log LR
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    
        #Log train loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, logger=True)