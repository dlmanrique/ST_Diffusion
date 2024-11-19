import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from distributions import DiagonalGaussianDistribution
from losses import LPIPSWithDiscriminator, DummyLoss
from src.taming_transformers.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from src.taming_transformers.taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
import numpy as np

#from ldm.util import instantiate_from_config

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 learning_rate,
                 image_key="image",
                 ):
        super().__init__()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.encoder = Encoder(in_channels=1, num_res_blocks=2, ch=256, ch_mult=(1,2,4),
                               z_channels=1, double_z=False, resolution=512,
                               attn_resolutions=[], dropout=0.0, resamp_with_conv=True,
                               out_ch=1)
        
        self.decoder = Decoder(out_ch=1, z_channels=1, attn_resolutions=[], dropout=0.0,
                               resamp_with_conv=True, in_channels=1, num_res_blocks=2,
                               ch_mult=(1,2,4), resolution=512, ch=256, scale_factor=[1, 7/3, 3])

        self.loss = LPIPSWithDiscriminator(disc_start=50001, kl_weight=0.000001, disc_weight=0.5)
        self.z_channels = 1
        self.quant_conv = torch.nn.Conv2d(self.z_channels, embed_dim*2, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.z_channels, 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch):
        x = batch[0]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        # Get the optimizers
        opt_ae, opt_disc = self.optimizers()

        # Train encoder + decoder (autoencoder)
        opt_ae.zero_grad()  # Zero out gradients
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)  # Perform backpropagation manually
        opt_ae.step()  # Update weights for the autoencoder

        # Log loss for autoencoder
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # Train the discriminator
        opt_disc.zero_grad()  # Zero out gradients
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)  # Perform backpropagation manually
        opt_disc.step()  # Update weights for the discriminator

        # Log loss for discriminator
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return {"loss": aeloss}  # Return the primary loss for logging purposes


    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        
        # Compute the loss and log test metrics
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="test")
        self.log("test/rec_loss", log_dict_ae["test/rec_loss"])
        self.log_dict(log_dict_ae)
        
        return log_dict_ae

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModel(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 learning_rate,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 ):
        
        super().__init__()
        self.lr_g_factor = 1.0
        self.learning_rate = learning_rate
        self.n_embed = 8192
        self.encoder = Encoder(in_channels=1, num_res_blocks=2, ch=256, ch_mult=(1,2,4),
                               z_channels=1, double_z=False, resolution=512,
                               attn_resolutions=[], dropout=0.0, resamp_with_conv=True,
                               out_ch=1)
        
        self.decoder = Decoder(out_ch=1, z_channels=1, attn_resolutions=[], dropout=0.0,
                               resamp_with_conv=True, in_channels=1, num_res_blocks=2,
                               ch_mult=(1,2,4), resolution=512, ch=256, scale_factor=[1, 7/3, 3])

        self.loss = self.loss = VQLPIPSWithDiscriminator(disc_start=50001)
        self.quantize = VectorQuantizer(self.n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        
        self.z_channels = 1
        self.quant_conv = torch.nn.Conv2d(self.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        # Get the optimizers
        opt_ae, opt_disc = self.optimizers()

        # Train encoder + decoder (autoencoder)
        opt_ae.zero_grad()  # Zero out gradients
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)  # Perform backpropagation manually
        opt_ae.step()  # Update weights for the autoencoder

        # Log loss for autoencoder
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # Train the discriminator
        opt_disc.zero_grad()  # Zero out gradients
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)  # Perform backpropagation manually
        opt_disc.step()  # Update weights for the discriminator

        # Log loss for discriminator
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return {"loss": aeloss}  # Return the primary loss for logging purposes

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def test_step(self, batch, batch_idx):
        log_dict = self._test_step(batch, batch_idx)
        return log_dict
    
    def _test_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="test"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="test"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"test{suffix}/rec_loss"]
        self.log(f"test{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    

class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
