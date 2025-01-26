from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from metrics import get_metrics
import squidpy as sq
import numpy as np
import matplotlib
import argparse
import torch

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                        type=str,               default='villacampa_lung_organoid',      help='Dataset to use.')
    parser.add_argument('--pred_layer',                     type=str,               default='c_t_deltas',                    help='SpaRED prediction layer to use.')
    parser.add_argument('--num_neighs',                     type=int,               default=6,                               help='Amount of neighbors considered to build spot neighborhoods. Must be the same as the ones used to train the autoencoder. Use -1 to avoid neighbors info')
    parser.add_argument('--normalize_input',                type=str2bool,          default=True,                            help='Whether or not to normalize the DiT input data (encoded matrix) between -1 and 1 when preparing dataloader.')
    parser.add_argument('--autoencoder_ckpts_path',         type=str,               default='',                              help='Path to trained checkpoints of AE corresponding to the dataset used.')
    parser.add_argument('--decode_as_matrix',               type=str2bool,          default=True,                            help='Whether or not the decoder receives 2D inputs.')
    # Model parameters #######################################################################################################################################################################
    parser.add_argument('--dit_hidden_size',                type=int,               default=1024,                            help='')
    parser.add_argument('--dit_depth',                      type=int,               default=12,                              help='')
    parser.add_argument('--num_heads',                      type=int,               default=16,                              help='')
    parser.add_argument("--concat_dim",                     type=int,               default=0,                               help='Which dimension used to concat the condition.')
    parser.add_argument('--dit_ckpts_path',                 type=str,               default='',                              help='Path to trained checkpoints of DiT corresponding to the dataset used. Optional.')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--train',                          type=str2bool,          default=True,                            help='Train model.')
    parser.add_argument('--test',                           type=str2bool,          default=True,                            help='Test model.')
    parser.add_argument('--test_ckpts_path',                type=str,               default='',                              help='Path to checkpoints to be testing.')
    parser.add_argument('--normalized_data',                type=str2bool,          default=False,                           help='Whether or not to work with normalized expression matrix.')
    parser.add_argument('--lr',                             type=float,             default=0.0001,                          help='lr to train DiT.')
    parser.add_argument('--batch_size',                     type=int,               default=128,                             help='Batch size used to train the diffusion model.')
    parser.add_argument('--num_epochs',                     type=int,               default=3000,                            help='Number of training epochs.')
    parser.add_argument('--train_diffusion_steps',          type=int,               default=1500,                            help='Number of diffusion steps for training process.')
    parser.add_argument('--sample_diffusion_steps',         type=int,               default=1500,                            help='Number of diffusion steps for val or test process.')
    parser.add_argument('--step_size',                      type=float,             default=600,                             help='Step size to use in learning rate scheduler')
    parser.add_argument("--adjust_loss",                    type=str2bool,          default=True,                            help='If True the loss is obtained only on masked data. If False the loss takes into account the entire set of genes and spots.')
    parser.add_argument("--scheduler",                      type=str2bool,          default=True,                            help='Whether to use LR scheduler or not.')
    # Image encoder and Gene Autoencoder parameters ##########################################################################################################################################
    parser.add_argument('--image_encoder',                  type=str,               default='uni',                           help='Name of the image encoder to use')
    parser.add_argument('--gene_autoencoder',               type=str,               default=None,                            help='Name of the gene autoencoder to use. Only one by now: Transformer_encoder_mlp_decoder')
    parser.add_argument('--autoencoder_path',               type=str,               default='Pretrained_Encoders_Autoencoders/Genes_Autoencoders/Transformer_encoder_mlp_decoder/villacampa_lung_organoid/autoencoder_model.ckpt') 
    ##########################################################################################################################################################################################
    parser.add_argument('--debbug_wandb',                   type=str2bool,          default=False,                           help='Log in debbugs wandb')
    return parser


def data_normalization(data: torch.tensor, data_min, data_max):
    """ 
    This function receives a gene expression matrix and normalizes its content so that it has a range of [-1, 1].
    """
    norm_data = 2 * (data - data_min) / (data_max - data_min) - 1
    
    return norm_data

def data_denormalization(norm_data: torch.tensor, data_min: torch.tensor, data_max: torch.tensor):
    """
    This function receives a normalized gene expression matrix and denormalizes its content 
    back to the original range using the provided data_min and data_max.
    """
    denorm_data = (norm_data + 1) / 2 * (data_max - data_min) + data_min
    return denorm_data

