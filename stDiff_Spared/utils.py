import os
import numpy as np
import warnings
import torch
import anndata as ad
import argparse
# Later change again and use the softlink
from metrics_stdiff import get_metrics
import numpy as np
import torch
import squidpy as sq
import anndata as ad
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import copy
from spared_stdiff.datasets import get_dataset


warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]



def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for Diffusion Imputation Model')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset', type=str, default='villacampa_lung_organoid',  help='Dataset to use.')
    parser.add_argument('--prediction_layer',  type=str,  default='c_t_deltas', help='The prediction layer from the dataset to use.')
    parser.add_argument('--save_path',type=str,default='ckpt_W&B/',help='name model save path')
    parser.add_argument('--hex_geometry',                   type=bool,          default=True,                       help='Whether the geometry of the spots in the dataset is hexagonal or not.')
    parser.add_argument('--metrics_path',                   type=str,          default="output/metrics.csv",                       help='Path to the metrics file.')
    parser.add_argument("--loss_type",                        type=str,           default='noise',                                help='which type to calculate the loss with (noise, x_start, x_previous)')
    parser.add_argument("--concat_dim",                        type=int,           default=0,                                help='which dimension to concat the condition')
    parser.add_argument("--masked_loss",                        type=str2bool,           default=True,                                help='If True the loss if obtained only on masked data, if False the loos is obtained in all data')
    parser.add_argument("--model_type",                        type=str,           default="1D",                                help='If 1D is the Conv1D model and if 2D is the Conv2D model')
    parser.add_argument("--normalization_type",                        type=str,           default="1-1",                                help='If the normalization is done in range [-1, 1] (-1-1) or is done in range [0, 1] (0-1) or is none')
    parser.add_argument("--normalize_encoder",                        type=str,           default="none",                                help='If the normalization is done in range [-1, 1] (-1-1) or is done in range [0, 1] (0-1) or is none')
    parser.add_argument("--matrix",                        type=str2bool,           default=True,                                help='use transformer encoder decoder')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--seed',                   type=int,          default=1202,                       help='Seed to control initialization')
    parser.add_argument('--lr',type=float,default=0.0001,help='lr to use')
    parser.add_argument('--num_epoch', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--diffusion_steps', type=int, default=1500, help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size to train model')
    parser.add_argument('--optim_metric',                   type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--optimizer',                      type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--momentum',                       type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--step_size',                       type=float,         default=600,                         help='Step size to use in learning rate scheduler')
    parser.add_argument("--scheduler",                        type=str2bool,           default=True,                                help='Whether to use LR scheduler or not')
    # Autoencoder parameters #######################################################################################################################################################################
    parser.add_argument('--num_res_blocks',                   type=int,          default=8,                       help='Number of resnet blocks')
    parser.add_argument('--ch',                                type=int,        default=512,                        help='number of hidden dimensions in encoder')
    parser.add_argument('--ch_mult',                            type=tuple,        default=(1,2),                 help='Number of downsamplings')
    # Model parameters ########################################################################################################################################################################
    parser.add_argument('--depth', type=int, default=12, help='' )
    parser.add_argument('--hidden_size', type=int, default=1024, help='Size of latent space')
    parser.add_argument('--head', type=int, default=16, help='')
    # Transformer model parameters ############################################################################################################################################################
    parser.add_argument('--base_arch',                      type=str,           default='transformer_encoder',      help='Base architecture chosen for the imputation model.', choices=['transformer_encoder', 'MLP'])
    parser.add_argument('--transformer_dim',                type=int,           default=128,                        help='The number of expected features in the encoder/decoder inputs of the transformer.')
    parser.add_argument('--transformer_heads',              type=int,           default=1,                          help='The number of heads in the multiheadattention models of the transformer.')
    parser.add_argument('--transformer_encoder_layers',     type=int,           default=2,                          help='The number of sub-encoder-layers in the encoder of the transformer.')
    parser.add_argument('--transformer_decoder_layers',     type=int,           default=1,                          help='The number of sub-decoder-layers in the decoder of the transformer.')
    parser.add_argument('--include_genes',                  type=str2bool,      default=True,                       help='Whether or not to to include the gene expression matrix in the data inputed to the transformer encoder when using visual features.')
    parser.add_argument('--use_visual_features',            type=str2bool,      default=False,                      help='Whether or not to use visual features to guide the imputation process.')
    parser.add_argument('--use_double_branch_archit',       type=str2bool,      default=False,                      help='Whether or not to use the double branch transformer architecture when using visual features to guide the imputation process.')
    # Transformer model parameters ############################################################################################################################################################
    parser.add_argument('--num_workers',                    type=int,           default=0,                          help='DataLoader num_workers parameter - amount of subprocesses to use for data loading.')
    parser.add_argument('--num_assays',                     type=int,           default=10,                         help='Number of experiments used to test the model.')
    parser.add_argument('--sota',                           type=str,           default='pretrain',                 help='The name of the sota model to use. "None" calls main.py, "nn_baselines" calls nn_baselines.py, "pretrain" calls pretrain_backbone.py, and any other calls main_sota.py', choices=['None', 'pretrain', 'stnet', 'nn_baselines', "histogene"])
    parser.add_argument('--img_backbone',                   type=str,           default='ViT',                      help='Backbone to use for image encoding.', choices=['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin'])
    parser.add_argument('--use_pretrained_ie',              type=str,           default=True,                       help='Whether or not to use a pretrained image encoder model to get the patches embeddings.')
    parser.add_argument('--freeze_img_encoder',             type=str2bool,      default=False,                      help='Whether to freeze the image encoder. Only works when using pretrained model.')
    parser.add_argument('--matrix_union_method',            type=str,           default='concatenate',              help='Method used to combine the output of the gene processing transformer and the visual features processing transformer.', choices=['concatenate', 'sum'])
    parser.add_argument('--num_mlp_layers',                 type=int,           default=5,                          help='Number of layers stacked in the MLP architecture.')
    parser.add_argument('--ae_layer_dims',                  type=str2intlist,   default='512,384,256,128,64,128,256,384,512',                          help='Layer dimensions for ae in MLP base architecture.')
    parser.add_argument('--mlp_act',                        type=str,           default='ReLU',                     help='Activation function to use in the MLP architecture. Case sensitive, options available at: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity')
    parser.add_argument('--mlp_dim',                        type=int,           default=512,                        help='Dimension of the MLP layers.')
    parser.add_argument('--graph_operator',                 type=str,           default='None',                     help='The convolutional graph operator to use. Case sensitive, options available at: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers', choices=['GCNConv','SAGEConv','GraphConv','GATConv','GATv2Conv','TransformerConv', 'None'])
    parser.add_argument('--pos_emb_sum',                    type=str2bool,      default=False,                      help='Whether or not to sum the nodes-feature with the positional embeddings. In case False, the positional embeddings are only concatenated.')
    parser.add_argument('--h_global',                       type=str2h_list,    default='//-1//-1//-1',             help='List of dimensions of the hidden layers of the graph convolutional network.')
    parser.add_argument('--pooling',                        type=str,           default='None',                     help='Global graph pooling to use at the end of the graph convolutional network. Case sensitive, options available at but must be a global pooling: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers')
    parser.add_argument('--dropout',                        type=float,         default=0.0,                        help='Dropout to use in the model to avoid overfitting.')
    # Data masking parameters ################################################################################################################################################################
    parser.add_argument('--neighborhood_type',              type=str,           default='nn_distance',              help='The method used to select the neighboring spots.', choices=['circular_hops', 'nn_distance'])
    parser.add_argument('--num_neighs',                     type=int,           default=18,                          help='Amount of neighbors to consider for context during imputation.')
    parser.add_argument('--num_hops',                       type=int,           default=1,                          help='Amount of graph hops to consider for context during imputation if neighborhoods are built based on proximity rings.')
    # Visualization parameters ################################################################################################################################################################
    parser.add_argument('--gene_id',                        type=int,           default=0,                          help='Gene ID to plot.')
    # W&B usage parameters ####################################################################################################################################################################
    parser.add_argument('--debbug_wandb',                     type=str2bool,      default=False,                       help='Select if the experiment is logged in w&b debbug folder. If True, then is for debbuging purposes.')
    parser.add_argument('--vlo',                             type=str2bool,      default=False,                       help='Select just the vlo dataset, dont use get_dataset function.')
    return parser


def normalize_to_cero_to_one(X, X_max, X_min):
    # Apply the normalization formula to 0-1
    X_norm = (X-X_min)/(X_max - X_min)
    return X_norm

def denormalize_from_cero_to_one(X_norm, X_max, X_min):
    # Apply the denormalization formula 
    X_denorm = (X_norm*(X_max - X_min)) + X_min
    return X_denorm    
    
def normalize_to_minus_one_to_one(X, X_max, X_min):
    # Apply the normalization formula to -1-1
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
    return X_norm

def denormalize_from_minus_one_to_one(X_norm, X_max, X_min):
    # Apply the denormalization formula 
    X_denorm = ((X_norm + 1) / 2) * (X_max - X_min) + X_min
    return X_denorm

def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
    """ Get expression deltas from the mean.

    Compute the deviations from the mean expression of each gene in ``adata.layers[from_layer]`` and save them
    in ``adata.layers[to_layer]``. Also add the mean expression of each gene to ``adata.var[f'{from_layer}_avg_exp']``.
    Average expression is computed using only train data determined by the ``adata.obs['split']`` column. However, deltas
    are computed for all observations.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have expression values in ``adata.layers[from_layer]``. Must also have the ``adata.obs['split']`` column with ``'train'`` values.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with the deltas (``adata.layers[to_layer]``) and mean expression (``adata.var[f'{from_layer}_avg_exp']``) information.
    """

    # Get the expression matrix of both train and global data
    glob_expression = adata.to_df(layer=from_layer)
    train_expression = adata[adata.obs['split'] == 'train'].to_df(layer=from_layer)

    # Define scaler
    scaler = StandardScaler(with_mean=True, with_std=False)

    # Fit the scaler to the train data
    scaler = scaler.fit(train_expression)
    
    # Get the centered expression matrix of the global data
    centered_expression = scaler.transform(glob_expression)

    # Add the deltas to adata.layers[to_layer]	
    adata.layers[to_layer] = centered_expression

    # Add the mean expression to adata.var[f'{from_layer}_avg_exp']
    adata.var[f'{from_layer}_avg_exp'] = scaler.mean_

    # Return the updated AnnData object
    return adata

def get_mask_prob_tensor(masking_method, dataset, mask_prob=0.3, scale_factor=0.8):
    """
    This function calculates the probability of masking each gene present in the expression matrix. 
    Within this function, there are three different methods for calculating the masking probability, 
    which are differentiated by the 'masking_method' parameter. 
    The return value is a vector of length equal to the number of genes, where each position represents
    the masking probability of that gene.
    
    Args:
        masking_method (str): parameter used to differenciate the method for calculating the probabilities.
        dataset (SpatialDataset): the dataset in a SpatialDataset object.
        mask_prob (float): masking probability for all the genes. Only used when 'masking_method = mask_prob' 
        scale_factor (float): maximum probability of masking a gene if masking_method == 'scale_factor'
    Return:
        prob_tensor (torch.Tensor): vector with the masking probability of each gene for testing. Shape: n_genes  
    """

    # Convert glob_exp_frac to tensor
    glob_exp_frac = torch.tensor(dataset.adata.var.glob_exp_frac.values, dtype=torch.float32)
    # Calculate the probability of median imputation
    prob_median = 1 - glob_exp_frac

    if masking_method == "prob_median":
        # Calculate masking probability depending on the prob median
        # (the higher the probability of being replaced with the median, the higher the probability of being masked).
        prob_tensor = prob_median/(1-prob_median)

    elif masking_method == "mask_prob":
        # Calculate masking probability according to mask_prob parameter
        # (Mask everything with the same probability)
        prob_tensor = mask_prob/(1-prob_median)

    elif masking_method == "scale_factor":
        # Calculate masking probability depending on the prob median scaled by a factor
        # (Multiply by a factor the probability of being replaced with median to decrease the masking probability).
        prob_tensor = prob_median/(1-prob_median)
        prob_tensor = prob_tensor*scale_factor
        
    # If probability is more than 1, set it to 1
    prob_tensor[prob_tensor>1] = 1

    return prob_tensor

def mask_exp_matrix(adata: ad.AnnData, pred_layer: str, mask_prob_tensor: torch.Tensor, device):
    """
    This function recieves an adata and masks random values of the pred_layer based on the masking probability of each gene, then saves the masked matrix in the corresponding layer. 
    It also saves the final random_mask for metrics computation. True means the values that are real in the dataset and have been masked for the imputation model development.
    
    Args:
        adata (ad.AnnData): adata of the data split that will be masked and imputed.
        pred_layer (str): indicates the adata.layer with the gene expressions that should be masked and later reconstructed. Shape: spots_in_adata, n_genes
        mask_prob_tensor (torch.Tensor):  tensor with the masking probability of each gene for testing. Shape: n_genes
    
    Return:
        adata (ad.AnnData): adata of the data split with the gene expression matrix already masked and the corresponding random_mask in adata.layers.
    """

    # Extract the expression matrix
    expression_mtx = torch.tensor(adata.layers[pred_layer])
    # Calculate the mask based on probability tensor
    random_mask = torch.rand(expression_mtx.shape).to(device) < mask_prob_tensor.to(device)
    median_imp_mask = torch.tensor(adata.layers['mask']).to(device)
    # True es dato real y False es dato imputado con mediana
    # Combine random mask with the median imputation mask
    random_mask = random_mask.to(device) & median_imp_mask
    # Que este masqueado y ademas sea un dato real (no masquea datos imputados por mediana)
    # Mask chosen values.
    expression_mtx[random_mask] = 0
    # Save masked expression matrix in the data_split annData
    adata.layers['masked_expression_matrix'] = np.asarray(expression_mtx.cpu())
    #Save final mask for metric computation
    adata.layers['random_mask'] = np.asarray(random_mask.cpu())

    return adata
"""
def decode(imputation, model_decoder):
    imputation = torch.tensor(imputation)
    decoded_data = model_decoder(imputation[:,:,0])
    return decoded_data
"""
def decode(imputation, model_decoder, batch_size=128):
    # Convert imputation to a PyTorch tensor if it's not already
    imputation = torch.tensor(imputation[:,:,0], dtype=torch.float32)
    dataset = TensorDataset(imputation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_decoder.to("cuda")
    model_decoder.eval()  
    decoded_batches = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to("cuda")
            #LSTM
            #decoder_input = torch.zeros(batch.shape[0], batch.shape[1])
            #decoded_batch, _ = model_decoder.decoder(decoder_input, batch)
            
            #TRAMSFORMER
            decoded_batch = model_decoder.decoder(batch)
            
            decoded_batches.append(decoded_batch.cpu())  # Move to CPU for storage if needed
    decoded_data = torch.cat(decoded_batches, dim=0)

    return decoded_data

#TODO: acoplar a decoder
def decode_transformers(imputation, model_decoder, batch_size=128):
    # Convert imputation to a PyTorch tensor if it's not already
    imputation = torch.tensor(imputation, dtype=torch.float32).permute(0,2,1)
    dataset = TensorDataset(imputation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_decoder.to("cuda")
    model_decoder.eval()  
    decoded_batches = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to("cuda")

            #TRAMSFORMER
            decoded_batch = model_decoder.decoder(batch)
        
            decoded_batches.append(decoded_batch.cpu())  # Move to CPU for storage if needed
    decoded_data = torch.cat(decoded_batches, dim=0)

    return decoded_data


from torch.utils.data import DataLoader, TensorDataset

# Function to encode data and return a DataLoader
def encode_data_and_create_dataloader(data_loader, model_autoencoder, device, batch_size, is_shuffle):
    breakpoint()
    encoded_data = []
    encoded_masked_data = []
    original_masks = []
    # Iterate through the DataLoader
    for data, masked_data, mask in tqdm(data_loader):
        # Move data to the correct device (GPU/CPU)
        data = data.to(device).unsqueeze(dim=1)
        masked_data = masked_data.to(device).unsqueeze(dim=1)

        # Pass data and masked_data through the encoder
        with torch.no_grad():  # Disable gradient computation
            encoded = model_autoencoder.encoder(data)
            encoded_masked = model_autoencoder.encoder(masked_data)
        
        # Append encoded data, encoded masked_data, and original mask
        encoded_data.append(encoded.cpu())
        encoded_masked_data.append(encoded_masked.cpu())
        original_masks.append(mask.cpu())  # Keep masks as they are

    # Concatenate encoded data, encoded masked data, and masks
    encoded_data = torch.cat(encoded_data)
    encoded_masked_data = torch.cat(encoded_masked_data)
    original_masks = torch.cat(original_masks)
    # Create a new DataLoader with the required elements
    encoded_dataset = TensorDataset(encoded_data, encoded_masked_data, original_masks)
    generator = torch.Generator(device='cuda')
    encoded_dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=is_shuffle, generator=generator)

    return encoded_dataloader


def get_dit_predictions(adata, dataloader, data, masked_data, model, mask, mask_extreme_completion, max_norm, min_norm, avg_tensor, diffusion_step, device, args, model_autoencoder):
    # To avoid circular imports
    from model_stDiff.stDiff_scheduler import NoiseScheduler
    from model_stDiff.sample import sample_stDiff
    """
    Function designed to do inference for validation and test steps.
    Params:
        -dataloader (Pytorch.Dataloader): dataloader containing batches, each element has -> (st_data, st_masked_data, mask)
        -data (np.array): original st data
        -masked_data (np.array): masked original st data
        -model (diffusion model): diffusion model to do inference
        -mask (np.array): mask used for data
        -max_norm (float): max value of st data
        -diffusion_step (int): diffusion step set in argparse
        -device (str): device cpu or cuda

    Returns:
        -metrics_dict (dict): dictionary with all the metrics
    """
    # Sample model using test set
    #gt = masked_data
    #BoDiffusion
    gt = data
    # Define noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    
    # inference using test split
    imputation = sample_stDiff(model,
                        dataloader=dataloader,
                        noise_scheduler=noise_scheduler,
                        args=args,
                        device=device,
                        mask=mask,
                        gt=gt,
                        num_step=diffusion_step,
                        sample_shape=gt.shape,
                        is_condi=True,
                        sample_intermediate=diffusion_step,
                        is_classifier_guidance=False,
                        omega=0.2)
       
    if args.normalization_type == "0-1":
        #Normalización 0 a 1 
        #data = denormalize_from_cero_to_one(data, max_norm, min_norm)
        imputation = denormalize_from_cero_to_one(imputation, max_norm, min_norm)
    elif args.normalization_type == "1-1":
        #Normalización -1 a 1
        #data = denormalize_from_minus_one_to_one(data, max_norm, min_norm)
        imputation = denormalize_from_minus_one_to_one(imputation, max_norm, min_norm)
    else:
        print("no se aplica ningun tipo de normalizacion")
    
    return imputation


def inference_function(adata, dataloader, data, masked_data, model, mask, mask_extreme_completion, max_norm, min_norm, avg_tensor, diffusion_step, device, args, model_decoder, max_enc, min_enc):
    # To avoid circular imports
    from model_stDiff.stDiff_scheduler import NoiseScheduler
    from model_stDiff.sample import sample_stDiff
    """
    Function designed to do inference for validation and test steps.
    Params:
        -dataloader (Pytorch.Dataloader): dataloader containing batches, each element has -> (st_data, st_masked_data, mask)
        -data (np.array): original st data
        -masked_data (np.array): masked original st data
        -model (diffusion model): diffusion model to do inference
        -mask (np.array): mask used for data
        -max_norm (float): max value of st data
        -diffusion_step (int): diffusion step set in argparse
        -device (str): device cpu or cuda

    Returns:
        -metrics_dict (dict): dictionary with all the metrics
    """
    # Sample model using test set
    #gt = masked_data
    #BoDiffusion
    gt = data
    # Define noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    
    # inference using test split
    imputation = sample_stDiff(model,
                        dataloader=dataloader,
                        noise_scheduler=noise_scheduler,
                        args=args,
                        device=device,
                        mask=mask,
                        gt=gt,
                        num_step=diffusion_step,
                        sample_shape=gt.shape,
                        is_condi=True,
                        sample_intermediate=diffusion_step,
                        is_classifier_guidance=False,
                        omega=0.2)
    
    if args.normalization_type == "0-1":
        #Normalización 0 a 1 
        data = denormalize_from_cero_to_one(data, max_norm, min_norm)
        imputation = denormalize_from_cero_to_one(imputation, max_norm, min_norm)
    elif args.normalization_type == "1-1":
        #Normalización -1 a 1
        data = denormalize_from_minus_one_to_one(data, max_norm, min_norm)
        imputation = denormalize_from_minus_one_to_one(imputation, max_norm, min_norm)
    else:
        print("no se aplica ningun tipo de normalizacion")
    
    import torch.nn.functional as F
    print("Latent Input - Mean:", data.mean().item())
    print("Latent Output - Mean:", imputation.mean().item())
    print("Latent Input - Std Dev:", data.std().item())
    print("Latent Output - Std Dev:", imputation.std().item())
    
    imputation = torch.tensor(imputation)
    perturbation = torch.randn_like(imputation) * 0.01
    dec_imputation = decode(imputation=imputation, model_decoder=model_decoder, batch_size=args.batch_size)
    dec_perturbation = decode(imputation=perturbation, model_decoder=model_decoder, batch_size=args.batch_size)
    mse_pre = F.mse_loss(imputation, perturbation)
    print("mse pre: ", mse_pre)
    mse_post = F.mse_loss(dec_imputation, dec_perturbation)
    print("mse post: ", mse_post)
    
    dit_imputation = torch.tensor(imputation[:,:,0], dtype=torch.float32)
    dit_data = torch.tensor(data[:,:,0], dtype=torch.float32)
    dit_mse = F.mse_loss(dit_data, dit_imputation)
    print("mse del dit: ", dit_mse)
    #breakpoint()
    
    #Decoded imputation data
    if args.matrix:
        imputation = decode_transformers(imputation=imputation, model_decoder=model_decoder, batch_size=args.batch_size)
        imputation = imputation[:,0,:]
    else:
        imputation = decode(imputation=imputation, model_decoder=model_decoder, batch_size=args.batch_size)
    imputation = imputation.detach().cpu().numpy()
    
    # Normalize data
    if args.normalize_encoder == "1-1":
        imputation = denormalize_from_minus_one_to_one(imputation, max_enc[0].item(), min_enc[0].item())
    
    #mask_boolean = (1-mask).astype(bool) #for partial completion
    mask_boolean = mask_extreme_completion.astype(bool) #for extreme completion
    
    #Evaluate only on spot central
    mask_boolean = mask_boolean[:,:,0]
    #data = data[:,:,0]
    #GT data
    adata_1024, adata_128 = adata
    data = adata_1024.layers[args.prediction_layer]
    #data = adata_128.layers[args.prediction_layer]
    #imputation = imputation[:,0,:]
    #avg_tensor = None
    if avg_tensor != None:
        # Sumar deltas más la expresión del data
        
        #vector input
        #data_tensor = torch.tensor(data)
        #data_tensor = data_tensor + avg_tensor
        #data_tensor = np.array(data_tensor.cpu())
        #data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
        # Sumar deltas más la expresión de la imputacion
        imputation_tensor = torch.tensor(imputation)
        imputation_tensor = imputation_tensor + avg_tensor
        imputation_tensor = np.array(imputation_tensor.cpu())

    imputation_tensor = torch.tensor(imputation_tensor, dtype=torch.float32)
    #mse = F.mse_loss(data_tensor, imputation_tensor)
    #print("mse de 128 genes: ", mse)
    
    #matrix input
    data_128 = adata_1024.layers["c_t_log1p"]
    data_128_tensor = torch.tensor(data_128)
    mse_final = F.mse_loss(imputation_tensor[mask_boolean], data_128_tensor[mask_boolean])
    print(mse_final)
    mask_boolean = torch.tensor(mask_boolean)
    metrics_dict = get_metrics(data_128_tensor.cpu(), imputation_tensor.cpu(), mask_boolean.cpu())
    
    #vector input
    #mse_final = F.mse_loss(data_tensor[mask_boolean], imputation_tensor[mask_boolean])
    #print(mse_final)
    #mask_boolean = torch.tensor(mask_boolean)
    #metrics_dict = get_metrics(data_tensor.cpu(), imputation_tensor.cpu(), mask_boolean.cpu())
    
    return metrics_dict, dit_imputation


def build_neighborhood_from_distance(adata, pred_layer, num_neighs = 6):
    """
    This function gets the closest n neighbors of the spot in index idx and returns the final neighborhood gene expression matrix,
    as well as the mask that indicates which elements are missing in the original data. If the datasets has already been randomly 
    masked, it will also return the corresponding matrix.
    """
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=num_neighs)
    adj_mat = torch.tensor(adata.obsp['spatial_connectivities'].todense())
    expression_mtx = torch.tensor(adata.layers[pred_layer])
    
    # Define neighbors dict
    neighbors_dict_index = {}
    # Iterate through the rows of the output matrix
    for idx in range(expression_mtx.shape[0]):
        # Get gt expression for idx spot and its nn
        row_indices = adj_mat[:, idx].nonzero()
        spot_exp = expression_mtx[idx].unsqueeze(dim=0)
        nn_exp = expression_mtx[adj_mat[:,idx]==1.]
        exp_matrix = torch.cat((spot_exp, nn_exp), dim=0).type('torch.FloatTensor') # Original dtype was 'torch.float64'

        # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
        neighbors_dict_index[idx] = exp_matrix

    return neighbors_dict_index

        
def get_spatial_neighbors(adata: ad.AnnData, num_neighs: int, hex_geometry: bool) -> dict:
    """
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside a n_hops neighborhood of the observation.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): the size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                used to compute the spatial neighbors and only true for visium datasets.

    Returns:
        dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
    """
    
    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=num_neighs) # Hexagonal visium case
        #sc.pp.neighbors(adata, n_neighbors=6, knn=True)
    # Get the adjacency matrix (binary matrix of shape spots x spots)
    adj_matrix = adata.obsp['spatial_connectivities']
    
    # Define power matrix
    power_matrix = adj_matrix.copy() #(spots x spots)
    # Define the output matrix
    output_matrix = adj_matrix.copy() #(spots x spots)

    # Zero out the diagonal
    output_matrix.setdiag(0)  #(spots x spots) Apply 0 diagonal to avoid "auto-paths"
    # Threshold the matrix to 0 and 1
    output_matrix = output_matrix.astype(bool).astype(int)

    # Define neighbors dict
    neighbors_dict_index = {}
    # Iterate through the rows of the output matrix
    for i in range(output_matrix.shape[0]):
        # Get the non-zero elements of the row (non zero means a neighbour)
        non_zero_elements = output_matrix[:,i].nonzero()[0]
        # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
        #Key: int number (id of each spot) -> Value: list of spots ids
        neighbors_dict_index[i] = [i] + list(non_zero_elements)
    
    # Return the neighbors dict
    return neighbors_dict_index


def build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx, autoencoder_model, args):
    # Get nn indexes for the n_hop required
    nn_index_list = spatial_neighbors[idx] #Obtain the ids of the spots that are neigbors of idx
    # Normalize data
    
    min_exp = expression_mtx.min()
    max_exp = expression_mtx.max()
    if args.normalize_encoder == "1-1":
        expression_mtx = normalize_to_minus_one_to_one(expression_mtx, max_exp, min_exp)
    #Index the expression matrix (X processed) and obtain the neccesary data
    exp_matrix = expression_mtx[nn_index_list].type('torch.FloatTensor')
    #breakpoint()
    if autoencoder_model != None:
        exp_matrix = autoencoder_model.encoder(exp_matrix.to("cuda"))
    return exp_matrix, max_exp, min_exp #shape (n_neigbors, n_genes)

def get_neigbors_dataset(adata, prediction_layer, num_hops, autoencoder_model, args):
    """
    This function recives the name of a dataset and pred_layer. Returns a list of len = number of spots, each position of the list is an array 
    (n_neigbors + 1, n_genes) that has the information about the neigbors of the corresponding spot.
    """
    all_neighbors_info = {}
    max_min_info = {}
    #Dataset all info
    dataset = adata
    #get dataset splits
    splits = dataset.obs["split"].unique().tolist()
    #get num neighs
    num_neighs = 0
    for hop in range(1, num_hops+1):
        num_neighs += 6*hop
    #iterate over split adata
    for split in splits:
        split_neighbors_info = []
        adata = dataset[dataset.obs["split"]==split]
        # get slides for the correspoding split
        #slides = adata.obs["slide_id"].unique().tolist()
        #iterate over slides and get the neighbors info for every slide
        #Get dict with all the neigbors info for each spot in the dataset
        spatial_neighbors = get_spatial_neighbors(adata, num_neighs=num_neighs, hex_geometry=True)
        #Expression matrix (already applied post-processing)
        expression_mtx = torch.tensor(adata.layers[prediction_layer]) 
        for idx in tqdm(spatial_neighbors.keys()):
            # Obtainb expression matrix just wiht the neigbors of the corresponding spot
            neigbors_exp_matrix, max_enc, min_enc = build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx, autoencoder_model, args)
            split_neighbors_info.append(neigbors_exp_matrix)
            
        #append split neighbors info into the complete list
        all_neighbors_info[split] = split_neighbors_info
        max_min_info[split] = [max_enc, min_enc]

    return all_neighbors_info, max_min_info

#TODO: acople a tu encoder
def encode_transformers(list_nn, model_autoencoder, batch_size=128):
    # Convert imputation to a PyTorch tensor if it's not already
    encoded_list_nn = {}
    for split in list_nn.keys():
        data = torch.stack(list_nn[split])
        data = torch.tensor(data)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        model_autoencoder.to("cuda")
        model_autoencoder.eval()  

        encoded_batches = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to("cuda")
                encoded_batch = model_autoencoder.encoder(batch)
                encoded_batches.append(encoded_batch.cpu())  # Move to CPU for storage if needed
        encoded_data = torch.cat(encoded_batches, dim=0)
        encoded_list_nn[split] = encoded_data
        encoded_list_nn[split] = [encoded_list_nn[split][i] for i in range(encoded_list_nn[split].size(0))]
    return encoded_list_nn


def define_split_nn_mat(list_nn, list_nn_masked, split, args):
    
    """This function receives a list of all the spots and corresponding neighbors, both masked and unmasked and returns
    the st_data, st_masked_data and mask, where bothe the center spot and its neighbors are masked and used for completion.
    The data is returned as a list of matrices of shape (num_neighbors, num_genes) 

    Args:
        list_nn (_type_): list of all spots and 6 neighbors
        list_nn_masked (_type_): lista of all spots and 6 neighbors masked
        split (_type_): train, valid or test split

    Returns:
        tuple: contaning the st_data, the masked st_data, the mask and the max value used for normalization 
    """
    
    # Definir lista segun el split
    list_nn = list_nn[split]
    list_nn_masked = list_nn_masked[split]
    
    #Convertir la lista de tensores en un solo tensor tridimensional
    tensor_stack_nn = torch.stack(list_nn)
    st_data = tensor_stack_nn
    #st_data = tensor_stack_nn.reshape(tensor_stack_nn.size(0), -1)
    st_data = tensor_stack_nn.permute(0, 2, 1)
    
    tensor_stack_nn_masked = torch.stack(list_nn_masked)
    st_data_masked = tensor_stack_nn_masked
    st_data_masked = tensor_stack_nn_masked.permute(0, 2, 1)

    mask = st_data_masked!=0
    mask = mask.int()
    mask[:,:,1:] = 1
    
    #Convertir a numpy array
    st_data = st_data.detach().cpu().numpy()
    st_data_masked = st_data_masked.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    
    # Normalización
    max_data = st_data.max()
    min_data = st_data.min()
    
    if args.normalization_type == "0-1":
        #print("normalización 0 a 1")
        #Normalización 0 a 1 
        st_data = normalize_to_cero_to_one(st_data, max_data, min_data)
        st_data_masked = normalize_to_cero_to_one(st_data_masked, max_data, min_data)*mask
    elif args.normalization_type == "1-1":
        #print("normalización -1 a 1")
        #Normalización -1 a 1
        st_data = normalize_to_minus_one_to_one(st_data, max_data, min_data)
        st_data_masked = normalize_to_minus_one_to_one(st_data_masked, max_data, min_data)*mask
    else:
        print("no se aplica ningun tipo de normalizacion")
    
    
    return st_data, st_data_masked, mask, max_data, min_data

def mask_extreme_prediction(list_nn):
    list_nn_masked = {key: [copy.deepcopy(tensor.detach().cpu()) for tensor in value] for key, value in list_nn.items()}
    for i in list_nn_masked.keys():
        for j in range(len(list_nn_masked[i])):
            list_nn_masked[i][j][0][:] = 0
    return list_nn_masked
    
def get_mask_extreme_completion(adata, mask, genes):
    mask_extreme_completion = copy.deepcopy(mask)
    imp_values = adata.layers["mask"] #True en los valores reales y False en los valores imputados
    mask_extreme_completion[imp_values] = 1
    #mask_extreme_completion[:,:,0:] = 1 #TODO: eliminar
    mask_extreme_completion[:,:,1:] = 0
    genes = np.array(genes)[:,np.newaxis]
    mask_extreme_completion = mask_extreme_completion*genes
    for i in range(0, mask_extreme_completion.shape[0]):
        idx_mask = np.where(mask_extreme_completion[0,:,0]==1)[0]
        idx_genes = np.where(genes==1)[0]
        bool_mask = np.isin(idx_mask, idx_genes).all()
        assert bool_mask, "error en mascara"
        
    return mask_extreme_completion


def get_mask_extreme_completion_128(adata, mask):
    mask_extreme_completion = copy.deepcopy(mask)
    imp_values = adata.layers["mask"] #True en los valores reales y False en los valores imputados
    mask_extreme_completion[imp_values] = 1
    mask_extreme_completion[:,:,1:] = 0
    return mask_extreme_completion

def append_data(args, list_nn, train_data, val_data, test_data, max_value, min_value):
    for i, split in enumerate(list_nn):
        for spots in split:
            data = spots.permute(1,0).unsqueeze(dim=0)
            if i == 0:
                if args.normalization_type == "1-1":
                    data = normalize_to_minus_one_to_one(data, max_value, min_value)
                train_data.append(data)
            elif i == 1:
                if args.normalization_type == "1-1":
                    data = normalize_to_minus_one_to_one(data, max_value, min_value)
                val_data.append(data)
            elif i == 2:
                if args.normalization_type == "1-1":
                    data = normalize_to_minus_one_to_one(data, max_value, min_value)
                test_data.append(data)
    
    return train_data, val_data, test_data

def join_dataset(args, dataset_names, pred_layer):
    train_data = []
    val_data = []
    test_data = []
    
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name)
        adata = dataset.adata  
        list_nn = get_neigbors_dataset(adata, pred_layer, num_hops=1)
        concatenated_array = np.concatenate([np.array(sublist) for sublist in list_nn])
        max_value = np.max(concatenated_array)
        min_value = np.min(concatenated_array)
    
        train_data, val_data, test_data = append_data(args=args,
                                                      list_nn=list_nn, 
                                                      train_data=train_data, 
                                                      val_data=val_data, 
                                                      test_data=test_data,
                                                      max_value=max_value,
                                                      min_value=min_value)
    return train_data, val_data, test_data, max_value, min_value

def get_autoencoder_data(adata, prediction_layer, args):
    data = {}
    splits = adata.obs["split"].unique()
    for split in splits:
        split_adata = adata[adata.obs["split"]==split]
        data[split] = split_adata.layers[prediction_layer]
    
    combined_data = np.concatenate([arr for tup in data.values() for arr in tup])
    min_value = np.min(combined_data)
    max_value = np.max(combined_data)
    #min_value = 0
    #max_value = 0
    
    if args.normalize_encoder == "1-1":
        for split in data.keys():
            print("normalizing data")
            data[split] = normalize_to_minus_one_to_one(data[split], max_value, min_value)
        #data[split] = normalize_to_cero_to_one(data[split], max_value, min_value)
    
    return data, min_value, max_value

def load_autoencoder_data(data_loader, model):
    auto_pred = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            # Move data to the specified device
            inputs = data[0].to("cuda")
            inputs = inputs.float()
            model = model.to("cuda")
            # Make predictions
            outputs = model.encoder(inputs)
            # Move outputs to CPU and convert to NumPy if needed
            auto_pred.append(outputs.cpu().numpy())

    auto_pred = np.concatenate(auto_pred, axis=0)
 
    return auto_pred

def add_noise(inputs, noise_factor=0.01):
    """
    Adds Gaussian noise to the inputs for robust training.
    """
    noise = noise_factor * torch.randn_like(inputs)
    return inputs + noise



# Load data
from spared.filtering import *
from spared.layer_operations import *
from spared.denoising import *
from spared.gene_features import *
from tqdm import tqdm
from spared.gene_features import *


def get_new_adatas(path):
    dataset_name = path.split("/")[-3]
    
    dataset = get_dataset(dataset_name)
    adata = dataset.adata 
    param_dict = dataset.param_dict
    
    if "mouse" in dataset_name:
        param_dict["organism"] = "mouse"
    else:
        param_dict["organism"] = "human"

    param_dict["hex_geometry"] = True
    
    raw_adata = ad.read_h5ad(path)
    #breakpoint()
    #adata = filter_dataset(raw_adata, param_dict)
    #breakpoint()
    adata = raw_adata.copy()
    adata = get_exp_frac(adata)
    adata = get_glob_exp_frac(adata)
    adata.layers['counts'] = adata.X.toarray()
    adata = tpm_normalization(adata, param_dict["organism"], from_layer='counts', to_layer='tpm')
    adata = log1p_transformation(adata, from_layer='tpm', to_layer='log1p')
    adata = denoising.median_cleaner(adata, from_layer='log1p', to_layer='d_log1p', n_hops=4, hex_geometry=param_dict["hex_geometry"])
    adata = gene_features.compute_moran(adata, hex_geometry=param_dict["hex_geometry"], from_layer='d_log1p') 
    total_genes = adata.shape[1]
    adata = filtering.filter_by_moran(adata, n_keep=total_genes, from_layer='d_log1p')
    adata = combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
    adata = combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')
    adata = get_deltas(adata, from_layer='log1p', to_layer='deltas')
    adata = get_deltas(adata, from_layer='d_log1p', to_layer='d_deltas')
    adata = get_deltas(adata, from_layer='c_log1p', to_layer='c_deltas')
    adata = get_deltas(adata, from_layer='c_d_log1p', to_layer='c_d_deltas')
    adata.layers['mask'] = adata.layers['tpm'] != 0
    adata, _  = denoising.spackle_cleaner(adata=adata, dataset=dataset_name, from_layer="c_d_log1p", to_layer="c_t_log1p", device = "cuda")
    adata = get_deltas(adata, from_layer='c_t_log1p', to_layer='c_t_deltas')   
    #dict_genes[dataset_name] = adata.shape[1]
    adata.write(f'/home/dvegaa/ST_Diffusion/stDiff_Spared/adata_1024/{dataset_name}_1024.h5ad')
    
    return adata
            
def sort_adatas(adata, adata_128):
    adata.var.reset_index(drop=True, inplace=True)
    adata_128.var.reset_index(drop=True, inplace=True)

    # The previous index (gene_ids) is now a column; create a numeric index
    adata.var.index = range(adata.var.shape[0])
    adata_128.var.index = range(adata_128.var.shape[0])

    adata_sorted = adata.copy()
    adata_128_sorted = adata_128.copy()
    
    # Sort .var (genes) by index
    adata_sorted.var["original_index"] = adata_sorted.var.index
    adata_sorted.var = adata_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

    adata_128_sorted.var["original_index"] = adata_128_sorted.var.index
    adata_128_sorted.var = adata_128_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

    #Get indices
    sorted_indices = adata_sorted.var["original_index"].to_numpy()
    sorted_indices = [int(idx) for idx in sorted_indices]
    
    sorted_indices_128 = adata_128_sorted.var["original_index"].to_numpy()
    sorted_indices_128 = [int(idx) for idx in sorted_indices_128]
    
    # Reorder the main data matrix (.X) to match the new gene order
    #adata_sorted = adata[:, sorted_indices]
    #adata_128_sorted = adata_128[:, sorted_indices_128]

    # Reorder all layers to match the new gene order
    for layer in adata.layers.keys():
        adata_sorted.layers[layer] = adata.layers[layer][:, sorted_indices]

    for layer in adata_128.layers.keys():
        adata_128_sorted.layers[layer] = adata_128.layers[layer][:, sorted_indices_128]

    return adata_sorted, adata_128_sorted

def get_1204_adata(adata, adata_1024, genes):
    count = 1024 - adata.shape[1]
    if adata_1024.var.index.name != "gene_ids":
        adata_1024.var = adata_1024.var.set_index("gene_ids", drop=False)
    
    moran_dict = adata_1024.var["d_log1p_moran"].to_dict()
    
    genes_to_add = []

    for key, value in moran_dict.items():
        if len(genes_to_add) < count:
            if key not in genes:
                genes_to_add.append(key)
    
    for gene in tqdm(genes_to_add):
        adata_gene = adata_1024[:, adata_1024.var["gene_ids"] == gene]
        adata = ad.concat([adata, adata_gene], axis=1, merge="same")

    return adata
