
import os
import numpy as np
import warnings
import torch
import anndata as ad
import csv
import argparse
import matplotlib.pyplot as plt
from spared.metrics import get_metrics
from spared.datasets import get_dataset
import numpy as np
import torch
import squidpy as sq
import anndata as ad
from tqdm import tqdm
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import copy

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
    parser.add_argument('--prediction_layer',  type=str,  default='c_d_log1p', help='The prediction layer from the dataset to use.')
    parser.add_argument('--save_path',type=str,default='ckpt_W&B/',help='name model save path')
    parser.add_argument('--hex_geometry',                   type=bool,          default=True,                       help='Whether the geometry of the spots in the dataset is hexagonal or not.')
    parser.add_argument('--metrics_path',                   type=str,          default="output/metrics.csv",                       help='Path to the metrics file.')
    parser.add_argument("--loss_type",                        type=str,           default='noise',                                help='which type to calculate the loss with (noise, x_start, x_previous)')
    parser.add_argument("--concat_dim",                        type=int,           default=0,                                help='which dimension to concat the condition')
    parser.add_argument("--masked_loss",                        type=str2bool,           default=True,                                help='If True the loss if obtained only on masked data, if False the loos is obtained in all data')
    parser.add_argument("--model_type",                        type=str,           default="1D",                                help='If 1D is the Conv1D model and if 2D is the Conv2D model')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--seed',                   type=int,          default=1202,                       help='Seed to control initialization')
    parser.add_argument('--lr',type=float,default=0.0001,help='lr to use')
    parser.add_argument('--num_epoch', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--diffusion_steps', type=int, default=1500, help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size to train model')
    parser.add_argument('--optim_metric',                   type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--optimizer',                      type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--momentum',                       type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--step_size',                       type=float,         default=600,                         help='Step size to use in learning rate scheduler')
    parser.add_argument("--scheduler",                        type=str2bool,           default=True,                                help='Whether to use LR scheduler or not')
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
    return parser


def normalize_to_minus_one_to_one(X, X_max, X_min):
    # Apply the normalization formula
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
    return X_norm

def denormalize_from_minus_one_to_one(X_norm, X_min, X_max):
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

def inference_function(dataloader, data, masked_data, model, mask, mask_extreme_completion, max_norm, min_norm, avg_tensor, diffusion_step, device, args):
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

    #mask_boolean = (1-mask).astype(bool) #for partial completion
    mask_boolean = mask_extreme_completion.astype(bool) #for extreme completion
    
    #Evaluate only on spot central
    mask_boolean = mask_boolean[:,:,0]
    data = data[:,:,0]
    imputation = imputation[:,:,0]
        
    #data = data*max_norm
    #imputation = imputation*max_norm
    
    data = denormalize_from_minus_one_to_one(data, min_norm, max_norm)
    imputation = denormalize_from_minus_one_to_one(imputation, min_norm, max_norm)
    
    if avg_tensor != None:
        # Sumar deltas más la expresión del data
        data_tensor = torch.tensor(data)
        data = data_tensor + avg_tensor
        data = np.array(data.cpu())
        # Sumar deltas más la expresión de la imputacion
        imputation_tensor = torch.tensor(imputation)
        imputation = imputation_tensor + avg_tensor
        imputation = np.array(imputation.cpu())
    metrics_dict = get_metrics(data, imputation, mask_boolean)
    
    return metrics_dict, imputation

def define_splits(dataset, split:str, pred_layer:str):
    """
    Function that extract the desired split from the dataset and then prepare neccesary data for 
    the dataloader.
    Args:
        -dataset (dataset SpaRED class): class that has the adata.
        -split (str): desired split to obtain
    Returns:
        - st_data: spatial data
        - st_data_masked: masked spatial data
        - mask: mask used for calculations
    """
    ## Define the adata split
    adata = dataset[dataset.obs["split"]==split]
    
    # Define data
    st_data = adata.layers[pred_layer]
    # Define masked data
    st_data_masked = adata.layers["masked_expression_matrix"]
    # Define mask
    mask = adata.layers["random_mask"]
    mask = (1-mask)
    # En la mascara los valores masqueados son 0 y los valores reales deben ser 1
    
    # Normalize data
    max_data = st_data.max()
    st_data = st_data/max_data
    st_data_masked = st_data_masked/max_data

    #st used just for train
    return st_data, st_data_masked, mask, max_data
        
def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
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
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
        #sc.pp.neighbors(adata, n_neighbors=6, knn=True)
    # Get the adjacency matrix (binary matrix of shape spots x spots)
    adj_matrix = adata.obsp['spatial_connectivities']
    
    # Define power matrix
    power_matrix = adj_matrix.copy() #(spots x spots)
    # Define the output matrix
    output_matrix = adj_matrix.copy() #(spots x spots)

    # Iterate through the hops
    for i in range(n_hops-1):
        # Compute the next hop
        power_matrix = power_matrix * adj_matrix #Matrix Power Theorem: (i,j) is the he number of (directed or undirected) walks of length n from vertex i to vertex j.
        # Add the next hop to the output matrix
        output_matrix = output_matrix + power_matrix #Count the distance of the spots

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


def build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx):
    # Get nn indexes for the n_hop required
    nn_index_list = spatial_neighbors[idx] #Obtain the ids of the spots that are neigbors of idx
    #Index the expression matrix (X processed) and obtain the neccesary data
    #TODO: preguntarle a Daniela
    exp_matrix = expression_mtx[nn_index_list].type('torch.FloatTensor')
    return exp_matrix #shape (n_neigbors, n_genes)


def get_neigbors_dataset(adata, prediction_layer, num_hops):
    """
    This function recives the name of a dataset and pred_layer. Returns a list of len = number of spots, each position of the list is an array 
    (n_neigbors + 1, n_genes) that has the information about the neigbors of the corresponding spot.
    """
    all_neighbors_info = []
    #Dataset all info
    dataset = adata
    #get dataset splits
    splits = dataset.obs["split"].unique().tolist()
    #iterate over split adata
    for split in splits:
        split_neighbors_info = []
        adata = dataset[dataset.obs["split"]==split]
        # get slides for the correspoding split
        #slides = adata.obs["slide_id"].unique().tolist()
        #iterate over slides and get the neighbors info for every slide
        #Get dict with all the neigbors info for each spot in the dataset
        spatial_neighbors = get_spatial_neighbors(adata, n_hops=num_hops, hex_geometry=True)
        #Expression matrix (already applied post-processing)
        expression_mtx = torch.tensor(adata.layers[prediction_layer]) 
        for idx in tqdm(spatial_neighbors.keys()):
            # Obtainb expression matrix just wiht the neigbors of the corresponding spot
            neigbors_exp_matrix = build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx)
            split_neighbors_info.append(neigbors_exp_matrix)
        
        """
        for slide in slides:
            adata_slide = dataset[dataset.obs["slide_id"]==slide]
            #Get dict with all the neigbors info for each spot in the dataset
            spatial_neighbors = get_spatial_neighbors(adata_slide, n_hops=1, hex_geometry=True)
            #Expression matrix (already applied post-processing)
            expression_mtx = torch.tensor(dataset.layers[prediction_layer])
            #Empty list for saving data
            slide_neighbors_info = []
            #Iterate over all the spots
            for idx in tqdm(spatial_neighbors.keys()):
                # Obtainb expression matrix just wiht the neigbors of the corresponding spot
                neigbors_exp_matrix = build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx)
                slide_neighbors_info.append(neigbors_exp_matrix)

            #append slide neighbors info into corresponding split list
            split_neighbors_info.append(slide_neighbors_info)
            """
        #append split neighbors info into the complete list
        all_neighbors_info.append(split_neighbors_info)

    return all_neighbors_info  
  
def define_split_nn_mat(list_nn, list_nn_masked, split):
    
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
    if split == "train":
        list_nn = list_nn[0]
        list_nn_masked = list_nn_masked[0]
    elif split == "val":
        list_nn = list_nn[1]
        list_nn_masked = list_nn_masked[1]
    elif split == "test":
        list_nn = list_nn[2]
        list_nn_masked = list_nn_masked[2]
    
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
    #num_genes = int(mask.shape[1]/7)
    #num_genes = mask[0].shape[0]
    #mask[:, num_genes:] = 1
    
    #Convertir a numpy array
    st_data = st_data.numpy()
    st_data_masked = st_data_masked.numpy()
    mask = mask.numpy()
    
    # Normalización
    #max_data = st_data.max()
    #st_data = st_data/max_data
    #st_data_masked = st_data_masked/max_data
    max_data = st_data.max()
    min_data = st_data.min()
    st_data = normalize_to_minus_one_to_one(st_data, max_data, min_data)
    st_data_masked = normalize_to_minus_one_to_one(st_data_masked, max_data, min_data)*mask
    
    return st_data, st_data_masked, mask, max_data, min_data

def mask_extreme_prediction(list_nn):
    list_nn_masked = copy.deepcopy(list_nn)
    for i in range(len(list_nn_masked)):
        for j in range(len(list_nn_masked[i])):
            list_nn_masked[i][j][0][:] = 0
    return list_nn_masked
    
def get_mask_extreme_completion(adata, mask):
    mask_extreme_completion = copy.deepcopy(mask)
    imp_values = adata.layers["mask"] #True en los valores reales y False en los valores imputados
    mask_extreme_completion[imp_values] = 1
    mask_extreme_completion[:,:,1:] = 0
    return mask_extreme_completion
    
    
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def GaussianLogLikelihood(x0, mus, vars): 
    eps = 1e-09 
    coef = 1/(((2*torch.pi*vars)**0.5)) 
    likelihood = coef * torch.exp(-0.5*(((x0 - mus)/(vars**0.5))**2)) + eps 
    nll = -1 * torch.log(likelihood) 
    return nll 