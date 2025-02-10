from metrics import get_metrics
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                        type=str,               default='villacampa_lung_organoid',      help='Dataset to use.')
    parser.add_argument('--pred_layer',                     type=str,               default='c_t_deltas',                    help='SpaRED prediction layer to use.')
    parser.add_argument('--num_neighs',                     type=int,               default=6,                               help='Amount of neighbors considered to build spot neighborhoods. Must be the same as the ones used to train the autoencoder. Use -1 to avoid neighbors info')
    parser.add_argument('--normalize_img_fts',              type=str2bool,          default=False,                           help='Whether or not to normalize the image features to a reange [-1, 1].')
    # Model parameters #######################################################################################################################################################################
    parser.add_argument('--dit_hidden_size',                type=int,               default=1024,                            help='')
    parser.add_argument('--dit_depth',                      type=int,               default=12,                              help='')
    parser.add_argument('--num_heads',                      type=int,               default=16,                              help='')
    parser.add_argument("--concat_dim",                     type=int,               default=0,                               help='Which dimension used to concat the condition.')
    parser.add_argument('--dit_ckpts_path',                 type=str,               default='',                              help='Path to trained checkpoints of DiT corresponding to the dataset used. Optional.')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--train',                          type=str2bool,          default=True,                            help='Train model.')
    parser.add_argument('--test',                           type=str2bool,          default=True,                            help='Test model.')
    parser.add_argument('--visualizations',                 type=str2bool,          default=False,                           help='Whether or not to visualize the results.')
    parser.add_argument('--lr',                             type=float,             default=0.0001,                          help='lr to train DiT.')
    parser.add_argument('--batch_size',                     type=int,               default=256,                             help='Batch size used to train the diffusion model.')
    parser.add_argument('--num_epochs',                     type=int,               default=3000,                            help='Number of training epochs.')
    parser.add_argument('--train_diffusion_steps',          type=int,               default=1500,                            help='Number of diffusion steps for training process.')
    parser.add_argument('--sample_diffusion_steps',         type=int,               default=50,                            help='Number of diffusion steps for val or test process.')
    parser.add_argument('--step_size',                      type=float,             default=600,                             help='Step size to use in learning rate scheduler')
    parser.add_argument("--adjust_loss",                    type=str2bool,          default=True,                            help='If True the loss is obtained only on masked data. If False the loss takes into account the entire set of genes and spots.')
    parser.add_argument("--scheduler",                      type=str2bool,          default=True,                            help='Whether to use LR scheduler or not.')
    # Image encoder and Gene Autoencoder parameters ##########################################################################################################################################
    parser.add_argument('--image_encoder',                  type=str,               default='uni',                           help='Name of the image encoder to use. (uni, shufflenet)')
    parser.add_argument('--gene_autoencoder',               type=str,               default=None,                            help='Name of the gene autoencoder to use. Only one for now: Transformer_encoder_mlp_decoder_v1')
    ##########################################################################################################################################################################################
    parser.add_argument('--debbug_wandb',                   type=str2bool,          default=False,                           help='Log in debbugs wandb')
    return parser


def data_normalization(data: torch.tensor, data_min, data_max):
    """ 
    This function receives a gene expression matrix and normalizes its content so that it has a range of [-1, 1].
    """
    norm_data = 2 * (data - data_min) / (data_max - data_min) - 1
    
    return norm_data

'''def data_denormalization(norm_data: torch.tensor, data_min: torch.tensor, data_max: torch.tensor):
    """
    This function receives a normalized gene expression matrix and denormalizes its content 
    back to the original range using the provided data_min and data_max.
    """
    denorm_data = (norm_data + 1) / 2 * (data_max - data_min) + data_min
    return denorm_data'''

def denormalize_from_minus_one_to_one(X_norm, X_min, X_max):
    # Apply the denormalization formula 
    X_denorm = ((X_norm + 1) / 2) * (X_max - X_min) + X_min
    return X_denorm


def decode(imputation, model_decoder):
    imputation = torch.tensor(imputation.permute(0,2,1), dtype=torch.float32)[:,:,0] # shape torch.Size([439, 128])
    dataset = TensorDataset(imputation)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    model_decoder.to("cuda")
    model_decoder.eval()  
    decoded_samples = []
    with torch.no_grad():
        for batch in dataloader: 
            batch = batch[0].to("cuda") 
            decoded_batch = model_decoder.decoder(batch) 
            decoded_samples.append(decoded_batch)

    decoded_samples = torch.cat(decoded_samples, dim=0)

    return decoded_samples


def inference_function(data, model, diffusion_steps, device, args, model_autoencoder, wandb_logger, process = "val"):
    # To avoid circular imports
    from model_stDiff.stDiff_scheduler import NoiseScheduler
    from model_stDiff.sample import sample_stDiff
    """
    Function designed to do inference for validation and test steps.
    Params:
        - data (SpaREDData): class with all SpaRED data preprocessed
        - model (diffusion model): diffusion model to do inference
        - diffusion_steps (int): number of steps needed for denoising during sampling (set in argparse)
        - device (str): device cpu or cuda
        - args (argparse): parser with the values necessary for custom training and test
        - model_autoencoder (autoencoder): autoencoder with a "decoder()" attribute
        - process (str): either "val" or "test" to determine the data split that needs to be used

    Returns:
        - metrics_dict (dict): dictionary with all the evaluation metrics
    """
    # Define noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_steps,
        beta_schedule='cosine'
    )
    
    # Remember that if pred_layer is based on delta values, the layer ised for testing will change to c_XX_log1p (XX denotes the completion method also used in pred_layer).
    if process == "train":
        dataloader = data.train_dataloader()
        min_norm, max_norm = data.train_data.min_val, data.train_data.max_val
        log1p_data = torch.tensor(data.spared_train.layers[data.layer_for_test])
        xt_shape = data.train_data.all_st_data_shape
    elif process == "val":
        dataloader = data.val_dataloader()
        min_norm, max_norm = data.val_data.min_val, data.val_data.max_val
        log1p_data = torch.tensor(data.spared_val.layers[data.layer_for_test])
        xt_shape = data.val_data.all_st_data_shape
    elif process == "test":
        dataloader = data.test_dataloader()
        min_norm, max_norm = data.test_data.min_val, data.test_data.max_val
        log1p_data = torch.tensor(data.spared_test.layers[data.layer_for_test])
        xt_shape = data.test_data.all_st_data_shape
    else: # predict on all data
        dataloader = data.all_dataloader()
        min_norm, max_norm = data.all_data.min_val, data.all_data.max_val
        log1p_data = torch.tensor(data.spared_all.layers[data.layer_for_test])
        xt_shape = data.all_data.all_st_data_shape

    # inference using test split
    imputation = sample_stDiff(model,
                        dataloader=dataloader,
                        noise_scheduler=noise_scheduler,
                        x_t_shape=xt_shape,
                        args=args,
                        device=device,
                        num_step=diffusion_steps) # output shape total_samples x 128

    imputation = denormalize_from_minus_one_to_one(imputation, min_norm, max_norm)
        
    if args.gene_autoencoder:
        if args.num_neighs == -1:
            # Spot type
            encoded_st_data_key = 'encoded_spot_exp'
        else:
            # Neighbors matrix type
            encoded_st_data_key = 'encoded_exp_matrix'

       # Get all ground truths
        ground_truth = []
        # Iterate entire dataloader to get complete tensor of masks and x_conds
        test_mask = []

        for batch in dataloader:
            try:
                ground_truth.append(batch[encoded_st_data_key].permute(0,2,1))
                test_mask.append(batch["exp_mask"].permute(0,2,1))
            except:
                ground_truth.append(batch[encoded_st_data_key])
                test_mask.append(batch["exp_mask"].squeeze())

        ground_truth = torch.cat(ground_truth, dim=0)
        ground_truth = denormalize_from_minus_one_to_one(ground_truth, min_norm, max_norm)

        test_mask = torch.cat(test_mask, dim=0)

        # Check how much do minor perturbations in the model's prediction affect the output of the decoder
        imputation = torch.tensor(imputation) 
        perturbation = torch.randn_like(imputation) * 0.01
        
        if len(ground_truth.shape) < 3:
            #Encoded spots model
            decoded_imputation = decode(imputation=imputation.unsqueeze(dim=1), model_decoder=model_autoencoder)
            decoded_perturbation = decode(imputation=perturbation.unsqueeze(dim=1), model_decoder=model_autoencoder)
            dit_imputation = torch.tensor(imputation.unsqueeze(dim=1), dtype=torch.float32)[:,0,:]
            dit_gt = torch.tensor(ground_truth.unsqueeze(dim=1), dtype=torch.float32)[:,0,:]
        else:
            decoded_imputation = decode(imputation=imputation, model_decoder=model_autoencoder)
            decoded_perturbation = decode(imputation=perturbation, model_decoder=model_autoencoder)
            dit_imputation = torch.tensor(imputation, dtype=torch.float32)[:,0,:] # imputation shape is BSx7x128, thus we take only index 0 in the second dimension. 
            dit_gt = torch.tensor(ground_truth, dtype=torch.float32)[:,:,0]

        mse_pre = F.mse_loss(imputation, perturbation)
        print("MSE pred vs perturbed-pred before decoding: ", mse_pre)
        mse_post = F.mse_loss(decoded_imputation, decoded_perturbation)
        print("MSE pred vs perturbed-pred after decoding: ", mse_post)
        
        dit_mse = F.mse_loss(dit_gt.to('cuda'), dit_imputation.to('cuda'))
        print("mse del dit (encoded pred vs encoded gt): ", dit_mse.item()) # MSE between gt and prediction before decoding.
        wandb_logger.log({"encoded_pred_MSE": dit_mse})

    decoded_imputation = decoded_imputation.detach().cpu() # Sigo en c_t_deltas
    
    if len(decoded_imputation.shape) == 3:
        #Evaluate only on main spot
        decoded_imputation = decoded_imputation[:,0,:]
    if len(test_mask.shape) == 3:
        test_mask = test_mask[:,:,0]

    if "deltas" in args.pred_layer:
        imputation_tensor = decoded_imputation + data.average_vals.cpu()
        imputation_tensor = np.array(imputation_tensor) 

    imputation_tensor = torch.tensor(imputation_tensor, dtype=torch.float32)

    metrics_dict = get_metrics(log1p_data, imputation_tensor.to(device), test_mask.to(device)) 
    
    return metrics_dict, imputation_tensor
