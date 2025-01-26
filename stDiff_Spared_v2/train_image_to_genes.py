import os
import warnings
import torch
import scanpy as sc

from Encoders_helper import ImageEncoder, GeneAutoencoder
from model_stDiff.stDiff_model_2D import DiT_stDiff
from model_stDiff.stDiff_train import train_stDiff
from utils import *
from data import SpaREDData

import wandb
from datetime import datetime

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
    ### Wandb 
    if args.debbug_wandb:
        wandb.init(project='debbugs_v2', entity = 'spared_v2', config=vars(args), name=exp_name + '_debbug')

    else:
        wandb.init(project='Image_to_Genes', entity = 'spared_v2', config=vars(args), name=exp_name + '_debbug')
    
    #Save path
    save_path = os.path.join("Experiments", args.dataset, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # Load Image encoder (patch encoder)
    # Load the class 
    image_encoder = ImageEncoder(args.image_encoder)
    # Get the model weights of the patch encoder
    image_encoder_model, transforms = image_encoder.get_patch_encoder_model()

    # Load gene autoencoder
    #TODO: replace this using args or something else in order to experiment with different gene_autoencoder
    configs = {'input_dim': 1024,
               'latent_dim': 128,
               'embedding_dim': 256,
               'num_layers': 2,
               'num_heads':2}
    
    gene_autoencoder_model = None
    if args.gene_autoencoder:
        gene_autoencoder = GeneAutoencoder(args.gene_autoencoder, args.autoencoder_path)
        gene_autoencoder_model = gene_autoencoder.get_gene_autoencoder(configs = configs)

    
    spared_data = SpaREDData(args, gene_autoencoder_model, image_encoder_model, transforms)


    ### DIFFUSION MODEL ##########################################################################
    # Here I have the input dim for the DiT model, 128 or 7,128
    input_size_dit = spared_data.train_data.DiT_input_dim
    image_features_dim = spared_data.train_data.image_features_dim

    # Define the model
    model = DiT_stDiff(
        input_size=input_size_dit,  
        hidden_size=args.dit_hidden_size, 
        depth=args.dit_depth,
        num_heads=args.num_heads,
        images_features_dim=image_features_dim,
        classes=6, 
        args=args,
        mlp_ratio=4.0,
        dit_type='dit')
        
    model.to(device)
    #TODO: change this model name
    #save_path_prefix = args.dataset + "_" + str(args.depth) + "_" + str(args.hidden_size) + "_" + str(args.lr) + "_" + args.loss_type + ".pt"


    if args.train:
        best_model_path = train_stDiff(
        model,
        data=spared_data,
        wandb_logger=wandb,
        args=args,
        gene_autoencoder=gene_autoencoder_model,
        image_encoder=image_encoder_model,
        save_path=save_path,
        device=device,
        exp_name=exp_name
        )
        
    else: 
        best_model_path = args.dit_ckpts_path
        
    # Load the best model after training
    model.load_state_dict(torch.load(best_model_path))

    if args.test:
        test_dict, test_imputation_data = inference_function(
        data=spared_data,
        model=model,
        diffusion_steps=args.sample_diffusion_steps,
        device=device,
        args=args,
        model_autoencoder=autoencoder,
        process="test"
        )


if __name__=='__main__':

    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = get_main_parser()
    args = parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    main()