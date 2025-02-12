import os
import warnings
import torch
import scanpy as sc
import json

from Encoders_helper import ImageEncoder, GeneAutoencoder
from model_stDiff.stDiff_model_2D import DiT_stDiff
from model_stDiff.stDiff_train import train_stDiff
from utils import *
from data import SpaREDData
from visualizations import visualize_predictions

import wandb
from datetime import datetime

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
    ### Wandb 
    if args.debbug_wandb:
        wandb.init(project='debbugs_v2', entity = 'spared_v2', config=vars(args), name=exp_name + '_debbug')

    else:
        wandb.init(project='Image_to_Genes', entity = 'spared_v2', config=vars(args), name=exp_name)
    
    #Save path
    save_path = os.path.join("Experiments", args.dataset, exp_name)
    os.makedirs(save_path, exist_ok=True)

    #Open file with dataset name and genes info
    with open("dataset_genes_info.json", "r") as file:
        dataset2genes = json.load(file)


    # Load Image encoder (patch encoder)
    # Load the class 
    image_encoder = ImageEncoder(args.image_encoder, args.dataset, dataset2genes[args.dataset])
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
        gene_autoencoder = GeneAutoencoder(args.gene_autoencoder, args.dataset, args.pred_layer)
        gene_autoencoder_model = gene_autoencoder.get_gene_autoencoder(configs = configs)
        wandb.config.update({"autoencoder_ckpts_path": gene_autoencoder.autoencoder_path}, allow_val_change=True)
    
    # Prepare data
    spared_data = SpaREDData(args, gene_autoencoder_model, image_encoder_model, transforms)

    # Register the name of the layer that will be used for computing the final ST metrics with the get_metrics function after sampling
    if "deltas" in args.pred_layer:
        wandb.config.layer_for_test = spared_data.layer_for_test
    else:
        wandb.config.layer_for_test = args.pred_layer

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


    if args.train:
        best_model_path = train_stDiff(
        model,
        data=spared_data,
        wandb_logger=wandb,
        args=args,
        gene_autoencoder=gene_autoencoder_model,
        save_path=save_path,
        device=device,
        exp_name=exp_name
        )
        
    else: 
        best_model_path = args.dit_ckpts_path
    
    # Load the best model after training
    model.load_state_dict(torch.load(best_model_path))

    if args.test:  
        #Inference in train and val to control overfitting
        train_dict, _ = inference_function(
                                data=spared_data,
                                model=model,
                                diffusion_steps=args.sample_diffusion_steps,
                                device=device,
                                args=args,
                                model_autoencoder=gene_autoencoder_model,
                                wandb_logger=wandb,
                                process="train"
                                )
        
        valid_dict, _ = inference_function(
                                data=spared_data,
                                model=model,
                                diffusion_steps=args.sample_diffusion_steps,
                                device=device,
                                args=args,
                                model_autoencoder=gene_autoencoder_model,
                                wandb_logger=wandb,
                                process="val"
                                )
        
        test_dict, _ = inference_function(
                                data=spared_data,
                                model=model,
                                diffusion_steps=args.sample_diffusion_steps,
                                device=device,
                                args=args,
                                model_autoencoder=gene_autoencoder_model,
                                wandb_logger=wandb,
                                process="test"
                                )

        # Log the test results in wandb
        wandb.log({"Test_MSE_train": train_dict["MSE"], "Test_PCC_train": train_dict["PCC-Gene"]})
        wandb.log({"Test_MSE_valid": valid_dict["MSE"], "Test_PCC_valid": valid_dict["PCC-Gene"]})
        wandb.log({"Test_MSE_test": test_dict["MSE"], "Test_PCC_test": test_dict["PCC-Gene"]})

    if args.visualizations:
        
        # Get the predicted data
        all_dict, pred_data = inference_function(
                                data=spared_data,
                                model=model,
                                diffusion_steps=args.sample_diffusion_steps,
                                device=device,
                                args=args,
                                model_autoencoder=gene_autoencoder_model,
                                wandb_logger=wandb,
                                process="all"
                                )

        #Visualization plots for predicted data
        pred_data_128 = pred_data[:,spared_data.gene_weights]
        visualize_predictions(args.dataset, spared_data.original_full_adata, pred_data_128, exp_name)

if __name__=='__main__':

    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = get_main_parser()
    args = parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    main()