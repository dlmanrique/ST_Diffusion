import os
import warnings
import torch
import scanpy as sc

from Encoders_helper import ImageEncoder, GeneAutoencoder
from model_stDiff.stDiff_model_2D import DiT_stDiff
from model_stDiff.stDiff_train import normal_train_stDiff
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
    
    breakpoint()
    
    model.to(device)
    #TODO: change this model name
    save_path_prefix = args.dataset + "_" + str(args.depth) + "_" + str(args.hidden_size) + "_" + str(args.lr) + "_" + args.loss_type + ".pt"

    ### Train the model
    model.train()
    if not os.path.isfile(save_path_prefix):
        normal_train_stDiff(model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=val_dataloader,
                            max_norm = [max_train, max_valid],
                            min_norm = [min_train, min_valid],
                            avg_tensor = avg_tensor,
                            wandb_logger=wandb,
                            args=args,
                            st_data_train=norm_st_data_train,
                            st_data_val=norm_st_data_valid,
                            adata_valid = val_adata,
                            lr=lr,
                            num_epoch=num_epoch,
                            device=device,
                            save_path=save_path_prefix,
                            exp_name=exp_name)
    else:
        model.load_state_dict(torch.load(save_path_prefix))

    if "test" in splits:
        model.load_state_dict(torch.load(os.path.join("Experiments", exp_name, save_path_prefix)))

        model.eval()
        
        metrics_dict_train, imputation_data_train = inference_function(dataloader=train_dataloader,
                                    gt_data=norm_st_data_train, 
                                    model=model,
                                    max_norm = max_train,
                                    min_norm = min_train,
                                    avg_tensor = avg_tensor,
                                    diffusion_step=args.diffusion_steps_test,
                                    device=device,
                                    args=args
                                    )
        
        metrics_dict_val, imputation_data_val = inference_function(dataloader=val_dataloader,
                                        gt_data=norm_st_data_valid, 
                                        model=model,
                                        max_norm = max_valid,
                                        min_norm = min_valid,
                                        avg_tensor = avg_tensor,
                                        diffusion_step=args.diffusion_steps_test,
                                        device=device,
                                        args=args
                                        )

        #log_pred_image_extreme_completion(adata_valid, args, epoch)
        
    
        test_metrics, imputation_data = inference_function(dataloader=test_dataloader,
                                        gt_data= norm_st_data_test, 
                                        model=model,
                                        max_norm = max_test,
                                        min_norm = min_test,
                                        avg_tensor = avg_tensor,
                                        diffusion_step=args.diffusion_steps_test,
                                        device=device,
                                        args=args
                                        )
        
        torch.save(model.state_dict(), os.path.join("Experiments", exp_name, 'post_test.pth'))
        
        adata_test = adata[adata.obs["split"]=="test"]
        adata_test.layers["diff_pred"] = imputation_data
        #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
        #log_pred_image_extreme_completion(adata_test, args, -1)
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        wandb.log({"test_MSE_train": metrics_dict_train["MSE"], "test_PCC_train": metrics_dict_train["PCC-Gene"]})
        wandb.log({"test_MSE_val": metrics_dict_val["MSE"], "test_PCC_val": metrics_dict_val["PCC-Gene"]})
        wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})



if __name__=='__main__':

    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = get_main_parser()
    args = parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    main()