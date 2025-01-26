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
    image_encoder_model = image_encoder.get_patch_encoder_model()

    # Load gene autoencoder
    #TODO: replace this using args or something else in order to experiment with different gene_autoencoder
    configs = {'input_dim': 1024,
               'latent_dim': 128,
               'embedding_dim': 256,
               'num_layers': 2,
               'num_heads':2}
    
    gene_autoencoder = GeneAutoencoder(args.gene_autoencoder)
    gene_autoencoder_model = gene_autoencoder.get_gene_autoencoder(configs = configs)

    spared_data = SpaREDData(args, gene_autoencoder_model, image_encoder_model)



    
    # Get dataset
    if args.vlo:
        # Carga el archivo .h5ad
        adata = sc.read_h5ad(os.path.join('Example_dataset', 'adata.h5ad'))
    else:
        adata = sc.read_h5ad(os.path.join('datasets', 'original', args.dataset, 'adata.h5ad'))

    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer

    
    # Definir un tensor de promedio en caso de predecir una capa delta
    num_genes = adata.shape[1]
    if "deltas" in pred_layer:
        format = args.prediction_layer.split("deltas")[0]
        avg_tensor = torch.tensor(adata.var[f"{format}log1p_avg_exp"]).view(1, num_genes)
    else:
        avg_tensor = None
    
    # Split the data into train, val and test
    # Load patch features data
    train_adata = adata[adata.obs["split"]=="train"]
    st_data_train = torch.tensor(train_adata.layers[pred_layer])
    features_train = torch.load(os.path.join('UNI', args.dataset, 'train.pt'))

    val_adata = adata[adata.obs["split"]=="val"]
    st_data_val = torch.tensor(val_adata.layers[pred_layer])
    features_val = torch.load(os.path.join('UNI', args.dataset, 'val.pt'))
    
    if len(splits) == 3:
        test_adata = adata[adata.obs["split"]=="test"]
        st_data_test = torch.tensor(test_adata.layers[pred_layer])
        features_test = torch.load(os.path.join('UNI', args.dataset, 'test.pt'))


    # Get dataloaders
    # Define train and valid dataloaders
    train_dataloader, norm_st_data_train, max_train, min_train = get_data_loader_image_to_gene(
        st_data_train, # Datos de expresion de la layer que es
        features_train, # Features de los parches asociados
        batch_size=batch_size, 
        is_shuffle=True)
    
    val_dataloader, norm_st_data_valid, max_valid, min_valid = get_data_loader_image_to_gene(
        st_data_val, # Datos de expresion de la layer que es
        features_val, # Features de los parches asociados
        batch_size=batch_size, 
        is_shuffle=False)

    if len(splits) == 3:
        test_dataloader, norm_st_data_test, max_test, min_test = get_data_loader_image_to_gene(
        st_data_test, # Datos de expresion de la layer que es
        features_test, # Features de los parches asociados
        batch_size=batch_size, 
        is_shuffle=False)

    ### DIFFUSION MODEL ##########################################################################
    #FIXME: how to replace this num_nn calculation
    num_nn = st_data_train[0].shape

    # Define the model
    model = DiT_stDiff(
        input_size=num_nn,  
        hidden_size=args.hidden_size, 
        depth=args.depth,
        num_heads=args.head,
        classes=6, 
        args=args,
        mlp_ratio=4.0,
        dit_type='dit')
    
    model.to(device)
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