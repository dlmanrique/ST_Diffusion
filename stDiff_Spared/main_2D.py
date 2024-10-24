import numpy as np
import pandas as pd
import os
import warnings
import torch

from model_stDiff.stDiff_model_2D import DiT_stDiff
from model_stDiff.stDiff_model_2D_Conv2D import DiT_stDiff_2D
from model_stDiff.stDiff_model_BoDiffusion import DiT_stDiff_BoDiffusion
from model_stDiff.stDiff_train import normal_train_stDiff
from process_stDiff.data_2D import *
import anndata as ad
from spared.datasets import get_dataset

from utils import *

from spared_imputation.model import GeneImputationModel
from spared_imputation.predictions import get_predictions
from spared_imputation.utils import apply_median_imputation_method
from spared_imputation.utils import prepare_preds_for_visuals

from visualize_imputation import log_pred_image
import wandb
from datetime import datetime

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args) #Not uses, maybe later usage

# seed everything
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    breakpoint()
    ### Wandb 
    wandb.login()
    if args.debbug_wandb:
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wandb.init(project="debbugs", entity="spared_v2", name=exp_name + '_debbug')

    else:
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wandb.init(project="stDiff_Modelo_2D", entity="spared_v2", name=exp_name )
        
    wandb.config = {"lr": args.lr, "dataset": args.dataset}
    wandb.log({"lr": args.lr, 
               "dataset": args.dataset, 
               "num_epoch": args.num_epoch, 
               "num_heads": args.head,
               "depth": args.depth, "hidden_size": args.hidden_size, 
               "save_path": args.save_path, "loss_type": args.loss_type,
               "concat_dim": args.concat_dim,
               "masked_loss": args.masked_loss,
               "model_type": args.model_type,
               "scheduler": args.scheduler})
    
    ### Parameters
    # Define the training parameters
    lr = args.lr
    depth = args.depth
    num_epoch = args.num_epoch
    diffusion_step = args.diffusion_steps
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    head = args.head
    device = torch.device('cuda')

    # Get dataset
    dataset = get_dataset(args.dataset)
    adata = dataset.adata
    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer
    # Masking
    prob_tensor = get_mask_prob_tensor(masking_method="mask_prob", dataset=dataset, mask_prob=0.3, scale_factor=0.8)
    # Add neccesary masking layers in the adata object
    mask_exp_matrix(adata=adata, pred_layer=pred_layer, mask_prob_tensor=prob_tensor, device=device)

    # Get neighbors
    neighbors = 7
    list_nn = get_neigbors_dataset(adata, pred_layer, args.num_hops)
    list_nn_masked = get_neigbors_dataset(adata, 'masked_expression_matrix', args.num_hops)
    ### Define splits
    ## Train
    st_data_train, st_data_masked_train, mask_train, max_train, min_train = define_split_nn_mat(list_nn, list_nn_masked, "train")
    ## Validation
    st_data_valid, st_data_masked_valid, mask_valid, max_valid, min_valid = define_split_nn_mat(list_nn, list_nn_masked, "val")

    ## Test
    if "test" in splits:
        st_data_test, st_data_masked_test, mask_test, max_test, min_test = define_split_nn_mat(list_nn, list_nn_masked, "test")

    # Definir un tensor de promedio en caso de predecir una capa delta
    num_genes = adata.shape[1]
    if "deltas" in pred_layer:
        avg_tensor = torch.tensor(adata.var["c_d_log1p_avg_exp"]).view(1, num_genes)
    else:
        avg_tensor = None
    
    # Define train and valid dataloaders
    train_dataloader = get_data_loader(
        st_data_train, 
        st_data_masked_train, 
        mask_train,
        batch_size=batch_size, 
        is_shuffle=True)

    valid_dataloader = get_data_loader(
        st_data_valid, 
        st_data_masked_valid,
        mask_valid, 
        batch_size=batch_size, 
        is_shuffle=False)

    # Define test dataloader if it exists
    if 'test' in splits:
        test_dataloader = get_data_loader(
        st_data_test, 
        st_data_masked_test,
        mask_test, 
        batch_size=batch_size, 
        is_shuffle=False)

    ### DIFFUSION MODEL ##########################################################################
    num_nn = st_data_train[0].shape

    # Define the model
    """
    if num_nn > 1024:
        hidden_size = 1024
    if num_nn > 2048:
        hidden_size = 2048
    """
    
    if args.model_type == "1D":
        model = DiT_stDiff(
            input_size=num_nn,  
            hidden_size=hidden_size, 
            depth=depth,
            num_heads=head,
            classes=6, 
            args=args,
            mlp_ratio=4.0,
            dit_type='dit')
    elif args.model_type == "2D":
        model = DiT_stDiff_2D(
            input_size=num_nn,  
            hidden_size=hidden_size, 
            depth=depth,
            num_heads=head,
            classes=6, 
            args=args,
            mlp_ratio=4.0,
            dit_type='dit')
    elif args.model_type == "BoDiffusion":
        model = DiT_stDiff_BoDiffusion(
            input_size=num_nn,  
            hidden_size=hidden_size, 
            depth=depth,
            num_heads=head,
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
                                valid_dataloader=valid_dataloader,
                                valid_data = st_data_valid,
                                valid_masked_data = st_data_masked_valid,
                                mask_valid = mask_valid,
                                max_norm = [max_train, max_valid],
                                min_norm = [min_train, min_valid],
                                avg_tensor = avg_tensor,
                                wandb_logger=wandb,
                                args=args,
                                lr=lr,
                                num_epoch=num_epoch,
                                diffusion_step=diffusion_step,
                                device=device,
                                save_path=save_path_prefix,
                                exp_name=exp_name)
    else:
        model.load_state_dict(torch.load(save_path_prefix))

    if "test" in splits:
        model.load_state_dict(torch.load(os.path.join("Experiments", exp_name, save_path_prefix)))
        test_metrics, imputation_data = inference_function(dataloader=test_dataloader, 
                                    data=st_data_test, 
                                    masked_data=st_data_masked_test, 
                                    mask=mask_test,
                                    max_norm = max_test,
                                    min_norm = min_test,
                                    avg_tensor = avg_tensor,
                                    model=model,
                                    diffusion_step=diffusion_step,
                                    device=device,
                                    args=args)

        adata_test = adata[adata.obs["split"]=="test"]
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})
        #print(test_metrics)
    """
    else: 
        model.load_state_dict(torch.load(save_path_prefix))
        valid_metrics, imputation_data = inference_function(dataloader=valid_dataloader, 
                                    data=st_data_valid, 
                                    masked_data=st_data_masked_valid, 
                                    mask=mask_valid,
                                    max_norm = max_valid,
                                    model=model,
                                    diffusion_step=diffusion_step,
                                    device=device)

        #adata_test = adata[adata.obs["split"]=="test"]
        # FIXME: (siguiente linea comentada por ahora) 
        # ValueError: Value passed for key 'diff_pred' is of incorrect shape. Values of layers must match dimensions (0, 1) of parent. Value had shape (439, 128) while it should have had (533, 128).
        #adata_test.layers["diff_pred"] = imputation_data.T
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        wandb.log({"valid_MSE": valid_metrics["MSE"], "valid_PCC": valid_metrics["PCC-Gene"]})
        #print(test_metrics)
    """ 
     
if __name__=='__main__':
    main()

# VISUALIZATION
"""
log_pred_image(adata=adata_test,
               args=args,  
               slide = "",
               gene_id=args.gene_id)
"""
