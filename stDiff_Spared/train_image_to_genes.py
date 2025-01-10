import os
import warnings
import torch
import scanpy as sc


from model_stDiff.stDiff_model_2D import DiT_stDiff
from model_stDiff.stDiff_train import normal_train_stDiff
from process_stDiff.data_2D import *


from utils import *

from visualize_imputation import *
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

if args.vlo == False:
    #TODO: arreglar el tema del env para que esto sirva
    # Esto molestaba la instalacion con lo de UNI, solo lo quito mientras leo los adata, problema futuro
    #from spared.datasets import get_dataset
    pass

def main():
    ### Wandb 
    wandb.login()
    if args.debbug_wandb:
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wandb.init(project="debbugs", entity="spared_v2", name=exp_name + '_debbug')

    else:
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wandb.init(project="stDiff_Modelo_2D", entity="spared_v2", name=exp_name )
    #wandb.init(project="Diffusion_Models_NN", entity="sepal_v2", name=exp_name)
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
               "scheduler": args.scheduler,
               "layer": args.prediction_layer,
               "normalizacion": args.normalization_type,
               "batch_size": args.batch_size,
               "num_hops": args.num_hops,
               'scheduler_fixed': True,
               "diffusion_steps_train": args.diffusion_steps_train, 
               "diffusion_steps_test": args.diffusion_steps_test, 
               'noise_scheduler': args.noise_scheduler})
    
    ### Parameters
    # Define the training parameters
    lr = args.lr
    depth = args.depth
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    head = args.head
    device = torch.device('cuda')

    
    # Get dataset
    if args.vlo:
        # Carga el archivo .h5ad
        adata = sc.read_h5ad(os.path.join('Example_dataset', 'adata.h5ad'))
    else:
        adata = sc.read_h5ad(os.path.join('datasets', args.dataset, 'adata.h5ad'))

    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer

    breakpoint()
    # Get neighbors
    if args.neighbors_info:
        neighbors = 7
        dict_nn = get_neigbors_dataset(adata, pred_layer, args.num_hops)

    









if __name__=='__main__':
    main()