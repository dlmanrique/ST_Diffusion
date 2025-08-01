import os
import warnings
import torch
import scanpy as sc
from model_stDiff.stDiff_model_2D import DiT_stDiff
from model_stDiff.stDiff_train import normal_train_stDiff
from process_stDiff.data_2D import *
from utils import *
from visualization_results.visualize_imputation import *
import wandb
from datetime import datetime
from Transformer_encoder_decoder import *
from Transformer_simple import Transformer
from scipy.sparse import csr_matrix
from get_prediction_layers import spackle_layer, get_prob_mse, get_mask_preds_layers
from visualization_results.visualize_mse_probs import plot_mse_plot

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
               "scheduler": args.scheduler,
               "layer": args.prediction_layer,
               "normalizacion": args.normalization_type})
    
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
    #dataset = get_dataset(args.dataset)
    #adata_128 = dataset.adata
    adata_128 = ad.read_h5ad(f'/media/SSD0/pcardenasg2/c_dif_layers/datasets/original/{args.dataset}.h5ad')
    adata = ad.read_h5ad(f'/media/SSD0/pcardenasg2/c_dif_layers/datasets/1024/{args.dataset}_1024.h5ad')

    ### AUTOENCODER ADATA ###
    num_genes = adata_128.shape[1]

    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer

    # create mask for 128 genes
    genes_evaluate = []
    genes_128 = adata_128.var["gene_ids"].unique().tolist()
    genes_1024 = adata.var["gene_ids"].unique().tolist()
    
    #Get updated 1024 adata
    adata, adata_128 = sort_adatas(adata=adata, adata_128=adata_128)

    genes_128 = adata_128.var["gene_ids"].unique().tolist()
    genes_1024 = adata.var["gene_ids"].unique().tolist()
    
    for gene in genes_1024:
        if gene in genes_128:
            genes_evaluate.append(1)
        else:
            genes_evaluate.append(0)
            
    gene_weights = torch.tensor(genes_evaluate, dtype=torch.float32)
    
    model_autoencoder = None
    
    if 'test' in splits:
        adata_test = [adata[adata.obs["split"]=="test"], adata_128[adata_128.obs["split"]=="test"]]
    else:
        adata_test = [adata[adata.obs["split"]=="val"], adata_128[adata_128.obs["split"]=="val"]]

    adata, adata_128 = get_mask_preds_layers(adata_test=adata_test, pred_layer=pred_layer, genes=gene_weights, mask_percentage=args.mask_percentage, args=args)
    
    #matrix input
    list_nn, max_min_enc = get_neigbors_dataset(adata, pred_layer, args.num_hops, model_autoencoder, args)
   
    #Transformer model
    num_layers = 2
    n_heads = 2
    embedding_dim =  256
    feedforward_dim = embedding_dim * 2
    
    model_autoencoder = Transformer(input_dim=1024, 
                                    latent_dim=128, 
                                    output_dim=1024,
                                    embedding_dim=embedding_dim,
                                    feedforward_dim=feedforward_dim,
                                    num_layers=num_layers,
                                    num_heads=n_heads,
                                    lr=args.lr,
                                    gene_weights=gene_weights)
    
    checkpoint_path = os.path.join("/media/SSD0/pcardenasg2/ST_Diffusion/stDiff_Spared_v2/transformer_autoencoders", f"{args.dataset}", "autoencoder_model.ckpt") 
    #checkpoint_path = os.path.join("/home/dvegaa/ST_Diffusion/stDiff_Spared/mask_autoencoder", f"{args.dataset}", "autoencoder_model.ckpt") 
    
    checkpoint = torch.load(checkpoint_path)
    model_autoencoder.load_state_dict(checkpoint['state_dict'])
    model_autoencoder.to(device)
    
    #matrix input
    list_nn = encode_transformers(list_nn=list_nn, model_autoencoder=model_autoencoder, batch_size=args.batch_size)
    list_nn_masked = mask_extreme_prediction(list_nn)
    #####TODO: revisar
    
    ### Define splits
    ## Test
    if "test" in splits:
        st_data_test, st_data_masked_test, mask_test, max_test, min_test = define_split_nn_mat(list_nn, list_nn_masked, "test", args)
        mask_extreme = np.zeros((mask_test.shape[0], mask_test.shape[1]*8, mask_test.shape[2]))
        mask_extreme_completion_test = get_mask_extreme_completion(adata[adata.obs["split"]=="test"], mask_extreme, genes_evaluate, args)
    else:
        ## Validation
        st_data_test, st_data_masked_test, mask_test, max_test, min_test = define_split_nn_mat(list_nn, list_nn_masked, "val", args)
        mask_extreme = np.zeros((mask_test.shape[0], mask_test.shape[1]*8, mask_test.shape[2]))
        mask_extreme_completion_test = get_mask_extreme_completion(adata[adata.obs["split"]=="val"], mask_extreme, genes_evaluate, args)
        
    # Definir un tensor de promedio en caso de predecir una capa delta
    num_deltas = adata.shape[1]
    if "deltas" in pred_layer:
        format = args.prediction_layer.split("deltas")[0]
        avg_tensor = torch.tensor(adata.var[f"{format}log1p_avg_exp"]).view(1, num_deltas)
    else:
        avg_tensor = None

    test_dataloader = get_data_loader(
        st_data_test, 
        st_data_masked_test,
        mask_test, 
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
    num_nn = st_data_test[0].shape

    # Define the model
    model = DiT_stDiff(
        input_size=num_nn,  
        hidden_size=hidden_size, 
        depth=depth,
        num_heads=head,
        classes=6, 
        args=args,
        mlp_ratio=4.0,
        dit_type='dit')

    dit_path = os.path.join("/media/SSD0/pcardenasg2/ST_Diffusion/completion_task", f"{args.dataset}", f"{args.dataset}_12_1024_0.0001_noise.pt")
    dit_state_dict = torch.load(dit_path)
    model.load_state_dict(dit_state_dict)
    model.to(device)
    model.eval()
    
    if 'test' in splits:
        adata_test = [adata[adata.obs["split"]=="test"], adata_128[adata_128.obs["split"]=="test"]]
        max_enc = max_min_enc["test"][0]
        min_enc = max_min_enc["test"][1]
    else:
        adata_test = [adata[adata.obs["split"]=="val"], adata_128[adata_128.obs["split"]=="val"]]
        max_enc = max_min_enc["val"][0]
        min_enc = max_min_enc["val"][1]
        
    test_metrics, imputation_data, mse = inference_function(adata=adata_test,
                                                        dataloader=test_dataloader, 
                                                        data=st_data_test, 
                                                        masked_data=st_data_masked_test, 
                                                        mask=mask_test,
                                                        mask_extreme_completion=mask_extreme_completion_test,
                                                        max_norm = max_test,
                                                        min_norm = min_test,
                                                        avg_tensor = avg_tensor,
                                                        model=model,
                                                        diffusion_step=diffusion_step,
                                                        device=device,
                                                        args=args,
                                                        model_decoder=model_autoencoder,
                                                        max_enc=max_enc,
                                                        min_enc=min_enc)
    if args.partial:
        wandb.log({"Partial MSE":mse})
        
    #spackle_layer(adata_test=adata_test, imputation_data=imputation_data, args=args)
    #adata_mse, list_probs = get_prob_mse(adata_test=adata_test, imputation_data=imputation_data, args=args)
    #plot_mse_plot(adata=adata_mse, list_mse=list_probs, args=args)

    
if __name__=='__main__':
    main()
# Concatenate all latent representations
#DiT_latent_representations = torch.cat(DiT_latent_representations, dim=0)