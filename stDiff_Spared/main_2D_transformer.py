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
from encoder import Encoder
from decoder import Decoder
from autoencoder_mse import Autoencoder
from autoencoder_LSTM import Autoencoder_LSTM
from Transformer_encoder_decoder import *
from Transformer_simple import Transformer

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
    from spared_stdiff.datasets import get_dataset


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
    if args.vlo:
        # Carga el archivo .h5ad
        adata_128 = sc.read_h5ad(os.path.join('Example_dataset', 'adata.h5ad'))
    else:
        dataset = get_dataset(args.dataset)
        adata_128 = dataset.adata
    # Masking
    #prob_tensor = get_mask_prob_tensor(masking_method="mask_prob", dataset=dataset, mask_prob=0.3, scale_factor=0.8)
    # Add neccesary masking layers in the adata object
    #mask_exp_matrix(adata=adata, pred_layer=pred_layer, mask_prob_tensor=prob_tensor, device=device)

    ### AUTOENCODER ADATA ###
    num_genes = adata_128.shape[1]
    #num_genes = 128
    
    adata = ad.read_h5ad(f'/media/SSD4/dvegaa/ST_Diffusion/stDiff_Spared/adata_1024/{args.dataset}_1024.h5ad')
    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer

    # create mask for 128 genes
    genes_evaluate = []
    genes_128 = adata_128.var["gene_ids"].unique().tolist()
    genes_1024 = adata.var["gene_ids"].unique().tolist()
    
    #Get updated 1024 adata
    #adata = get_1204_adata(adata=adata_128, adata_1024=adata, genes=genes_128)
    #Sort adatas and get genes again
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
    #matrix input
    list_nn, max_min_enc = get_neigbors_dataset(adata, pred_layer, args.num_hops, model_autoencoder, args)
    #train, val, test
    #[[7X1024],[7X1024], ....]
    
    #Transformer model
    num_layers = 2
    n_heads = 2
    embedding_dim =  256
    feedforward_dim = embedding_dim * 2
    
    model_autoencoder = Transformer(input_dim=1024, 
                    latent_dim=128, 
                    output_dim=1024,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=n_heads,
                    lr=args.lr,
                    gene_weights=gene_weights)
    
    checkpoint_path = os.path.join("transformer_models", f"{args.dataset}", "2025-01-07-12-48-45", "autoencoder_model.ckpt") 
    
    checkpoint = torch.load(checkpoint_path)
    model_autoencoder.load_state_dict(checkpoint['state_dict'])
    model_autoencoder.to(device)
    
    #matrix input
    list_nn = encode_transformers(list_nn=list_nn, model_autoencoder=model_autoencoder, batch_size=args.batch_size)
    #7X128
    
    #vector input
    #list_nn, max_min_enc = get_neigbors_dataset(adata, pred_layer, args.num_hops, model_autoencoder, args)
    
    list_nn_masked = mask_extreme_prediction(list_nn) 
    #####TODO: revisar
    
    ### Define splits
    ## Train
    st_data_train, st_data_masked_train, mask_train, max_train, min_train = define_split_nn_mat(list_nn, list_nn_masked, "train", args)
    mask_extreme = np.zeros((mask_train.shape[0], mask_train.shape[1]*8, mask_train.shape[2]))
    mask_extreme_completion_train = get_mask_extreme_completion(adata[adata.obs["split"]=="train"], mask_extreme, genes_evaluate)
    #mask_extreme_completion_train = get_mask_extreme_completion_128(adata_128[adata_128.obs["split"]=="train"], mask_train)
    
    ## Validation
    st_data_valid, st_data_masked_valid, mask_valid, max_valid, min_valid = define_split_nn_mat(list_nn, list_nn_masked, "val", args)
    mask_extreme = np.zeros((mask_valid.shape[0], mask_valid.shape[1]*8, mask_valid.shape[2]))
    mask_extreme_completion_valid = get_mask_extreme_completion(adata[adata.obs["split"]=="val"], mask_extreme, genes_evaluate)
    #mask_extreme_completion_valid = get_mask_extreme_completion_128(adata_128[adata_128.obs["split"]=="val"], mask_valid)
    
    ## Test
    if "test" in splits:
        st_data_test, st_data_masked_test, mask_test, max_test, min_test = define_split_nn_mat(list_nn, list_nn_masked, "test", args)
        mask_extreme = np.zeros((mask_test.shape[0], mask_test.shape[1]*8, mask_test.shape[2]))
        mask_extreme_completion_test = get_mask_extreme_completion(adata[adata.obs["split"]=="test"], mask_extreme, genes_evaluate)
        #mask_extreme_completion_test = get_mask_extreme_completion_128(adata_128[adata_128.obs["split"]=="test"], mask_test)
    
    # Definir un tensor de promedio en caso de predecir una capa delta
    num_deltas = adata.shape[1]
    if "deltas" in pred_layer:
        format = args.prediction_layer.split("deltas")[0]
        avg_tensor = torch.tensor(adata.var[f"{format}log1p_avg_exp"]).view(1, num_deltas)
    else:
        avg_tensor = None
    
    # Define train and valid dataloaders
    train_dataloader = get_data_loader(
        st_data_train, 
        st_data_masked_train, 
        mask_train,
        batch_size=batch_size, 
        is_shuffle=True) #TODO: change to True

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
    #num_nn = (64,3)
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
    
    model.to(device)
    save_path_prefix = args.dataset + "_" + str(args.depth) + "_" + str(args.hidden_size) + "_" + str(args.lr) + "_" + args.loss_type + ".pt"
    
    ## LOAD THE DECODER
    #decoder = Decoder(sizes=list_dec)
    # Load the saved state dictionary
    #decoder.load_state_dict(torch.load(os.path.join("decoder_models", f"{args.dataset}", "fine_tuned_decoder.pt")))

    #decoder = model_autoencoder.decoder()
    ### Train the model
    model.train()
    if not os.path.isfile(save_path_prefix):
        adata_valid = [adata[adata.obs["split"]=="val"], adata_128[adata_128.obs["split"]=="val"]]
        adata_train = [adata[adata.obs["split"]=="train"], adata_128[adata_128.obs["split"]=="train"]]
        normal_train_stDiff(model,
                            adata_train=adata_train,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            valid_data = st_data_valid,
                            valid_masked_data = st_data_masked_valid,
                            mask_valid = mask_valid,
                            mask_extreme_completion = mask_extreme_completion_valid,
                            mask_extreme_completion_train = mask_extreme_completion_train,
                            max_norm = [max_train, max_valid],
                            min_norm = [min_train, min_valid],
                            max_enc = [max_min_enc["val"][0]],
                            min_enc = [max_min_enc["val"][1]],
                            avg_tensor = avg_tensor,
                            wandb_logger=wandb,
                            args=args,
                            adata_valid=adata_valid,
                            model_decoder=model_autoencoder,
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
        adata_test = [adata[adata.obs["split"]=="test"], adata_128[adata_128.obs["split"]=="test"]]
        test_metrics, imputation_data = inference_function(adata=adata_test,
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
                                                            max_enc = [max_min_enc["test"][0]],
                                                            min_enc = [max_min_enc["test"][1]])

        #adata_test = adata[adata.obs["split"]=="test"]
        #adata_test.layers["diff_pred"] = imputation_data
        #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
        #log_pred_image_extreme_completion(adata_test, args, -1)
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        #np.save(f"{args.dataset}_test_imputation_data.npy", imputation_data)
        wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})
        #print(test_metrics)
     
if __name__=='__main__':
    main()
