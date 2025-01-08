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
    
    adata = ad.read_h5ad(f'/media/SSD4/dvegaa/ST_Diffusion/stDiff_Spared/adata_{num_genes*8}/{args.dataset}_{num_genes*8}.h5ad')
    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer
    
    # create mask for 128 genes
    genes_evaluate = []
    genes_128 = adata_128.var["gene_ids"].unique().tolist()
    genes_1024 = adata.var["gene_ids"].unique().tolist()
    
    for gene in genes_1024:
        if gene in genes_128:
            genes_evaluate.append(1)
        else:
            genes_evaluate.append(0)
            
    gene_weights = torch.tensor(genes_evaluate, dtype=torch.float32)

    #Initiate model
    list_enc = [num_genes*8, num_genes*4, num_genes]
    list_dec = [num_genes, num_genes*4, num_genes*8]

    """
    model_autoencoder = Autoencoder(size_list_enc=list_enc,
                        size_list_dec=list_dec,
                        lr=args.lr,
                        gene_weights=gene_weights)
    """
    model_autoencoder = Autoencoder_LSTM(input_size=num_genes*8,
                    hidden_size=num_genes,
                    lr=args.lr,
                    gene_weights=gene_weights).to(device)
    
    #checkpoint_path = os.path.join("autoencoder_models", f"{args.dataset}", "autoencoder_model.ckpt")
    checkpoint_path = os.path.join("lstm_models", f"{args.dataset}", "2024-12-12-16-01-14", "autoencoder_model.ckpt")
    #checkpoint_path = os.path.join("autoencoder_models", f"{args.dataset}", "2024-12-11-17-28-55", "autoencoder_model.ckpt")
    checkpoint = torch.load(checkpoint_path)
    model_autoencoder.load_state_dict(checkpoint['state_dict'])
    model_autoencoder.to(device)
    
    neighbors = 7
    list_nn = get_neigbors_dataset(adata, pred_layer, args.num_hops, model_autoencoder)
    #list_nn_masked = get_neigbors_dataset(adata, 'masked_expression_matrix', args.num_hops)
    list_nn_masked = mask_extreme_prediction(list_nn)
    #####TODO: revisar
    
    ### Define splits
    ## Train
    st_data_train, st_data_masked_train, mask_train, max_train, min_train = define_split_nn_mat(list_nn, list_nn_masked, "train", args)
    
    mask_extreme = np.zeros((mask_train.shape[0], mask_train.shape[1]*8, mask_train.shape[2]))
    mask_extreme_completion_train = get_mask_extreme_completion(adata[adata.obs["split"]=="train"], mask_extreme, genes_evaluate)
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
    num_genes = adata.shape[1]
    if "deltas" in pred_layer:
        format = args.prediction_layer.split("deltas")[0]
        avg_tensor = torch.tensor(adata.var[f"{format}log1p_avg_exp"]).view(1, num_genes)
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

    dit_path = os.path.join("Experiments", "2024-12-12-16-54-17", f"{args.dataset}_12_1024_0.0001_noise.pt")
    #dit_path = os.path.join("Experiments", "2024-12-10-15-46-45", f"{args.dataset}_12_1024_0.0001_noise.pt") #villacampa
    dit_state_dict = torch.load(dit_path)
    model.load_state_dict(dit_state_dict)
    model.to(device)
    model.eval()
    """
    imputation_data = get_dit_predictions(adata=adata[adata.obs["split"]=="train"],
                                                                dataloader=train_dataloader, 
                                                                data=st_data_train, 
                                                                masked_data=st_data_masked_train, 
                                                                mask=mask_train,
                                                                mask_extreme_completion=mask_extreme_completion_train,
                                                                max_norm = max_train,
                                                                min_norm = min_train,
                                                                avg_tensor = avg_tensor,
                                                                model=model,
                                                                diffusion_step=diffusion_step,
                                                                device=device,
                                                                args=args,
                                                             model_autoencoder=model_autoencoder)
    
    imputation_data = imputation_data[:,:,0]
    np.save(f"{args.dataset}_imputation_data.npy", imputation_data)
    """
    breakpoint()
    imputation_data = np.load(f"{args.dataset}_imputation_data.npy")
    #imputation_data_2 = np.load(f"{args.dataset}_10_imputation_data.npy")
    #imputation_data_3 = np.load(f"{args.dataset}_10_imputation_data.npy")
    #imputation_data_4 = np.load(f"{args.dataset}_10_imputation_data.npy")
    
    #imputation_data = np.concatenate([imputation_data_1, imputation_data_2, imputation_data_3, imputation_data_4], axis=0)
    
    import torch.nn.functional as F
    data_tensor = torch.tensor(st_data_train[:,:,0], dtype=torch.float32)
    imputation_tensor = torch.tensor(imputation_data, dtype=torch.float32)
    mse = F.mse_loss(data_tensor, imputation_tensor)
    print(mse)
    
    gt = adata[adata.obs["split"]=="train"].layers[args.prediction_layer]
    
    imputation_tensor = torch.tensor(imputation_data, dtype=torch.float32)
    gt_tensor = torch.tensor(gt, dtype=torch.float32)

    # Create a TensorDataset
    fine_tune_dataset = TensorDataset(imputation_tensor, gt_tensor)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    for param in model_autoencoder.encoder.parameters():
        param.requires_grad = False
        
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR 
    
    # Define optimizer and loss function
    #decoder = Decoder(sizes=list_dec)
    # Load the saved state dictionary
    #decoder.load_state_dict(torch.load(os.path.join("decoder_models", f"{args.dataset}", "fine_tuned_decoder.pt")))
    breakpoint()
    decoder = model_autoencoder.decoder  # Get the decoder from the autoencoder
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    scheduler = StepLR(decoder_optimizer, step_size=1500, gamma=0.1)  # Adjust `step_size` and `gamma` as needed

    # Fine-tuning loop
    decoder.train()
    n_epochs = 3000  # Number of fine-tuning epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Save the finetuned decoder
    if not os.path.exists(os.path.join("decoder_models", f"{args.dataset}")):
        os.makedirs(os.path.join("decoder_models", f"{args.dataset}"))
    
    for epoch in range(n_epochs):
        epoch_total_loss = 0.0
        epoch_important_loss = 0.0
        epoch_auxiliary_loss = 0.0
        
        for latent_batch, target_batch in fine_tune_loader:
            latent_batch = latent_batch.to(device)
            target_batch = target_batch.to(device)
            decoder_input = torch.zeros(latent_batch.shape[0], latent_batch.shape[1])
            # Forward pass through the decoder
            reconstructed, _ = decoder(decoder_input, latent_batch)

            # Compute weighted loss
            #weights = model_autoencoder.weights.unsqueeze(0).repeat(reconstructed.size(0), 1).to(device)
            weights_test = model_autoencoder.weights.unsqueeze(0).repeat(reconstructed.size(0), 1)
            important_mask = (weights_test == 1).bool()  # Máscara para genes importantes
            auxiliary_mask = (weights_test == 0).bool()  # Máscara para genes auxiliares
            
            # Pérdida para genes importantes
            important_loss = F.mse_loss(target_batch[important_mask], reconstructed[important_mask])

            # Pérdida para genes auxiliares
            auxiliary_loss = F.mse_loss(target_batch[auxiliary_mask], reconstructed[auxiliary_mask])

            # Combinar las pérdidas con un peso alpha
            alpha = 0.9
            total_loss = alpha * important_loss + (1 - alpha) * auxiliary_loss

            # Backpropagation and optimization
            decoder_optimizer.zero_grad()
            total_loss.backward()
            decoder_optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_important_loss += important_loss.item()
            epoch_auxiliary_loss += auxiliary_loss.item()
            
        scheduler.step()
        # Calculate average loss for the epoch
        avg_total_loss = epoch_total_loss / len(fine_tune_loader)
        avg_important_loss = epoch_important_loss / len(fine_tune_loader)
        avg_auxiliary_loss = epoch_auxiliary_loss / len(fine_tune_loader)
        # Log the loss to wandb
        wandb.log({"epoch": epoch + 1, "loss total": avg_total_loss, "loss important": avg_important_loss, "loss auxiliary": avg_auxiliary_loss})

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss total: {avg_total_loss}, Loss important: {avg_important_loss}, Loss auxiliary: {avg_auxiliary_loss}")
    
        # Save the fine-tuned decoder
        torch.save(decoder.state_dict(), os.path.join("decoder_models", f"{args.dataset}", "fine_tuned_decoder.pt"))   

if __name__=='__main__':
    main()
# Concatenate all latent representations
#DiT_latent_representations = torch.cat(DiT_latent_representations, dim=0)