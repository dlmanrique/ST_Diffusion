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
        is_shuffle=True)

    if len(splits) == 3:
        test_dataloader, norm_st_data_test, max_test, min_test = get_data_loader_image_to_gene(
        st_data_test, # Datos de expresion de la layer que es
        features_test, # Features de los parches asociados
        batch_size=batch_size, 
        is_shuffle=True)

    ### DIFFUSION MODEL ##########################################################################
    num_nn = st_data_train[0].shape

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
        test_metrics, imputation_data = inference_function(dataloader=test_dataloader,
                                        data= norm_st_data_test, 
                                        model=model,
                                        max_norm = max_test,
                                        min_norm = min_test,
                                        avg_tensor = avg_tensor,
                                        diffusion_step=args.diffusion_steps_train,
                                        device=device,
                                        args=args
                                        )

        adata_test = adata[adata.obs["split"]=="test"]
        adata_test.layers["diff_pred"] = imputation_data
        #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
        #log_pred_image_extreme_completion(adata_test, args, -1)
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})
        #print(test_metrics)




if __name__=='__main__':
    main()