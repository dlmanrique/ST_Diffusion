import os
import warnings
import torch

from model_stDiff.stDiff_model_2D import DiT_stDiff
from process_stDiff.data_2D import *
import anndata as ad
from spared.datasets import get_dataset

from utils import *

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


### Wandb 
wandb.login()
if args.debbug_wandb:
    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="debbugs", entity="spared_v2", name=exp_name + '_debbug')

else:
    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="stDiff_Modelo_2D_partial", entity="spared_v2", name=exp_name + '_inference')

# FIXME: change this to an option that not involves comments in the code


#load_path = 'Experiments/2024-12-14-22-05-35/vicari_human_striatium_12_1024_0.0001_noise.pt'
#load_path = 'Experiments/2024-12-14-22-02-56/mirzazadeh_human_small_intestine_12_1024_0.0001_noise.pt'
#load_path = 'Experiments/2024-12-09-20-46-07/villacampa_mouse_brain_12_1024_0.0001_noise.pt'
#load_path = 'Experiments/2024-12-09-19-35-56/abalo_human_squamous_cell_carcinoma_12_1024_0.0001_noise.pt'
#load_path = 'Experiments/2024-12-09-09-23-05/mirzazadeh_mouse_bone_12_1024_0.0001_noise.pt'
#load_path = 'Experiments/2024-12-09-09-02-30/10xgenomic_mouse_brain_sagittal_posterior_12_1024_0.0001_noise.pt'
#load_path = 'Experiments/2024-12-03-14-55-38/villacampa_lung_organoid_12_1024_0.0001_noise.pt'


args.dataset =  args.load_path.split('_12')[0].split('/')[-1]

wandb.config = {"lr": args.lr, "dataset": args.dataset}
wandb.log({"lr": args.lr, 
            "dataset": args.dataset, 
            "num_epoch": args.num_epoch, 
            "num_heads": args.head,
            "depth": args.depth,
            "hidden_size": args.hidden_size, 
            "load_path": args.load_path,
            "loss_type": args.loss_type,
            "concat_dim": args.concat_dim,
            "masked_loss": args.masked_loss,
            "model_type": args.model_type,
            "scheduler": args.scheduler,
            "layer": args.prediction_layer,
            "normalizacion": args.normalization_type,
            "batch_size": args.batch_size,
            "num_hops": args.num_hops, 
            "partial": True,
            "diffusion_steps_train": args.diffusion_steps_train, 
            "diffusion_steps_test": args.diffusion_steps_test})

### Parameters
# Define the training parameters
lr = args.lr
depth = args.depth
num_epoch = args.num_epoch
diffusion_step_test = args.diffusion_steps_test
batch_size = args.batch_size
hidden_size = args.hidden_size
head = args.head
device = torch.device('cuda')

# Get dataset
dataset = get_dataset(args.dataset)
adata = dataset.adata
splits = adata.obs["split"].unique().tolist() #['train', 'val', 'test']
pred_layer = args.prediction_layer

# Masking
prob_tensor = get_mask_prob_tensor(masking_method="mask_prob", dataset=dataset, mask_prob=0.3, scale_factor=0.8)
# Add neccesary masking layers in the adata object
mask_exp_matrix(adata=adata, pred_layer=pred_layer, mask_prob_tensor=prob_tensor, device=device)

# Get neighbors
neighbors = 7
dict_nn = get_neigbors_dataset(adata, pred_layer, args.num_hops)
dict_nn_masked = get_neigbors_dataset(adata, 'masked_expression_matrix', args.num_hops)
### Define splits
## Validation
st_data_valid, st_data_masked_valid, mask_valid, max_valid, min_valid = define_split_nn_mat(dict_nn, dict_nn_masked, "val", args)

## Test
if "test" in splits:
    st_data_test, st_data_masked_test, mask_test, max_test, min_test = define_split_nn_mat(dict_nn, dict_nn_masked, "test", args)

# Definir un tensor de promedio en caso de predecir una capa delta
num_genes = adata.shape[1]
if "deltas" in pred_layer:
    format = args.prediction_layer.split("deltas")[0]
    avg_tensor = torch.tensor(adata.var[f"{format}log1p_avg_exp"]).view(1, num_genes)
else:
    avg_tensor = None


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
num_nn = st_data_valid[0].shape

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
model.load_state_dict(torch.load(args.load_path))

if "test" in splits:
    test_metrics, imputation_data = inference_function(dataloader=test_dataloader, 
                                data=st_data_test, 
                                masked_data=st_data_masked_test, 
                                mask=mask_test,
                                max_norm = max_test,
                                min_norm = min_test,
                                avg_tensor = avg_tensor,
                                model=model,
                                diffusion_step=diffusion_step_test,
                                device=device,
                                args=args)

    adata_test = adata[adata.obs["split"]=="test"]
    adata_test.layers["diff_pred"] = imputation_data
    #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}_completion_parcial_inference_extreme.pt'))
    wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})
else:
    valid_metrics, imputation_data = inference_function(dataloader=valid_dataloader, 
                                data=st_data_valid, 
                                masked_data=st_data_masked_valid, 
                                mask=mask_valid,
                                max_norm = max_valid,
                                min_norm = min_valid,
                                avg_tensor = avg_tensor,
                                model=model,
                                diffusion_step=diffusion_step_test,
                                device=device,
                                args=args)

    adata_test = adata[adata.obs["split"]=="val"]
    adata_test.layers["diff_pred"] = imputation_data
    #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}_completion_parcial_inference_extreme.pt'))
    wandb.log({"val_MSE": valid_metrics["MSE"], "val_PCC": valid_metrics["PCC-Gene"]})