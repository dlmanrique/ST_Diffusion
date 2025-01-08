import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
#from stDiff_Spared.vae import AutoencoderKL, VQModel, VQModelInterface
from Transformer_encoder_decoder import *
from Transformer_simple import Transformer
from spared_stdiff.datasets import get_dataset
from utils import *
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import torch.nn.functional as F
import glob
from scipy.sparse import hstack, csr_matrix
import copy
from transformer_dataloader import CombinedDataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
current_device = torch.cuda.current_device()

#argparse
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args) #Not uses, maybe later usage

# Configurar el logger de wandb
wandb.login()
exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="autoencoder_project", entity="spared_v2", name=exp_name, dir="/media/SSD4/dvegaa/ST_Diffusion/stDiff_Spared/wandb/run-20241117_171807-oq59brin/")
wandb_logger = WandbLogger(log_model="best")

#organ = "brain"
pred_layer = args.prediction_layer

#Tranformer parameters
num_layers = 2
n_heads = 2
embedding_dim = 256
feedforward_dim = embedding_dim * 2

wandb.config = {"dataset": args.dataset}
wandb.log({"dataset": args.dataset, 
            "num_epoch": args.num_epoch,
            "prediction_layer": pred_layer,
            "normalizacion": args.normalization_type,
            "num_transformer_layers": num_layers,
            "n_heads": n_heads,
            "embedding_dim": embedding_dim,
            "feedforward_dim": feedforward_dim})

# Load data
from spared.filtering import *
from spared.layer_operations import *
from spared.denoising import *
from spared.gene_features import *

path_list = glob.glob("/home/dvegaa/ST_Diffusion/stDiff_Spared/spared_stdiff/processed_data/**/*.h5ad", recursive=True)

for path in path_list:
    if args.dataset in path:
        if "raw" in path:
            path_dataset = path
           
#path_dataset = '/home/dvegaa/ST_Diffusion/stDiff_Spared/spared_stdiff/processed_data/mirzazadeh_data/mirzazadeh_mouse_brain/2023-12-04-20-32-05/adata_raw.h5ad'   
#adata, num_genes = get_new_adatas(path_dataset)

dataset = get_dataset(args.dataset)
adata_128 = dataset.adata 
num_genes = adata_128.shape[1]
adata = ad.read_h5ad(f'/home/dvegaa/ST_Diffusion/stDiff_Spared/adata_1024/{args.dataset}_1024.h5ad')
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
#param_dict = dataset.param_dict
model_autoencoder = None
#list_nn, max_min_enc = get_neigbors_dataset(adata, pred_layer, args.num_hops, model_autoencoder, args)
#data = copy.deepcopy(list_nn)

# Apply the function to all tensors in the dictionary
#for key in list_nn.keys():
#    for spot in range(len(list_nn[key])):
#        list_nn[key][spot] = torch.roll(list_nn[key][spot], shifts=-1, dims=0)

#dataset_names = [args.dataset]
#train_data, val_data, test_data, max_value, min_value = join_dataset(args=args, dataset_names=dataset_names, pred_layer=pred_layer)
#data, min_value, max_value = get_autoencoder_data(adata, args.prediction_layer, args)
list_nn, max_min_enc = get_neigbors_dataset(adata, pred_layer, args.num_hops, model_autoencoder, args)
data = copy.deepcopy(list_nn)

splits = adata.obs["split"].unique().tolist()
train_data = data["train"]
val_data = data["val"]

if "test" in splits:
    test_data = data["test"]
    test_tensor = torch.stack([torch.tensor(arr) for arr in test_data])

    mask_extreme = np.zeros((test_tensor.shape[0], 1024, 7))
    #mask 1024
    mask_extreme_completion_test = get_mask_extreme_completion(adata[adata.obs["split"]=="test"], mask_extreme, genes_evaluate)
    #mask_extreme_completion_test = torch.tensor(mask_extreme_completion_test[:,:,0])
    mask_extreme_completion_test = torch.tensor(mask_extreme_completion_test).permute(0,2,1)

    #mask 128
    #mask_extreme[:,:,0] = 1
    #mask_extreme_completion_test = get_mask_extreme_completion_128(adata_128[adata_128.obs["split"]=="test"], mask_extreme)
    test_dataset = CombinedDataset(test_tensor, mask_extreme_completion_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_tensor = torch.stack([torch.tensor(arr) for arr in train_data])  
mask_extreme = np.zeros((train_tensor.shape[0], 1024, 7))
#mask 1024
mask_extreme_completion_train = get_mask_extreme_completion(adata[adata.obs["split"]=="train"], mask_extreme, genes_evaluate)
mask_extreme_completion_train = torch.tensor(mask_extreme_completion_train).permute(0,2,1)

#mask 128
#mask_extreme[:,:,0] = 1
#mask_extreme_completion_train = get_mask_extreme_completion_128(adata_128[adata_128.obs["split"]=="train"], mask_extreme)

val_tensor = torch.stack([torch.tensor(arr) for arr in val_data])
mask_extreme = np.zeros((val_tensor.shape[0], 1024, 7))
#mask 1024
mask_extreme_completion_val = get_mask_extreme_completion(adata[adata.obs["split"]=="val"], mask_extreme, genes_evaluate)
mask_extreme_completion_val = torch.tensor(mask_extreme_completion_val).permute(0,2,1)

#mask 128
#mask_extreme[:,:,0] = 1
#mask_extreme_completion_val = get_mask_extreme_completion_128(adata_128[adata_128.obs["split"]=="val"], mask_extreme)


train_dataset = CombinedDataset(train_tensor, mask_extreme_completion_train)
val_dataset = CombinedDataset(val_tensor, mask_extreme_completion_val)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Initiate model
############################################################################################
#Define model
model = Transformer(input_dim=1024, 
                    latent_dim=128, 
                    output_dim=1024,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=n_heads,
                    lr=args.lr,
                    gene_weights=gene_weights)

# Initialize the Trainer
trainer = pl.Trainer(
    max_epochs=args.num_epoch,
    logger = wandb_logger,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    enable_checkpointing=False
)

# Run the training loop
trainer.fit(model, train_loader, val_loader)


# Save the trained model
if not os.path.exists(os.path.join("transformer_models", f"{args.dataset}", f"{exp_name}")):
    os.makedirs(os.path.join("transformer_models", f"{args.dataset}", f"{exp_name}"))

trainer.save_checkpoint(os.path.join("transformer_models", f"{args.dataset}", f"{exp_name}", "autoencoder_model.ckpt"))

#Load the model for testing
model = Transformer(input_dim=1024, 
                    latent_dim=128, 
                    output_dim=1024,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=n_heads,
                    lr=args.lr,
                    gene_weights=gene_weights)

checkpoint_path = os.path.join("transformer_models", f"{args.dataset}", f"{exp_name}", "autoencoder_model.ckpt")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# Test the model
if "test" not in splits:
    test_split = "val"
    test_loader = val_loader
    adata_test = adata[adata.obs["split"] == "val"]
    #adata_test = adata_128[adata_128.obs["split"] == "val"]
    max_test = max_min_enc["val"][0].item()
    min_test = max_min_enc["val"][1].item()
else:
    test_split = "test"
    adata_test = adata[adata.obs["split"] == "test"]
    #adata_test = adata_128[adata_128.obs["split"] == "test"]
    max_test = max_min_enc["test"][0].item()
    min_test = max_min_enc["test"][1].item()


trainer.test(model, test_loader)

#Get MSE
#adata = get_dataset(args.dataset).adata
auto_pred = []

with torch.no_grad():
    for data in tqdm(test_loader):
        # Move data to the specified device
        inputs = data[0].to(device)
        inputs = inputs.float()
        model = model.to(device)
        # Make predictions
        outputs = model(inputs, inputs.shape[0])
        # Move outputs to CPU and convert to NumPy if needed
        auto_pred.append(outputs.cpu().numpy())
        
auto_pred = np.concatenate(auto_pred, axis=0)

auto_data = []
for spot in range(0, auto_pred.shape[0]):
    spot_data = auto_pred[spot]
    if args.normalize_encoder == "1-1":
        spot_data = denormalize_from_minus_one_to_one(spot_data, max_test, min_test)
    #auto_data.append(spot_data[0]) #TODO: si predigo todo y no solo la primera columna
    auto_data.append(spot_data)

#auto_data_array = np.vstack(auto_data) 
auto_data_array = np.stack(auto_data, axis=0)
gt = torch.tensor(adata_test.layers[args.prediction_layer])
pred = torch.tensor(auto_data_array)
#weights_test = gene_weights.unsqueeze(0).repeat(gt.size(0), 1)
breakpoint()
mask_boolean = mask_extreme_completion_test.bool()
mask_boolean = mask_boolean[:,0,:]
#MSE
mse = F.mse_loss(gt[mask_boolean], pred[mask_boolean])
#mse = F.mse_loss(gt, pred[:,0,:])
#mse = F.mse_loss(gt.flatten(), pred[:,0,:][mask_boolean])
#mse = F.mse_loss(gt, pred)
wandb.log({"mse":mse})

#visualize autoencoder reconstruction
#adata_test.layers["autoencoder_pred"] = auto_data_array

#from visualize_autoencoder import log_pred_image
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=20)
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=60)
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=100)
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=140)
