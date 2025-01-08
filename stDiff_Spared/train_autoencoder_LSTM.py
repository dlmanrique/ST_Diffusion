import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
#from stDiff_Spared.vae import AutoencoderKL, VQModel, VQModelInterface
from autoencoder_LSTM import Autoencoder_LSTM
from spared_stdiff.datasets import get_dataset
from utils import *
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import torch.nn.functional as F
import glob

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

wandb.config = {"dataset": args.dataset}
wandb.log({"dataset": args.dataset, 
            "num_epoch": args.num_epoch,
            "prediction_layer": pred_layer,
            "normalizacion": args.normalization_type})

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
adata = get_1204_adata(adata=adata_128, adata_1024=adata, genes=genes_128)
adata.write(f'/home/dvegaa/ST_Diffusion/stDiff_Spared/adata_1024/{args.dataset}_1024.h5ad')

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

#dataset_names = [args.dataset]
#train_data, val_data, test_data, max_value, min_value = join_dataset(args=args, dataset_names=dataset_names, pred_layer=pred_layer)
data, min_value, max_value = get_autoencoder_data(adata, args.prediction_layer, args)

splits = adata.obs["split"].unique().tolist()
train_data = data["train"]
val_data = data["val"]

if "test" in splits:
    test_data = data["test"]
    test_tensor = torch.stack([torch.tensor(arr) for arr in test_data])
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_tensor = torch.stack([torch.tensor(arr) for arr in train_data])  
val_tensor = torch.stack([torch.tensor(arr) for arr in val_data])

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Initiate model
#list_enc = [num_genes*8, num_genes*4, num_genes]
#list_dec = [num_genes, num_genes*4, num_genes*8]
# list_dec = [num_genes, num_genes, num_genes, num_genes]

model = Autoencoder_LSTM(input_size=1024,
                    hidden_size=128,
                    lr=args.lr,
                    gene_weights=gene_weights).to(device)

# Initialize the Trainer
trainer = pl.Trainer(
    max_epochs=args.num_epoch,
    logger = wandb_logger,
    enable_checkpointing=False
)

# Run the training loop
trainer.fit(model, train_loader, val_loader)


# Save the trained model
if not os.path.exists(os.path.join("lstm_models", f"{args.dataset}", f"{exp_name}")):
    os.makedirs(os.path.join("lstm_models", f"{args.dataset}", f"{exp_name}"))

trainer.save_checkpoint(os.path.join("lstm_models", f"{args.dataset}", f"{exp_name}", "autoencoder_model.ckpt"))

# Load the model for testing
model = Autoencoder_LSTM(input_size=1024,
                    hidden_size=128,
                    lr=args.lr,
                    gene_weights=gene_weights).to(device)

checkpoint_path = os.path.join("lstm_models", f"{args.dataset}", f"{exp_name}", "autoencoder_model.ckpt")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# Test the model
if "test" not in splits:
    test_split = "val"
    test_loader = val_loader
    #adata_test = adata[adata.obs["split"] == "val"]
    adata_test = adata_128[adata_128.obs["split"] == "val"]
else:
    test_split = "test"
    #adata_test = adata[adata.obs["split"] == "test"]
    adata_test = adata_128[adata_128.obs["split"] == "test"]
    
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
#split = adata.obs["split"].unique().tolist()[-1]

auto_data = []
for spot in range(0, auto_pred.shape[0]):
    spot_data = auto_pred[spot]
    auto_data.append(spot_data)

auto_data_array = np.vstack(auto_data) 
gt = torch.tensor(adata_test.layers[args.prediction_layer])
pred = torch.tensor(auto_data_array)
weights_test = gene_weights.unsqueeze(0).repeat(gt.size(0), 1)
mask_boolean = weights_test.bool()
#MSE

mse = F.mse_loss(gt.flatten(), pred[mask_boolean])
#mse = F.mse_loss(gt, pred)
wandb.log({"mse":mse})

#visualize autoencoder reconstruction
#adata_test.layers["autoencoder_pred"] = auto_data_array

#from visualize_autoencoder import log_pred_image
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=20)
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=60)
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=100)
#log_pred_image(adata=adata_test, dataset=args.dataset, slide = "", gene_id=140)
