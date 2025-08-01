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
wandb.init(project="autoencoder_project_2", entity="spared_v2", name=exp_name, dir="/media/SSD4/dvegaa/ST_Diffusion/stDiff_Spared/wandb/run-20241117_171807-oq59brin/")
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


adata_128 = ad.read_h5ad(f"/media/SSD0/pcardenasg2/c_dif_layers/datasets/original/{args.dataset}.h5ad")
num_genes = adata_128.shape[1]

genes_evaluate = []
genes_128 = adata_128.var["gene_ids"].unique().tolist()

for gene in genes_128:
    if gene in genes_128:
        genes_evaluate.append(1)
    else:
        genes_evaluate.append(0)

#breakpoint()
gene_weights = torch.tensor(genes_evaluate, dtype=torch.float32)
model_autoencoder = None
list_nn, max_min_enc = get_neigbors_dataset(adata_128, pred_layer, args.num_hops, model_autoencoder, args)
data = copy.deepcopy(list_nn)

splits = adata_128.obs["split"].unique().tolist()
train_data = data["train"]
val_data = data["val"]

if "test" in splits:
    test_data = data["test"]
    test_tensor = torch.stack([torch.tensor(arr) for arr in test_data])

    mask_extreme = np.zeros((test_tensor.shape[0], num_genes, 7))
    #mask 1024
    mask_extreme_completion_test = get_mask_extreme_completion(adata_128[adata_128.obs["split"]=="test"], mask_extreme, genes_evaluate, args)
    mask_extreme_completion_test = torch.tensor(mask_extreme_completion_test).permute(0,2,1)

    test_dataset = CombinedDataset(test_tensor, test_tensor, mask_extreme_completion_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_tensor = torch.stack([torch.tensor(arr) for arr in train_data])  
mask_extreme = np.zeros((train_tensor.shape[0], num_genes, 7))

#mask 1024
mask_extreme_completion_train = get_mask_extreme_completion(adata_128[adata_128.obs["split"]=="train"], mask_extreme, genes_evaluate, args)
mask_extreme_completion_train = torch.tensor(mask_extreme_completion_train).permute(0,2,1)

val_tensor = torch.stack([torch.tensor(arr) for arr in val_data])
mask_extreme = np.zeros((val_tensor.shape[0], num_genes, 7))

#mask 1024
mask_extreme_completion_val = get_mask_extreme_completion(adata_128[adata_128.obs["split"]=="val"], mask_extreme, genes_evaluate, args)
mask_extreme_completion_val = torch.tensor(mask_extreme_completion_val).permute(0,2,1)

train_dataset = CombinedDataset(train_tensor, train_tensor, mask_extreme_completion_train)
val_dataset = CombinedDataset(val_tensor, val_tensor, mask_extreme_completion_val)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Initiate model
############################################################################################
#Define model
model = Transformer(args=args,
                    input_dim=num_genes, 
                    latent_dim=num_genes, 
                    output_dim=num_genes,
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
if not os.path.exists(os.path.join("/home/dvegaa/ST_Diffusion/stDiff_Spared/autoencoder_128", f"{args.dataset}")):
    os.makedirs(os.path.join("/home/dvegaa/ST_Diffusion/stDiff_Spared/autoencoder_128", f"{args.dataset}"))

trainer.save_checkpoint(os.path.join("/home/dvegaa/ST_Diffusion/stDiff_Spared/autoencoder_128", f"{args.dataset}", "autoencoder_model.ckpt"))

#Load the model for testing
model = Transformer(args=args,
                    input_dim=num_genes, 
                    latent_dim=num_genes, 
                    output_dim=num_genes,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=n_heads,
                    lr=args.lr,
                    gene_weights=gene_weights)

checkpoint_path = os.path.join("/home/dvegaa/ST_Diffusion/stDiff_Spared/autoencoder_128", f"{args.dataset}", "autoencoder_model.ckpt")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# Test the model
if "test" not in splits:
    test_split = "val"
    test_loader = val_loader
    adata_test = adata_128[adata_128.obs["split"] == "val"]
    max_test = max_min_enc["val"][0].item()
    min_test = max_min_enc["val"][1].item()
    mask_extreme_completion_test = mask_extreme_completion_val
else:
    test_split = "test"
    adata_test = adata_128[adata_128.obs["split"] == "test"]
    max_test = max_min_enc["test"][0].item()
    min_test = max_min_enc["test"][1].item()


trainer.test(model, test_loader)

#Get MSE
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
    auto_data.append(spot_data)

auto_data_array = np.stack(auto_data, axis=0)
gt = torch.tensor(adata_test.layers[args.prediction_layer])
pred = torch.tensor(auto_data_array)
mask_boolean = mask_extreme_completion_test.bool()
mask_boolean = mask_boolean[:,0,:]

#MSE
mse = F.mse_loss(gt[mask_boolean], pred[mask_boolean])
wandb.log({"mse":mse})