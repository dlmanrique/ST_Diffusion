import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
from autoencoder import AutoencoderKL, VQModel, VQModelInterface
from autoencoder_mse import Autoencoder
from spared.datasets import get_dataset
from utils import *
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import torch.nn.functional as F

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
            "normalizacion": args.normalization_type,
            "num_res_blocks": args.num_res_blocks,
            "ch": args.ch,
            "ch_mult": args.ch_mult})

# Load data
dataset_names = [args.dataset]
train_data, val_data, test_data, max_value, min_value = join_dataset(args=args, dataset_names=dataset_names, pred_layer=pred_layer)

train_tensor = torch.stack(train_data)    
val_tensor = torch.stack(val_data)    
test_tensor = torch.stack(test_data) 

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)
test_dataset = TensorDataset(test_tensor)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

#Initiate model
model = Autoencoder(num_res_blocks=args.num_res_blocks,
                    ch=args.ch,
                    ch_mult=args.ch_mult,
                    lr=args.lr)

# Initialize the Trainer
trainer = pl.Trainer(
    max_epochs=args.num_epoch,
    logger = wandb_logger,
    enable_checkpointing=False
)

# Run the training loop
trainer.fit(model, train_loader, val_loader)


# Save the trained model
if not os.path.exists(os.path.join("autoencoder_models", exp_name)):
    os.makedirs(os.path.join("autoencoder_models", exp_name))

trainer.save_checkpoint(os.path.join("autoencoder_models", exp_name, "autoencoder_model.ckpt"))

# Load the model for testing
model = Autoencoder(num_res_blocks=args.num_res_blocks,
                    ch=args.ch,
                    ch_mult=args.ch_mult,
                    lr=args.lr)

checkpoint_path = os.path.join("autoencoder_models", exp_name, "autoencoder_model.ckpt")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])


# Test the model
trainer.test(model, test_loader)

#Get MSE
adata = get_dataset(args.dataset).adata
auto_pred = []

with torch.no_grad():
    for data in tqdm(test_loader):
        # Move data to the specified device
        inputs = data[0]
        # Make predictions
        outputs = model(inputs)
        # Move outputs to CPU and convert to NumPy if needed
        auto_pred.append(outputs.cpu().numpy())

auto_pred = np.concatenate(auto_pred, axis=0)
split = adata.obs["split"].unique().tolist()[-1]
adata_test = adata[adata.obs["split"] == split]

auto_data = []
for spot in range(0, auto_pred.shape[0]):
    spot_data = auto_pred[spot,:,:,0]
    spot_data = denormalize_from_minus_one_to_one(spot_data, max_value, min_value)
    auto_data.append(spot_data)

auto_data_array = np.vstack(auto_data) 
gt = torch.tensor(adata_test.layers["c_t_log1p"])
pred = torch.tensor(auto_data_array)

#MSE
mse = F.mse_loss(gt, pred) 
wandb.log({"mse":mse})