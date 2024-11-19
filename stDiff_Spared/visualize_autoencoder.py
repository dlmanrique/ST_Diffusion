import anndata as ad
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch
import pandas as pd
# FIXME: merge get_metrics functions from metrics.py and metrics_ids
import matplotlib
import matplotlib.pyplot as plt
import squidpy as sq
import wandb
import argparse
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
from autoencoder import AutoencoderKL, VQModel, VQModelInterface
from autoencoder_mse import Autoencoder
from spared.datasets import get_dataset
from utils import *
from tqdm import tqdm
import torch.nn.functional as F

def log_pred_image(adata: ad.AnnData, dataset, slide = "", gene_id=0):
    """
    This function receives an adata with the prediction layers of the median imputation model and transformer
    imputation model and plots the visualizations to compare the performance of both methods.

    Args:
        adata (ad.AnnData): adata containing the predictions, masks and groundtruth of the imputations methods.
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
    """
    # Define visualization layers
    autoencoder_layer = "autoencoder_pred"
    gt_layer = "c_t_log1p"
    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
        
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    
    # TODO: finish documentation for the log_genes_for_slides function
    def log_genes_for_slide(slide_adata, autoencoder_layer, gt_layer, gene_id):
        """
        This function receives a slide adata and the names of the prediction, groundtruth and masking layers 
        and logs the visualizations for the top and bottom genes

        Args:
            trans_layer (str): name of the layer that contains the transformer imputation model
            median_layer (str): name of the layer that contains median imputation model
            mask_layer (str): name of the mask layer
        """
        max_missing_gene = {}
        
        # Get the slide
        slide = list(slide_adata.obs.slide_id.unique())[0]
        print(slide)
        # Define gene to plot
        gene = slide_adata.var.gene_ids[gene_id]
        print(gene)
        """
        for i in range(0, slide_adata.layers["mask_visualization"].shape[1]):
            mask=slide_adata.layers["mask_visualization"][:,i]
            nan_val =  np.sum(np.isnan(mask))
            max_missing_gene[i] = nan_val
        
        sorted_dict = dict(sorted(max_missing_gene.items(), key=lambda item: item[1], reverse=True))   
        print("10 Genes IDs with max missing values: ",list(sorted_dict.items())[0:40])
        """
        # Declare figure
        fig, ax = plt.subplots(nrows=1, ncols=2, layout='constrained')
        fig.set_size_inches(12, 6)
        
        # Find min and max of gene for color map
        gene_min_gt = slide_adata[:, gene].layers[gt_layer].min() 
        gene_max_gt = slide_adata[:, gene].layers[gt_layer].max()
        
        gene_min_auto = slide_adata[:, gene].layers[autoencoder_layer].min() 
        gene_max_auto = slide_adata[:, gene].layers[autoencoder_layer].max()
        
        gene_min = min([gene_min_gt, gene_min_auto])
        gene_max = max([gene_max_gt, gene_max_auto])

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

        # Plot gt and pred of gene in the specified slides
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=gt_layer, ax=ax[0], cmap='jet', norm=norm, colorbar=False, title="")
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=autoencoder_layer, ax=ax[1], cmap='jet', norm=norm, colorbar=False, title="")
        #crop_coord=(4000,4000,42000,43000)
        
        # Eliminate Labels
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        
        ax[0].set_ylabel('')
        ax[1].set_ylabel('')
        
        # Format figure
        for axis in ax.flatten():
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)           
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
            
        # Set titles
        ax[0].set_title('Ground Truth (c_t_log1p)', fontsize='xx-large')
        ax[1].set_title('Autoencoder Prediction', fontsize='xx-large')
        
        # Log plot 
        #wandb.log({top_bottom: fig})
        if not os.path.exists(f"/home/dvegaa/ST_Diffusion/stDiff_Spared/visualizations_autoencoder/{dataset}/experimento_no_norm/"):
            os.makedirs(f"/home/dvegaa/ST_Diffusion/stDiff_Spared/visualizations_autoencoder/{dataset}/experimento_no_norm/")
        
        fig.savefig(f"/home/dvegaa/ST_Diffusion/stDiff_Spared/visualizations_autoencoder/{dataset}/experimento_no_norm/{slide}_{gene}_{gene_id}.jpg")
        
    log_genes_for_slide(slide_adata=slide_adata, gt_layer=gt_layer, autoencoder_layer=autoencoder_layer, gene_id=gene_id)


# LOAD PREDICTION TO VISUALIZE
#dataset_name = "abalo_human_squamous_cell_carcinoma" #mse = 0.63 (100) --> 0.5834 (200)
#dataset_name = "villacampa_lung_organoid" #mse = 0.5880
#dataset_name = "mirzazadeh_mouse_bone" #mse = 0.3997
#dataset_name = 'mirzazadeh_mouse_brain_p1'#mse = 0.4616 
#dataset_name = 'mirzazadeh_mouse_brain_p2'#mse = 0.4925
#dataset_name = 'mirzazadeh_mouse_brain'#mse = 0.4178
dataset_name = "abalo_human_squamous_cell_carcinoma" #mse = 0.75 no norm (100) --> 0.5895 no norm (200)

dataset = get_dataset(dataset_name)
adata = dataset.adata

splits = adata.obs["split"].unique().tolist()
pred_layer = "c_t_log1p"

# Get neighbors
neighbors = 7
list_nn = get_neigbors_dataset(adata, pred_layer, num_hops=1)
concatenated_array = np.concatenate([np.array(sublist) for sublist in list_nn])

# Obtenemos el valor máximo y mínimo de todo el arreglo
max_value = np.max(concatenated_array)
min_value = np.min(concatenated_array)

print("Valor máximo:", max_value)
print("Valor mínimo:", min_value)
# Load data
train_data = []
val_data = []
test_data = []

for i, split in enumerate(list_nn):
    for spots in split:
        data = spots.permute(1,0).unsqueeze(dim=0)
        if i == 0:
            data = normalize_to_minus_one_to_one(data, max_value, min_value)
            train_data.append(data)
        elif i == 1:
            data = normalize_to_minus_one_to_one(data, max_value, min_value)
            val_data.append(data)
        elif i == 2:
            data = normalize_to_minus_one_to_one(data, max_value, min_value)
            test_data.append(data)
            
    
train_tensor = torch.stack(train_data)    
val_tensor = torch.stack(val_data)    
test_tensor = torch.stack(test_data) 

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)
test_dataset = TensorDataset(test_tensor)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)


trainer = pl.Trainer(
    max_epochs=50
)

model = Autoencoder()

checkpoint_path = os.path.join("autoencoder_models", "2024-11-18-17-13-08", "autoencoder_model.ckpt")  # Replace with your checkpoint file path
#checkpoint_path = "last_autoencoder_model.ckpt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
trainer.test(model, test_loader)
#auto_pred = model(test_tensor)

auto_pred = []

model = Autoencoder()

checkpoint_path = os.path.join("autoencoder_models", "2024-11-18-17-13-08", "autoencoder_model.ckpt") # Replace with your checkpoint file path
#checkpoint_path = "last_autoencoder_model.ckpt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# Disable gradient calculation for inference
with torch.no_grad():
    for data in tqdm(test_loader):
        # Move data to the specified device
        inputs = data[0].to("cuda")
        # Make predictions
        outputs = model(inputs)
        # Move outputs to CPU and convert to NumPy if needed
        auto_pred.append(outputs.cpu().numpy())

auto_pred = np.concatenate(auto_pred, axis=0)
adata_test = adata[adata.obs["split"] == "test"]

auto_data = []
for spot in range(0, auto_pred.shape[0]):
    spot_data = auto_pred[spot,:,:,0]
    spot_data = denormalize_from_minus_one_to_one(spot_data, max_value, min_value)
    auto_data.append(spot_data)

auto_data_array = np.vstack(auto_data) 
gt = torch.tensor(adata_test.layers["c_t_log1p"])
pred = torch.tensor(auto_data_array)

#MSE
breakpoint()
mse = F.mse_loss(gt, pred) 
print(mse)

adata_test.layers["autoencoder_pred"] = auto_data_array
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=0)
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=20)
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=40)
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=60)
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=80)
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=100)
log_pred_image(adata=adata_test, dataset=dataset_name, slide = "", gene_id=120)

    
