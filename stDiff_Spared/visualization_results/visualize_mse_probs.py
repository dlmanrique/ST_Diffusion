import anndata as ad
from spared.metrics import get_metrics
#from metrics_stdiff import get_metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import squidpy as sq
import argparse
import os

def plot_mse_plot(adata, list_mse, args):
    
    gt_layer = adata.layers['c_t_log1p']
    pred_layer = adata.layers['diff_pred'] 
    mask_layer = adata.layers['spackle_mask'] 
    
    # Get detailed metrics from partition for diffusion
    diffusion_detailed_metrics = get_metrics(
        gt_mat = gt_layer, 
        pred_mat = pred_layer,
        mask = mask_layer,
        detailed=True
    ) 
    
    mse_genes = diffusion_detailed_metrics["detailed_mse_gene"]
    
    # Plot figure
    plt.figure(figsize=(10, 6))
    plt.scatter(list_mse, mse_genes, marker='o', label='MSE Comparison')
    
    plt.xlabel("Masking percentage")
    plt.ylabel("MSE Genes")
    plt.title("Masking probability vs MSE per gene")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"mse_probs_plots/{args.dataset}_mse_probs.jpg", dpi=300, bbox_inches='tight')
    
    