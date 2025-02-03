import anndata as ad
#from spared.spared.metrics import get_metrics
from metrics_stdiff import get_metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import squidpy as sq
import argparse
import os

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Parser for prediction visualization")
        self._add_arguments()

    def _add_arguments(self):
        str2bool = lambda x: (str(x).lower() == 'true')
        self.parser.add_argument('--dataset_name', type=str, default='villacampa_lung_organoid', help='Name of the dataset')
        self.parser.add_argument('--llama_preds_path', type=str, help='Path to the saved llama predictions matrix. (pt file)')
        self.parser.add_argument('--diffusion_preds_path', type=str, help='Path to the saved diffusion predictions matrix. (pt file)')
        self.parser.add_argument('--is_deltas', type=str2bool, default=False, help='Whether or not it is a "Deltas" experiment')
        self.parser.add_argument('--pretrained_transformer_results', type=bool, default=False, help='Whether or not to include the pretrained transformer results')
        self.parser.add_argument('--model2select_genes', type=str, default='diffusion', help='Model to select the best  and worst genes')
        self.parser.add_argument('--metric2select_genes', type=str, default='mse', help='Metric to select the best and  worst genes')

    def parse(self):
        return self.parser.parse_args()


def log_genes_for_slide(genes, slide_adata, experiment_name = 'results', set_name = '', is_deltas=False, pretrained_transformer_results=False, model2select_genes='diffusion', metric2select_genes='mse'):
    """
    This function receives a slide adata and the names of the prediction, groundtruth and masking layers 
    and logs the visualizations for the top and bottom genes

    Args:
        genes (list): genes to visualize
        gene_to_df (dictionary): dictionary of genes and genes in dataframe that correspond to accurate metrics
        slide_adata (AnnData): slide AnnData
        gt_layer (str): name of groundtruth layer
        pred_layer (str): name of the prediction layer 
        random_mask_layer (str): name of the random mask
        input_mask_layer (str): name of the input mask for visualizations

    """
    # Get the slide
    slide = list(slide_adata.obs.slide_id.unique())[0]
    # Define order of rows in dict
    order_dict = {}
    for i, gene in enumerate(genes):
        order_dict[gene] = i

    # Get the layers of additional prediction methods. TODO: include more layer retrievals if needed. Layers should be already found in the adata and already dequantized.
    # Set gt layer
    gt_layer = "c_d_log1p"
    # Set pred layer
    pred_layer = "diff_pred"
    # Set median layer
    median_layer = "median_pred"

    # Declare figure TODO: modify number of columns if needed (ncols = mask + # of pred methods + gt + linear plot)
    num_cols = 4
    fig, ax = plt.subplots(nrows=len(genes), ncols=num_cols, layout='constrained')
    fig.set_size_inches(18, 4 * len(genes))

    # Iterate over the genes
    for g in genes:         
        # Get current row
        row = order_dict[g]
        # Get min and max of the selected top genes in the slide
        gene_min_pred = slide_adata[:, g].layers[pred_layer].min() 
        gene_max_pred = slide_adata[:, g].layers[pred_layer].max()
        
        gene_min_median = slide_adata[:, g].layers[median_layer].min() 
        gene_max_median = slide_adata[:, g].layers[median_layer].max()
        
        gene_min_gt = slide_adata[:, g].layers[gt_layer].min() 
        gene_max_gt = slide_adata[:, g].layers[gt_layer].max() 
        
        gene_min = min([gene_min_pred, gene_min_median, gene_min_gt])
        gene_max = max([gene_max_pred, gene_max_median, gene_max_gt])

        # Set PCC diffusion
        pcc_diffusion = str(round(slide_adata.var["diffusion_pcc_test"][g], 3))
        # Set MSE diffusion
        mse_diffusion = str(round(slide_adata.var["diffusion_mse_test"][g], 3))
        
        # Set PCC median
        pcc_median = str(round(slide_adata.var["median_pcc_test"][g], 3))
        # Set MSE median
        mse_median = str(round(slide_adata.var["median_mse_test"][g], 3))

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
        
        gt_masked = np.where(slide_adata.layers["mask"], slide_adata.layers[gt_layer], np.nan)
        slide_adata.layers["gt_masked"] = gt_masked
        
        # Plot layers
        slide_adata.layers["mask"] = slide_adata.layers["mask"].astype(int)
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=gt_layer, fig=fig, ax=ax[row,0], cmap='jet', norm=norm, colorbar=True, title="", na_color="black")
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=pred_layer, fig=fig, ax=ax[row,1], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=median_layer, fig=fig, ax=ax[row,2], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
        
        # Set titles
        ax[row, 1].set_title(f'PCC = {pcc_diffusion} & MSE = {mse_diffusion}', fontsize='xx-large')
        ax[row, 2].set_title(f'PCC = {pcc_median} & MSE = {mse_median}', fontsize='xx-large')
        
        # Set y labels
        ax[row,0].set_ylabel(f'{g}:\n{slide}\n', fontsize='large')
        ax[row,0].set_xticks([])
        ax[row,0].set_yticks([])
        ax[row,1].set_ylabel('')
        ax[row,2].set_ylabel('')
        
        # Set x labels 
        ax[row,0].set_xlabel('')
        ax[row,1].set_xlabel('')
        ax[row,2].set_xlabel('')
        
        # Define gene adata
        gene_adata = slide_adata[:,g].copy()

        # Define models prediction and ground truth (only masked spots)
        ceros_gt = [gene_adata.layers[gt_layer][gene_adata.layers["mask"]==True]][0]
        ceros_pred = [gene_adata.layers[pred_layer][gene_adata.layers["mask"]==True]][0]
        ceros_median = [gene_adata.layers[median_layer][gene_adata.layers["mask"]==True]][0]

        # Plot gen predictions and ground truth
        ax[row,3].plot(ceros_gt, ceros_gt, color="black", linestyle="-", label="Ground Truth")
        ax[row,3].plot(ceros_gt, ceros_pred, color="blue", marker="o",  markersize=3, linestyle="None", label="Diffusion")
        ax[row,3].plot(ceros_gt, ceros_median, color="red", marker="o",  markersize=3, linestyle="None", label="Median")
        ax[row,3].set_xlabel("Ground Truth")
        ax[row,3].set_ylabel("Prediction")
        ax[row, 3].legend(loc="upper left", fontsize="small")
    
    
    # Format figure
    for i, axis in enumerate(ax.flatten()):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if ((i+1)%num_cols) != 0: 
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
    
    # Set PCC
    pcc_diffusion = str(round(slide_adata.var["diffusion_pcc_test"][genes[0]], 3))
    # Set MSE
    mse_diffusion = str(round(slide_adata.var["diffusion_mse_test"][genes[0]], 3))
    
    # Set PCC median
    pcc_median = str(round(slide_adata.var["median_pcc_test"][genes[0]], 3))
    # Set MSE median
    mse_median = str(round(slide_adata.var["median_mse_test"][genes[0]], 3))

    # Set titles
    ax[0, 0].set_title('Ground Truth', fontsize='xx-large')
    ax[0, 1].set_title(f'Diffusion\nPCC = {pcc_diffusion} & MSE = {mse_diffusion}', fontsize='xx-large')
    ax[0, 2].set_title(f'Median\nPCC = {pcc_median} & MSE = {mse_median}', fontsize='xx-large')

    fig_path = os.path.join('qualitative_results', experiment_name)
    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, f'preds_{experiment_name}_{set_name}_{model2select_genes}_{metric2select_genes}.png'))
        

def plot_pred_image(adata: ad.AnnData, exp_name: str, n_genes: int = 3, slide = "", is_deltas=False, pretrained_transformer_results=False, model2select_genes='diffusion', metric2select_genes='mse'):
    """
    This function receives an adata with the prediction layer of: diffusion, 
    and plots the visualizations of the predictions.

    Args:
        adata (ad.AnnData): adata containing the predictions, masks and groundtruth of the imputation methods.
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
    """
    # Get diffusion gt, preds and mask
    gt_layer = adata.layers['c_t_log1p']
    pred_layer = adata.layers['diff_pred'] 
    median_layer = adata.layers["median_pred"]
    mask_layer = adata.layers['mask'] 

    # Get detailed metrics from partition for diffusion
    diffusion_detailed_metrics = get_metrics(
        gt_mat = gt_layer, 
        pred_mat = pred_layer,
        mask = mask_layer,
        detailed=True
    ) 
    
    median_detailed_metrics = get_metrics(
        gt_mat = gt_layer, 
        pred_mat = median_layer,
        mask = mask_layer,
        detailed=True
    ) 
    
    # Add global metrics to adata
    adata.var['diffusion_pcc_global'] = diffusion_detailed_metrics['PCC-Gene']
    adata.var['diffusion_mse_global'] = diffusion_detailed_metrics['MSE']
    # Add detalied metrics to adata
    adata.var['diffusion_pcc_test'] = diffusion_detailed_metrics['detailed_PCC-Gene']
    adata.var['diffusion_mse_test'] = diffusion_detailed_metrics['detailed_mse_gene']
    
    # Add global metrics to adata
    adata.var['median_pcc_global'] = median_detailed_metrics['PCC-Gene']
    adata.var['median_mse_global'] = median_detailed_metrics['MSE']
    # Add detalied metrics to adata
    adata.var['median_pcc_test'] = median_detailed_metrics['detailed_PCC-Gene']
    adata.var['median_mse_test'] = median_detailed_metrics['detailed_mse_gene']

    gene_len = len(diffusion_detailed_metrics['detailed_PCC-Gene'])

    # Get selected genes based on the best and worst mse
    selected_genes = []
    n_bottom = adata.var.nlargest(gene_len, columns=f'{model2select_genes}_{metric2select_genes}_test').index.to_list()
    n_top = adata.var.nsmallest(gene_len, columns=f'{model2select_genes}_{metric2select_genes}_test').index.to_list()
    
    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
    
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    
    # Takes top and worst genes that contain at leats 10% of missing spots
    top_genes = []
    bottom_genes = []
    print('Extracting top and bottom performing genes ...')
    for g in n_top: 
        gene_adata = slide_adata[:,g].copy()
        true_values = np.count_nonzero(gene_adata.layers['mask'])
        if true_values > (adata.shape[0]*0.05):
            top_genes.append(g)
        if len(top_genes) == n_genes:
            break
    
    for g in n_bottom: 
        gene_adata = slide_adata[:,g].copy()
        true_values = np.count_nonzero(gene_adata.layers['mask'])
        if true_values > (adata.shape[0]*0.05):
            bottom_genes.append(g)
        if len(bottom_genes) == n_genes:
            break

    # Best and worst genes to plot
    selected_genes.append(top_genes)
    selected_genes.append(bottom_genes)

    top_bottom = ["Top_Genes", "Bottom_Genes"]

    print('Creating visualization plots ...')

    
    for i, gene in enumerate(selected_genes):
        log_genes_for_slide(
            genes=gene, 
            slide_adata=slide_adata, 
            experiment_name=exp_name,
            set_name=top_bottom[i],
            is_deltas=is_deltas,
            pretrained_transformer_results=pretrained_transformer_results,
            model2select_genes=model2select_genes,
            metric2select_genes=metric2select_genes
        )

