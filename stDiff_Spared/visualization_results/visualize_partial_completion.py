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
import matplotlib.gridspec as gridspec
from get_csv import read_csv_to_dict
import seaborn as sn
import matplotlib.cm as cm

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


def log_genes_for_slide(gene, slide_adata, experiment_name = 'results', set_name = '', is_deltas=False, pretrained_transformer_results=False, model2select_genes='diffusion', metric2select_genes='mse'):
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
    #order_dict = {}
    #for i, gene in enumerate(genes):
    #    order_dict[gene] = i

    # Get the layers of additional prediction methods. TODO: include more layer retrievals if needed. Layers should be already found in the adata and already dequantized.
    # Set gt layer
    gt_layer = "c_t_log1p"
    # Set pred layer
    pred_layer = "diff_pred"
    # Set spackle layer
    spackle_layer = "spackle_pred"

    # Declare figure TODO: modify number of columns if needed (ncols = mask + # of pred methods + gt + linear plot)
    num_cols = 5
    fig, ax = plt.subplots(nrows=3, ncols=5, layout='constrained')
    fig.set_size_inches(20, 4 * 3)
    
    # Get current row
    #row = order_dict[g]
    # Get min and max of the selected top genes in the slide
    gene_min_pred = slide_adata[:, gene].layers[pred_layer].min() 
    gene_max_pred = slide_adata[:, gene].layers[pred_layer].max()
    
    gene_min_spackle = slide_adata[:, gene].layers[f"{spackle_layer}_10"].min() 
    gene_max_spackle = slide_adata[:, gene].layers[f"{spackle_layer}_10"].max()
    
    gene_min_gt = slide_adata[:, gene].layers[gt_layer].min() 
    gene_max_gt = slide_adata[:, gene].layers[gt_layer].max() 
    
    gene_min = min([gene_min_pred, gene_min_spackle, gene_min_gt])
    gene_max = max([gene_max_pred, gene_max_spackle, gene_max_gt])
    
    mask_visualizatons = ["10", "50", "70"]
    
    for i, p in enumerate(mask_visualizatons):

        # Set MSE diffusion
        mse_diffusion = str(round(slide_adata.var[f"diffusion_mse_test_{p}"][gene], 3))

        # Set MSE spackle
        mse_spackle = str(round(slide_adata.var[f"spackle_mse_test_{p}"][gene], 3))

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
        
        plot_mask = (1-slide_adata.layers[f"spackle_mask_{p}"])
        plot_mask = plot_mask.astype(bool) #True en los valores reales y False en los valores masqueados
        # valores reales en donde la máscara es True y NaN en donde la mascara el False
        gt_masked = np.where(plot_mask, slide_adata.layers[gt_layer], np.nan)
        slide_adata.layers["gt_masked"] = gt_masked
        
        # Plot layers
        slide_adata.layers[f"spackle_mask_{p}"] = slide_adata.layers[f"spackle_mask_{p}"].astype(int)
        if i == 2:
            sq.pl.spatial_scatter(slide_adata, color=[gene], layer="gt_masked", fig=fig, ax=ax[0,i+2], cmap='jet', norm=norm, colorbar=True, title="", na_color="black")
            sq.pl.spatial_scatter(slide_adata, color=[gene], layer=f"{spackle_layer}_{p}", fig=fig, ax=ax[1,i+2], cmap='jet', norm=norm, colorbar=True, title="", na_color="black")   
            sq.pl.spatial_scatter(slide_adata, color=[gene], layer=pred_layer, fig=fig, ax=ax[2,i+2], cmap='jet', norm=norm, colorbar=True, title="", na_color="black")
        else:
            sq.pl.spatial_scatter(slide_adata, color=[gene], layer="gt_masked", fig=fig, ax=ax[0,i+2], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
            sq.pl.spatial_scatter(slide_adata, color=[gene], layer=f"{spackle_layer}_{p}", fig=fig, ax=ax[1,i+2], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")   
            sq.pl.spatial_scatter(slide_adata, color=[gene], layer=pred_layer, fig=fig, ax=ax[2,i+2], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
          
        # Set titles
        ax[0, i+2].set_title(f'{p}% Missing Values', fontsize='xx-large')
        ax[1, i+2].set_title(f'MSE = {mse_spackle}', fontsize='xx-large')
        ax[2, i+2].set_title(f'MSE = {mse_diffusion}', fontsize='xx-large')
    
        # Set y labels
        ax[0,i+2].set_ylabel('')
        ax[0,i+2].set_xticks([])
        ax[0,i+2].set_yticks([])
        ax[1,i+2].set_ylabel('')
        ax[2,i+2].set_ylabel('')
        
        # Set x labels 
        ax[0,i+2].set_xlabel('')
        ax[1,i+2].set_xlabel('')
        ax[2,i+2].set_xlabel('')
    
    sq.pl.spatial_scatter(slide_adata, color=[gene], layer=gt_layer, fig=fig, ax=ax[2,1], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
    ax[2,1].set_ylabel('Ground Truth', fontsize='xx-large')
    ax[2,1].set_xlabel('')
    
    ax[0,2].set_ylabel('Mask', fontsize='xx-large')
    ax[1,2].set_ylabel('Spackle', fontsize='xx-large')
    ax[2,2].set_ylabel('Diffusion', fontsize='xx-large')
    
    percentages = ["10", "30", "50", "70", "80"]
    mse_diff = []
    mse_spackle = []
    for p in percentages:
        mse_diff.append(round(slide_adata.var[f"diffusion_mse_test_{p}"].mean(), 3))
        mse_spackle.append(round(slide_adata.var[f"spackle_mse_test_{p}"].mean(), 3))
    
    gs = gridspec.GridSpec(3, num_cols, height_ratios=[0.8, 0.8, 0.9], width_ratios=[0.8, 0.8, 1, 1, 1], figure=fig)  # Make graph even shorter
    ax_graph = fig.add_subplot(gs[1, 0:2])
    # Plot MSE comparison graph
    percentages_int = [10,30,50,70,80]
    ax_graph.plot(percentages_int, mse_spackle, label="Spackle MSE", marker="o", linestyle="--", color="blue")
    ax_graph.plot(percentages_int, mse_diff, label="Diffusion MSE", marker="o", linestyle="--", color="red")
    ax_graph.set_xlabel("% Missing Values", fontsize='x-large')
    ax_graph.set_ylabel("Completion MSE", fontsize='x-large')
    ax_graph.legend()
    ax_graph.set_title("Completion MSE vs. Missing Values", fontsize='x-large')
    
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xlabel('')
            ax[i,j].set_xticks([])
            ax[i,j].set_ylabel('')
            ax[i,j].set_yticks([])
            
    
    # **Remove upper and right borders**
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)
    
    dict_mse = read_csv_to_dict("mse_dataset.csv")
    violin_spackle = dict_mse["Spackle"]
    violin_diff = dict_mse["Diffusion"]
    
    ax_violin = fig.add_subplot(gs[0, 0:2])
    data = [violin_spackle, violin_diff]
    
    labels = ["Spackle", "Diffusion"]

    # Create the violin plot
    sn.violinplot(data=data, ax=ax_violin, palette=["blue", "red"])
    ax_violin.set_xticks([0, 1])
    ax_violin.set_xticklabels(labels)
    ax_violin.set_ylabel("Completion MSE", fontsize='x-large')
    ax_violin.set_title("Completions MSE for all SpaRED Datasets", fontsize='x-large')
    
    ax[2,0].set_xlabel('')
    ax[2,0].set_xticks([])
    ax[2,0].set_ylabel('')
    ax[2,0].set_yticks([])

    # Remove upper and right borders
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)
    
    # Manually Create the Colorbar
    #sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    #cbar = fig.colorbar(sm, cax=ax[2])  # Use a separate axis for colorbar
    #cbar.ax.set_ylabel(gene, rotation=270, labelpad=15)  # Label the colorbar
    
    #for i in [0,1,2]:
    #    ax[i,5].set_xlabel('')
    #    ax[i,5].set_xticks([])
    #    ax[i,5].set_ylabel('')
    #    ax[i,5].set_yticks([])
        
    # Make graph smaller by adjusting layout
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.12, hspace=0.3, wspace=0.1)
    
    # Format figure
    for i, axis in enumerate(ax.flatten()):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if ((i+1)%num_cols) != 0: 
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
            
    fig_path = os.path.join('mask_results', experiment_name)
    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, f'{gene}_preds_{experiment_name}_{set_name}_{model2select_genes}_{metric2select_genes}.png'))

        
def plot_pred_image(adata: ad.AnnData, exp_name: str, n_genes: int = 3, slide = "", is_deltas=False, pretrained_transformer_results=False, model2select_genes='diffusion', metric2select_genes='mse'):
    """
    This function receives an adata with the prediction layer of: diffusion, 
    and plots the visualizations of the predictions.

    Args:
        adata (ad.AnnData): adata containing the predictions, masks and groundtruth of the imputation methods.
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
    """
    percentages = ["10", "30", "50", "70", "80"]
    
    for p in percentages:
        # Get diffusion gt, preds and mask
        gt_layer = adata.layers['c_t_log1p']
        pred_layer = adata.layers['diff_pred'] 
        spackle_layer = adata.layers[f"spackle_pred_{p}"]
        mask_layer = adata.layers[f'spackle_mask_{p}'].astype(bool) 

        # Get detailed metrics from partition for diffusion
        diffusion_detailed_metrics = get_metrics(
            gt_mat = gt_layer, 
            pred_mat = pred_layer,
            mask = mask_layer,
            detailed=True
        ) 
        
        spackle_detailed_metrics = get_metrics(
            gt_mat = gt_layer, 
            pred_mat = spackle_layer,
            mask = mask_layer,
            detailed=True
        ) 
        
        # Add global metrics to adata
        adata.var[f'diffusion_pcc_global_{p}'] = diffusion_detailed_metrics['PCC-Gene']
        adata.var[f'diffusion_mse_global_{p}'] = diffusion_detailed_metrics['MSE']
        # Add detalied metrics to adata
        #adata.var['diffusion_pcc_test'] = diffusion_detailed_metrics['detailed_PCC-Gene']
        adata.var[f'diffusion_mse_test_{p}'] = diffusion_detailed_metrics['detailed_mse_gene']
        
        # Add global metrics to adata
        adata.var[f'spackle_pcc_global_{p}'] = spackle_detailed_metrics['PCC-Gene']
        adata.var[f'spackle_mse_global_{p}'] = spackle_detailed_metrics['MSE']
        # Add detalied metrics to adata
        #adata.var['spackle_pcc_test'] = spackle_detailed_metrics['detailed_PCC-Gene']
        adata.var[f'spackle_mse_test_{p}'] = spackle_detailed_metrics['detailed_mse_gene']

    gene_len = len(diffusion_detailed_metrics['detailed_mse_gene'])

    # Get selected genes based on the best and worst mse
    selected_genes = []
    n_bottom = adata.var.nlargest(gene_len, columns=f'{model2select_genes}_{metric2select_genes}_test_10').index.to_list()
    n_top = adata.var.nsmallest(gene_len, columns=f'{model2select_genes}_{metric2select_genes}_test_10').index.to_list()
    
    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
    
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    
    #gene = n_top[10]
    gene = "9"
    log_genes_for_slide(
            gene=gene, 
            slide_adata=slide_adata, 
            experiment_name=exp_name,
            is_deltas=is_deltas,
            pretrained_transformer_results=pretrained_transformer_results,
            model2select_genes=model2select_genes,
            metric2select_genes=metric2select_genes
        )
    
    # Best and worst genes to plot
    #selected_genes.append(n_top)
    #selected_genes.append(n_bottom)

    #top_bottom = ["Top_Genes", "Bottom_Genes"]

    #print('Creating visualization plots ...')
    
    

    """
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
    """
