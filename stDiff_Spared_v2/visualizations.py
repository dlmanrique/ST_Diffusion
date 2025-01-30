from metrics import get_metrics
import anndata as ad
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import squidpy as sq
import argparse
import os

def log_genes_for_slide(dataset_name, genes, slide_adata, input_mask_layer, experiment_name = 'results', set_name = '', model2select_genes='diffusion', metric2select_genes='mse', consider_imputed_values=False):
    """
    This function receives a slide adata and the names of the prediction, groundtruth and masking layers 
    and logs the visualizations for the top and bottom genes

    Args:
        genes (list): genes to visualize
        slide_adata (AnnData): slide AnnData
        input_mask_layer (str): name of the input mask for visualizations
        experiment_name (str, optional): experiment name. Defaults to 'results'.
        set_name (str, optional): set name. Defaults to ''. It can be 'Top_Genes' or 'Bottom_Genes'.
        model2select_genes (str, optional): model to select genes. Defaults to 'diffusion'.
        metric2select_genes (str, optional): metric to select genes. Defaults to 'mse'.
        consider_imputed_values (bool, optional): consider imputed values or no. Plots include empy spots if false. Defaults to False.
    """

    # Define order of rows in dict
    order_dict = {}
    for i, gene in enumerate(genes):
        order_dict[gene] = i

    # Set gt layer
    gt_layer = "c_t_log1p"
    # Set diffusion pred layer
    diffusion_pred_layer = "diffusion_preds"
    # Set stnet pred layer
    stnet_pred_layer = "stnet_preds"

    if not consider_imputed_values:
        # Create real layers for only plot real values
        gt_real = np.where(slide_adata.layers[input_mask_layer], slide_adata.layers[gt_layer], np.nan)
        diffusion_real = np.where(slide_adata.layers[input_mask_layer], slide_adata.layers[diffusion_pred_layer], np.nan)
        stnet_real = np.where(slide_adata.layers[input_mask_layer], slide_adata.layers[stnet_pred_layer], np.nan)
        gt_layer = "gt_real"
        diffusion_pred_layer = "diffusion_real"
        stnet_pred_layer = "stnet_real"
        slide_adata.layers[gt_layer] = gt_real
        slide_adata.layers[diffusion_pred_layer] = diffusion_real
        slide_adata.layers[stnet_pred_layer] = stnet_real

    # Declare figure TODO: modify number of columns if needed (ncols = gt + # of pred methods + linear plot)
    num_cols = 4
    fig, ax = plt.subplots(nrows=len(genes), ncols=num_cols, layout='constrained')
    fig.set_size_inches(22, 4 * len(genes))

    # Iterate over the genes
    for g in genes: 

        # Get current row
        row = order_dict[g]

        # Get min and max of the selected top genes in the slide        
        gene_min_gt = np.nanmin(slide_adata[:, g].layers[gt_layer]) 
        gene_max_gt = np.nanmax(slide_adata[:, g].layers[gt_layer])

        gene_min_diffusion = np.nanmin(slide_adata[:, g].layers[diffusion_pred_layer])
        gene_max_diffusion = np.nanmax(slide_adata[:, g].layers[diffusion_pred_layer])

        gene_min_stnet = np.nanmin(slide_adata[:, g].layers[stnet_pred_layer]) 
        gene_max_stnet = np.nanmax(slide_adata[:, g].layers[stnet_pred_layer])
        
        gene_min = min([gene_min_gt, gene_min_diffusion, gene_min_stnet])
        gene_max = max([gene_max_gt, gene_max_diffusion, gene_max_stnet])

        # Set PCC
        pcc_stnet = str(round(slide_adata.var["stnet_pcc_test"][g], 3))
        pcc_diffusion = str(round(slide_adata.var["diffusion_pcc_test"][g], 3))
        
        # Set MSE
        mse_stnet = str(round(slide_adata.var["stnet_mse_test"][g], 3))
        mse_diffusion = str(round(slide_adata.var["diffusion_mse_test"][g], 3))

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
        norm_gt = matplotlib.colors.Normalize(vmin=gene_min_gt, vmax=gene_max_gt)
        norm_stnet = matplotlib.colors.Normalize(vmin=gene_min_stnet, vmax=gene_max_stnet)
        norm_diffusion = matplotlib.colors.Normalize(vmin=gene_min_diffusion, vmax=gene_max_diffusion)
                
        # Plot layers
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=gt_layer, fig=fig, ax=ax[row,0], cmap='jet', norm=norm, colorbar=True, title="")
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=diffusion_pred_layer, fig=fig, ax=ax[row,1], cmap='jet', norm=norm, colorbar=False, title="")
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=stnet_pred_layer, fig=fig, ax=ax[row,2], cmap='jet', norm=norm, colorbar=False, title="")

        # Set titles
        ax[row, 1].set_title(f'PCC = {pcc_diffusion} & MSE = {mse_diffusion}', fontsize='xx-large')
        ax[row, 2].set_title(f'PCC = {pcc_stnet} & MSE = {mse_stnet}', fontsize='xx-large')
        
        # Set y labels
        slide_name = list(slide_adata.obs.slide_id.unique())[0]
        ax[row,0].set_ylabel(f'{g}:\n{slide_name}\n', fontsize='xx-large')
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

        # Define models prediction and ground truth (only true spots)
        true_gt = gene_adata.layers[gt_layer][gene_adata.layers[input_mask_layer]==True]
        true_stnet_pred = gene_adata.layers[stnet_pred_layer][gene_adata.layers[input_mask_layer]==True]
        true_diffusion_pred = gene_adata.layers[diffusion_pred_layer][gene_adata.layers[input_mask_layer]==True]
        
        # Plot gen predictions and ground truth
        ax[row,3].plot(true_gt, true_gt, color="black", linestyle="-", label="Ground Truth")
        ax[row,3].plot(true_gt, true_stnet_pred, color="orange", marker="o",  markersize=3, linestyle="None", label=f"stnet\nPCC = {pcc_stnet} & MSE = {mse_stnet}")
        ax[row,3].plot(true_gt, true_diffusion_pred, color="green", marker="o",  markersize=3, linestyle="None", label=f"Diffusion\nPCC = {pcc_diffusion} & MSE = {mse_diffusion}")
        ax[row,3].legend(markerfirst=3, framealpha=0.4, loc="center left", bbox_to_anchor=(1, 0.5))
        ax[row,3].set_xlabel("Ground Truth")
        ax[row,3].set_ylabel("Prediction")
    
    
    # Format figure
    for i, axis in enumerate(ax.flatten()):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if ((i+1)%num_cols) != 0: 
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
    
    # Set PCC
    pcc_stnet = str(round(slide_adata.var["stnet_pcc_test"][genes[0]], 3))
    pcc_diffusion = str(round(slide_adata.var["diffusion_pcc_test"][genes[0]], 3))
    # Set MSE
    mse_stnet = str(round(slide_adata.var["stnet_mse_test"][genes[0]], 3))
    mse_diffusion = str(round(slide_adata.var["diffusion_mse_test"][genes[0]], 3))

    # Set titles
    ax[0, 0].set_title('Ground Truth', fontsize='xx-large')
    ax[0, 1].set_title(f'Diffusion\nPCC = {pcc_diffusion} & MSE = {mse_diffusion}', fontsize='xx-large')
    ax[0, 2].set_title(f'stnet\nPCC = {pcc_stnet} & MSE = {mse_stnet}', fontsize='xx-large')
    ax[0, 3].set_title('Pred vs Trues', fontsize='xx-large')

    fig_path = os.path.join('qualitative_results', dataset_name, experiment_name)
    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, f'{set_name}_{model2select_genes}_{metric2select_genes}.png'))
        

def plot_pred_image(dataset_name, adata, stnet_preds: torch.Tensor, diffusion_preds: torch.Tensor, exp_name: str, n_genes: int = 3, slide = "", model2select_genes='diffusion', metric2select_genes='mse'):
    """
    This function receives the predictions of stnet and difussion model, as well as the gt and mask for visualizing the predictions comparison.

    Args:
        dataset_name (str): dataset name
        adata (AnnData): test adata
        stnet_preds (torch.Tensor): stnet predictions
        diffusion_preds (torch.Tensor): diffusion model predictions
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the test adata.
    """
    
    # Add predictions to adata
    adata.layers["diffusion_preds"] = np.array(diffusion_preds.cpu())
    adata.layers["stnet_preds"] = np.array(stnet_preds.cpu())

    # Get detailed metrics for stnet
    stnet_detailed_metrics = get_metrics(
        gt_mat = adata.layers["c_t_log1p"], 
        pred_mat = adata.layers["stnet_preds"],
        mask = adata.layers["mask"],
        detailed=True
    ) 

    # Get detailed metrics from partition for diffusion
    diffusion_detailed_metrics = get_metrics(
        gt_mat = adata.layers["c_t_log1p"], 
        pred_mat = adata.layers["diffusion_preds"],
        mask = adata.layers["mask"],
        detailed=True
    )
    
    # Add detalied metrics to adata
    adata.var['stnet_pcc_test'] = stnet_detailed_metrics['detailed_PCC-Gene']
    adata.var['stnet_mse_test'] = stnet_detailed_metrics['detailed_mse_gene']
    adata.var['diffusion_pcc_test'] = diffusion_detailed_metrics['detailed_PCC-Gene']
    adata.var['diffusion_mse_test'] = diffusion_detailed_metrics['detailed_mse_gene']

    gene_len = len(diffusion_detailed_metrics['detailed_PCC-Gene'])
    
    # Get selected genes based on the best and worst mse
    selected_genes = []
    n_top = adata.var.nlargest(gene_len, columns=f'{model2select_genes}_{metric2select_genes}_test').index.to_list()
    n_bottom = adata.var.nsmallest(gene_len, columns=f'{model2select_genes}_{metric2select_genes}_test').index.to_list()
    
    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        test_data_available = True if 'test' in adata.obs['split'].unique() else False
        test_data = adata[adata.obs["split"]=="test"] if test_data_available else  adata[adata.obs["split"]=="val"]
        slide = list(test_data.obs.slide_id.unique())[0]
    
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    
    # Takes top and worst genes that contain at leats 5% of missing spots
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
            dataset_name=dataset_name,
            genes=gene, 
            slide_adata=slide_adata, 
            input_mask_layer='mask',
            experiment_name=exp_name,
            set_name=top_bottom[i],
            model2select_genes=model2select_genes,
            metric2select_genes=metric2select_genes,
            consider_imputed_values=False,
        )

def visualize_predictions(dataset_name, adata, pred_data, exp_name):

    # Get stnet predictions
    stnet_preds = torch.load(os.path.join("predictions_stnet",f"{dataset_name}.pt"))

    plot_pred_image(
        dataset_name = dataset_name,
        adata = adata,
        stnet_preds = stnet_preds, 
        diffusion_preds = pred_data,
        exp_name = exp_name, 
        n_genes = 3, 
        slide = "",
        model2select_genes="diffusion",
        metric2select_genes="mse",
    )
