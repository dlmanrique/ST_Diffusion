import os
import torch
from process_stDiff.data_2D import *
from utils import *
from visualization_results.visualize_imputation import *
from Transformer_encoder_decoder import *
from scipy.sparse import csr_matrix

#path_medians = "/home/dvegaa/ST_Diffusion/stDiff_Spared/baseline_results/villacampa_lung_organoid/2024-11-27-06-45-33/adata_2024-11-27-06-45-33.h5ad"
#path_medians = "/home/dvegaa/ST_Diffusion/stDiff_Spared/baseline_results/vicari_human_striatium/2024-11-27-07-14-09/adata_2024-11-27-07-14-09.h5ad"
#path_medians = "/home/dvegaa/ST_Diffusion/stDiff_Spared/baseline_results/mirzazadeh_mouse_bone/2024-11-27-07-11-28/adata_2024-11-27-07-11-28.h5ad"

def get_mask_preds_layers(adata_test, pred_layer, genes, mask_percentage, args):
    
    p = mask_percentage

    path_mask = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_mask.pt")
    path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_preds.pt")
    

    #path_mask = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, "spackle_mask.pt")
    #path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset,"spackle_preds.pt")
    

    mask = torch.load(path_mask, map_location="cpu")  # Load first file
    spackle_preds = torch.load(path_spackle, map_location="cpu")  # Load second file

    # Convert to NumPy arrays if they are not already
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    if isinstance(spackle_preds, torch.Tensor):
        spackle_preds = spackle_preds.numpy()
    
    # Save as layers in adata
    adata_test[1].layers[f"spackle_mask_{p}"] = mask
    adata_test[1].layers[f"spackle_pred_{p}"] = spackle_preds
    
    #La máscara tiene True en los valores masqueados y False en los no masqueados (ejemplo: 80% True)
    adata_test[1].layers[pred_layer] = adata_test[1].layers[pred_layer]*(1-mask) # Pongo 0 en los valores masqueados y pred_layer en los no masqueados (ejemplo: 80% 0)
    indices = [i for i, value in enumerate(genes) if value == 1]
    #Actualizar directamente los valores de los genes compartidos
    adata_test[0].layers[pred_layer][:, indices] = adata_test[1].layers[pred_layer][:, :]
   
    return adata_test[0], adata_test[1]    
    
def median_layer(adata_test, imputation_data, splits, path_medians, args):
    adata_test[0].layers["diff_pred"] = imputation_data.detach().cpu().numpy()
    genes_to_keep = adata_test[1].var["gene_ids"]
    subset_mask = adata_test[0].var['gene_ids'].isin(genes_to_keep)
    adata_subsampled = adata_test[0][:, subset_mask].copy()
    
    # Agregar capa de medianas y sumarle el avg_tensor
    adata_visualization = ad.read_h5ad(path_medians)
    format = args.prediction_layer.split("deltas")[0]
    avg_tensor_128 = torch.tensor(adata_visualization.var[f"{format}log1p_avg_exp"]).view(1, adata_visualization.shape[1])
    avg_tensor_128_np = avg_tensor_128.cpu().numpy()
    avg_tensor_sparse = csr_matrix(avg_tensor_128_np)
    avg_tensor_dense = avg_tensor_sparse.toarray() if isinstance(avg_tensor_sparse, csr_matrix) else avg_tensor_sparse
    expanded_avg_tensor = np.broadcast_to(avg_tensor_dense, adata_visualization.layers["median_prediction_expression_matrix"].shape)

    # Suma directa
    adata_visualization.layers["median_prediction_expression_matrix"] += expanded_avg_tensor
    if 'test' in splits:
        adata_visualization = adata_visualization[adata_visualization.obs["split"]=="test"]
    else:
        adata_visualization = adata_visualization[adata_visualization.obs["split"]=="val"]
    
    # Add median and diffusion prediction layer to the 128 adata
    adata_test[1].layers["median_pred"] = adata_visualization.layers["median_prediction_expression_matrix"]
    adata_test[1].layers["diff_pred"] = adata_subsampled.layers["diff_pred"]
    #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
    
    from visualization_results.visualize_imputation import plot_pred_image
    plot_pred_image(adata = adata_test[1], exp_name = args.dataset, n_genes = 3, slide = "")


def get_prob_median_layer(adata_test, imputation_data, args):
    
    # Add diffusion prediction layer to 128 adata
    adata_test[0].layers["diff_pred"] = imputation_data.detach().cpu().numpy()
    genes_to_keep = adata_test[1].var["gene_ids"]
    subset_mask = adata_test[0].var['gene_ids'].isin(genes_to_keep)
    adata_subsampled = adata_test[0][:, subset_mask].copy()
    adata_test[1].layers["diff_pred"] = adata_subsampled.layers["diff_pred"]
    #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
    # Load tensors from .pt files

    
    path_mask = os.path.join("/home/pcardenasg/pcardenasg2/SpaCKLE_testing", args.dataset, "spackle_mask.pt")
    path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, "spackle_preds.pt")
    
    #path_mask = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_mask.pt")
    #path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_preds.pt")
    
    mask = torch.load(path_mask, map_location="cpu")  # Load first file
    spackle_preds = torch.load(path_spackle, map_location="cpu")  # Load second file

    # Convert to NumPy arrays if they are not already
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    if isinstance(spackle_preds, torch.Tensor):
        spackle_preds = spackle_preds.numpy()

    # Save as layers in adata
    adata_test[1].layers[f"spackle_mask"] = mask
    adata_test[1].layers[f"spackle_pred"] = spackle_preds
    
    #Get only prediction only in the masked values and gt on the rest
    gt_layer = adata_test[1].layers["c_t_log1p"]
    diff_preds = np.where(mask, adata_test[1].layers["diff_pred"], gt_layer)
    adata_test[1].layers["diff_pred"] = diff_preds
    
    spackle_preds = np.where(mask, adata_test[1].layers[f"spackle_pred"], gt_layer)
    adata_test[1].layers[f"spackle_pred"] = spackle_preds
        
    from visualization_results.visualize_mask_completion import plot_pred_image
    plot_pred_image(adata = adata_test[1], exp_name = args.dataset, n_genes = 3, slide = "", metric2select_genes="mse")
    
    

def spackle_layer(adata_test, imputation_data, args):
    
    # Add diffusion prediction layer to 128 adata
    adata_test[0].layers["diff_pred"] = imputation_data.detach().cpu().numpy()
    genes_to_keep = adata_test[1].var["gene_ids"]
    subset_mask = adata_test[0].var['gene_ids'].isin(genes_to_keep)
    adata_subsampled = adata_test[0][:, subset_mask].copy()
    adata_test[1].layers["diff_pred"] = adata_subsampled.layers["diff_pred"]
    #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
    # Load tensors from .pt files
    percentages = ["10", "30", "50", "70", "80"]
    
    for p in percentages:
        
        path_mask = os.path.join("/home/pcardenasg/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}_2", p, "spackle_mask.pt")
        path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}_2", p, "spackle_preds.pt")
        
        #path_mask = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_mask.pt")
        #path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_preds.pt")
        
        mask = torch.load(path_mask, map_location="cpu")  # Load first file
        spackle_preds = torch.load(path_spackle, map_location="cpu")  # Load second file

        # Convert to NumPy arrays if they are not already
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        if isinstance(spackle_preds, torch.Tensor):
            spackle_preds = spackle_preds.numpy()

        # Save as layers in adata
        adata_test[1].layers[f"spackle_mask_{p}"] = mask
        adata_test[1].layers[f"spackle_pred_{p}"] = spackle_preds
        
        #Get only prediction only in the masked values and gt on the rest
        gt_layer = adata_test[1].layers["c_t_log1p"]
        diff_preds = np.where(mask, adata_test[1].layers["diff_pred"], gt_layer)
        adata_test[1].layers["diff_pred"] = diff_preds
        
        spackle_preds = np.where(mask, adata_test[1].layers[f"spackle_pred_{p}"], gt_layer)
        adata_test[1].layers[f"spackle_pred_{p}"] = spackle_preds
        
    from visualization_results.visualize_partial_completion import plot_pred_image
    plot_pred_image(adata = adata_test[1], exp_name = args.dataset, n_genes = 3, slide = "", metric2select_genes="mse")
    
    
def get_prob_mse(adata_test, imputation_data, args):
    
    # Add diffusion prediction layer to 128 adata
    adata_test[0].layers["diff_pred"] = imputation_data.detach().cpu().numpy()
    genes_to_keep = adata_test[1].var["gene_ids"]
    subset_mask = adata_test[0].var['gene_ids'].isin(genes_to_keep)
    adata_subsampled = adata_test[0][:, subset_mask].copy()
    adata_test[1].layers["diff_pred"] = adata_subsampled.layers["diff_pred"]
    #torch.save(imputation_data, os.path.join('Predictions', f'predictions_{args.dataset}.pt'))
    
    mask_percentage = "30"
    # Load tensors from .pt files
    path_mask = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{mask_percentage}", mask_percentage, "spackle_mask.pt")
    path_spackle = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{mask_percentage}", mask_percentage, "spackle_preds.pt")
    
    mask = torch.load(path_mask, map_location="cpu")  # Load first file
    spackle_preds = torch.load(path_spackle, map_location="cpu")  # Load second file

    # Convert to NumPy arrays if they are not already
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    if isinstance(spackle_preds, torch.Tensor):
        spackle_preds = spackle_preds.numpy()

    # Save as layers in adata
    adata_test[1].layers["spackle_mask"] = mask
    adata_test[1].layers["spackle_pred"] = spackle_preds
    
    #Get only prediction only in the masked values and gt on the rest
    gt_layer = adata_test[1].layers["c_t_log1p"]
    diff_preds = np.where(mask, adata_test[1].layers["diff_pred"], gt_layer)
    adata_test[1].layers["diff_pred"] = diff_preds
    
    spackle_preds = np.where(mask, adata_test[1].layers["spackle_pred"], gt_layer)
    adata_test[1].layers["spackle_pred"] = spackle_preds
    
    #Get masking percentage
    partial_mask = adata_test[1].layers["random_mask"]
    list_percentage = []
    for i in range(partial_mask.shape[1]):
        list_percentage.append(partial_mask[:,i].sum()/partial_mask.shape[0])
        
    return adata_test[1], list_percentage


def mask_autoencoder(adata, pred_layer, genes, p, args):
    #path_mask = os.path.join("/media/SSD0/pcardenasg2/SpaCKLE_testing", args.dataset, f"mask_{p}", p, "spackle_mask.pt")
    mask = adata[1].layers["random_mask"]
    adata[1].layers[f"masked_{pred_layer}"] = adata[1].layers[pred_layer]*(1-mask) # Pongo 0 en los valores masqueados y pred_layer en los no masqueados (ejemplo: 80% 0)
    
    indices = [i for i, value in enumerate(genes) if value == 1]
    
    #Actualizar directamente los valores de los genes compartidos
    adata[0].layers[f"masked_{pred_layer}"] = copy.deepcopy(adata[0].layers[pred_layer])
    adata[0].layers[f"masked_{pred_layer}"][:, indices] = adata[1].layers[f"masked_{pred_layer}"][:, :]
    adata[0].layers["mask_evaluation"] = (adata[0].layers["masked_c_t_deltas"] == 0)
    return adata[0], adata[1] 
