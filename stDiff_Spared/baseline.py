# Script of the baseline (median completion) for the "extreme completion" task to compare with LLaMA.
from utils import get_spatial_neighbors
import torch
from datetime import datetime
import os
import anndata as ad
from spared.metrics import get_metrics
import json
import squidpy as sq
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
   
def get_parser():
    str2bool = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser(description='Script for extreme completion task baseline using medians.')
    parser.add_argument('--dataset_name', type=str, default='villacampa_lung_organoid', help='The preference dataset to use from SpaRED.')
    parser.add_argument('--num_neighbors', type=int, default=6, help='Amount of neighbors to take into account for calculating the median of the central spot.')
    parser.add_argument('--output_dir', type=str, default='./baseline_results', help='Directory to save results.')
    parser.add_argument('--pred_layer', type=str, default='c_t_log1p', help='Layer from which extract data.')
    parser.add_argument('--adaptive_median', type=str2bool, default=True, help='Whether to use the adaptive median filter or not.')

    return parser.parse_args()

def adaptive_median_filter_pepper(adata: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
    """
    This function computes the adaptive median filter for pairs (obs, gene) with a zero value (peper noise) in the layer 'from_layer' and
    stores the result in the layer 'to_layer'. The max window size is a neighborhood of n_hops defined by the conectivity hex_geometry
    inputed by parameter. This means the number of concentric rings in a graph to take into account to compute the median.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        from_layer (str): the layer to compute the adaptive median filter from.
        to_layer (str): the layer to store the results of the adaptive median filter.
        n_hops (int): the maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                            used to compute the spatial neighbors and only true for visium datasets.

    Returns:
        ad.AnnData: The AnnData object with the results of the adaptive median filter stored in the layer 'to_layer'.
    """
    # Define original expression matrix
    
    original_exp = adata.layers[from_layer]    

    medians = np.zeros((adata.n_obs, n_hops, adata.n_vars))

    # Iterate over the hops
    for i in tqdm(range(1, n_hops+1)):
        
        # Get dictionary of neighbors for a given number of hops
        curr_neighbors_dict = get_spatial_neighbors(adata, i, hex_geometry)

        # Iterate over observations
        for j in range(adata.n_obs):
            # Get the list of indexes of the neighbors of the j'th observation
            neighbors_idx = curr_neighbors_dict[j]
            # Get the expression matrix of the neighbors
            neighbor_exp = original_exp[neighbors_idx, :]
            # Get the median of the expression matrix
            median = np.median(neighbor_exp, axis=0)

            # Store the median in the medians matrix
            medians[j, i-1, :] = median
    
    # Also robustly compute the median of the non-zero values for each gene
    general_medians = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, original_exp)
    general_medians[np.isnan(general_medians)] = 0.0 # Correct for possible nans

    # Define corrected expression matrix
    corrected_exp = np.zeros_like(original_exp)

    ### Now that all the possible medians are computed. We code for each observation:
    
    # Note: i indexes over observations, j indexes over genes
    for i in tqdm(range(adata.n_obs)):
        for j in range(adata.n_vars):
           
            # Definie initial stage and window size
            current_stage = 'A'
            k = 0

            while True:

                # Stage A:
                if current_stage == 'A':
                    
                    # Get median value
                    z_med = medians[i, k, j]

                    # If median is not zero then go to stage B
                    if z_med != 0:
                        current_stage = 'B'
                        continue
                    # If median is zero, then increase window and repeat stage A
                    else:
                        k += 1
                        if k < n_hops:
                            current_stage = 'A'
                            continue
                        # If we have the biggest window size, then return the median
                        else:
                            # NOTE: Big modification to the median filter here. Be careful
                            corrected_exp[i,j] = general_medians[j]
                            break


                # Stage B:
                elif current_stage == 'B':
                    
                    # Get window median
                    z_med = medians[i, k, j]

                    corrected_exp[i,j] = z_med
                    break

    # Add corrected expression to adata
    adata.layers[to_layer] = corrected_exp

    return adata

class STData():
    def __init__(self, adata, pred_layer='c_t_log1p', num_neighs = 6):
        self.adata = adata
        self.pred_layer = pred_layer
        self.num_neighs = num_neighs
        self.expression_mtx = torch.tensor(adata.layers[pred_layer])
        self.mask_layer = torch.tensor(adata.layers["mask"])
        self.adjacency_mt = self.get_adjacency(adata=adata, num_neighs=num_neighs)

    def get_adjacency(self, adata, num_neighs = 6):
        """
        Function description
        """
        # Get num_neighs nearest neighbors for each spot
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=num_neighs)
        adj_mat = torch.tensor(adata.obsp['spatial_connectivities'].todense())
        return adj_mat

    def __getitem__(self, idx):
        item = {}
        # Get expression of center spot and neighbors
        main_spot_expression = self.expression_mtx[idx]
        adjacent_spots_expression = self.expression_mtx[self.adjacency_mt[:,idx]==1.]
        # Get mask of center spot and neighbors
        main_spot_mask = self.mask_layer[idx]
        adjacent_spots_mask = self.mask_layer[self.adjacency_mt[:,idx]==1.]

        item['exp_matrix_gt'] = torch.concat([main_spot_expression.unsqueeze(0), adjacent_spots_expression])
        item['real_missing'] = torch.concat([main_spot_mask.unsqueeze(0), adjacent_spots_mask])
        item['model_input'] = torch.concat([torch.zeros(main_spot_expression.shape).unsqueeze(0), adjacent_spots_expression])

        return item

    def __len__(self):
        return len(self.adata)

def main(adata, args):

    if args.adaptive_median:
        adata = adaptive_median_filter_pepper(adata, from_layer=args.pred_layer, to_layer='median_prediction_expression_matrix', n_hops=4, hex_geometry=True)
        final_gts = adata.layers[args.pred_layer]
        final_masks = adata.layers['mask']
        final_preds = adata.layers['median_prediction_expression_matrix']

    else:
        # Prepare dataset for iterator
        data = STData(adata, pred_layer=args.pred_layer, num_neighs=args.num_neighbors)
        # Declare dataloader
        dataloader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
        # Declare empty matrices that will store the gts, masks, and final predictions
        final_gts = [] 
        final_masks = []
        final_preds = []
        # Iterate over dataloader
        for spot in dataloader:
            gt = spot['exp_matrix_gt'][:,0,:]
            mask = spot['real_missing'][:,0,:]
            neighbors_exp = spot['model_input'][0,1:,:]
            # Get predictions
            prediction = np.median(neighbors_exp, axis=0)
            # Save vectors
            final_gts.append(gt)
            final_masks.append(mask)
            final_preds.append(torch.tensor(prediction).unsqueeze(0)) 
        
        # Prepare data
        final_gts = torch.concat(final_gts)
        final_masks = torch.concat(final_masks)
        final_preds = torch.concat(final_preds)

    # Save median prediction layer in adata
    preds_name = 'median_preds_deltas' if "deltas" in args.pred_layer else 'medians_preds'
    adata.layers[preds_name] = np.asarray(final_preds)
    
    # Replace adata with only difference being the addition of the median prediction layer
    #adata.write(os.path.join(output_dir, f"adata_{exp_name}.h5ad"))
    
    # Temporarily add the reconstructed gts layer to adata
    #reconst_gt_name = 'reconstructed_gts_deltas' if "deltas" in args.pred_layer else 'reconstructed_gts'
    #adata.layers[reconst_gt_name] = np.asarray(final_gts)
    # Check if test split is available
    test_data_available = True if 'test' in adata.obs['split'].unique() else False
    # Get test split
    test_split = adata[adata.obs['split']=='test'] if test_data_available else adata[adata.obs['split']=='val']

    # Calculate metrics

    metrics = get_metrics(gt_mat = test_split.layers[args.pred_layer],
                          pred_mat = test_split.layers[preds_name],
                          mask = test_split.layers["mask"])
    print(metrics)

    # Save txt with metrics
    #file_path = os.path.join(output_dir, 'testing_results.txt')
    #with open(file_path, 'w') as txt_file:
    #    txt_file.write("Median results for extreme imputation:")
    #    # Save parser
    #    args_dict = vars(args)
    #    txt_file.write(json.dumps(args_dict, indent=4))
    #    # Convert the dictionary to a formatted string
    #    dict_str = json.dumps(metrics, indent=4)
    #    txt_file.write(dict_str)


if __name__ == "__main__":

    args = get_parser()

    # Create experiment name
    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(args.output_dir, args.dataset_name, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    '''# Set WandB experiment configs
    run = wandb.init(
        project = 'median_baseline', 
        name = exp_name, 
        entity = 'spared_v2',
        resume = 'allow'
        )'''
    
    adata_path = f"data/datasets/{args.dataset_name}/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    main(adata, args)