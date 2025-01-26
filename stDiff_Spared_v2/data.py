from torch.utils.data import random_split, DataLoader, TensorDataset
from tqdm import tqdm
import anndata as ad
from utils import *
import numpy as np
import torch
import squidpy as sq
from torchvision import transforms
import os


class TransformTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]

        if self.transform:
            x = self.transform(x)
        return x


class stLDMDataset(torch.utils.data.Dataset):
    def __init__(self, args, adata, split_name, genes_autoencoder, image_encoder, image_transforms):
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            adata (ad.AnnData): An anndata object with the data of the entire dataset.
            args (argparse): parser with the values necessary for data processing.
            split_name (str): name of the data split being processed. Useful for identifying which data split the model is being tested on.
            pre_masked (str, optional): specifies if the data incoming has already been masked for testing purposes. 
                    * If True, __getitem__() will return the random mask that was used to mask the original expression 
                    values instead of the median imputation mask, as well as the gt expressions and the masked data.
        """

        self.args = args
        self.pred_layer = args.pred_layer
        self.split_name = split_name
        self.adata = adata
        self.model_autoencoder = genes_autoencoder
        self.image_encoder = image_encoder
        self.image_transforms = transforms.Compose(image_transforms.transforms[-2:])
        # Get original expression matrix based on selected prediction layer.
        self.expression_mtx = torch.tensor(self.adata.layers[self.pred_layer])

        # Calculate patch_features
        self.calculate_patch_embeddings()
      
        # Get adjacency matrix.
        self.adj_mat = None
        self.get_adjacency(self.args.num_neighs)

        # Build and save each spot's neighborhood, and the min and max val of the data split
        self.min_val, self.max_val = np.inf, -np.inf 
        self.neighborhoods = self.build_neighborhoods()

        # Normalize data if needed (data that will be the model's input, i.e. encoded matrices)
        if self.args.normalize_input:
            self.normalize_full_data()


    def get_adjacency(self, num_neighs = 6):
        """
        Function description
        """
        # Get num_neighs nearest neighbors for each spot
        sq.gr.spatial_neighbors(self.adata, coord_type='generic', n_neighs=num_neighs)
        self.adj_mat = torch.tensor(self.adata.obsp['spatial_connectivities'].todense())

    def calculate_patch_embeddings(self):
        """
        This function uses the image encoder and calculates the enbedding for each patch
        """
        #FIXME: add possibility to load the files if it already exists

        os.path.join('Image_features', self.args.dataset, f'{self.split_name}.pt')
                     
        flat_patches = self.adata.obsm[f'patches_scale_1.0']
        patches = flat_patches.reshape((-1, 224, 224, 3))
        patches = np.array(patches)

        patches_dataset = TransformTensorDataset(patches, self.image_transforms)

        dataloader = DataLoader(patches_dataset, batch_size=256, shuffle=False)
        patch_features = []
        for batch in tqdm(dataloader):
            batch = batch.to('cuda')                                  
            batch_output = self.image_encoder(batch)    
            patch_features.append(batch_output)

        self.patch_features = torch.cat(patch_features, dim=0)



    def build_neighborhoods(self):
        """
        Creates a dictionary of dictionaries, where each element/key corresponds to an individual spot in the
        adata, and each inner-dictionary/value corresponds to its own neighborhood's expression matrix,
        gene-expression-mask, and encoded expression matrix.
        """
        all_neighborhoods = {}
        
        for idx, spot_name in enumerate(tqdm(self.adata.obs["unique_id"].unique())):
            # Get gt expression for idx spot and its nn
            spot_exp = self.expression_mtx[idx].unsqueeze(dim=0)
            nn_exp = self.expression_mtx[self.adj_mat[:,idx]==1.]
            exp_matrix = torch.cat((spot_exp, nn_exp), dim=0).type('torch.FloatTensor')
            exp_matrix = exp_matrix.unsqueeze(0)

            
            # Encode neighborhood
            self.model_autoencoder.eval()
            with torch.no_grad():
                encoded_exp_matrix = self.model_autoencoder.encoder(exp_matrix.to("cuda"))

            all_neighborhoods[str(idx)] = {"spot_id": spot_name, 
                                           "exp_matrix": exp_matrix, 
                                           "encoded_exp_matrix": encoded_exp_matrix.detach().cpu(),
                                           'patches': None} #FIXME: check if it is best to leave it in CUDA or in the CPU
            
            # Set min and max values of the data split
            if encoded_exp_matrix.min().item() < self.min_val:
                self.min_val = encoded_exp_matrix.min().item()
            if encoded_exp_matrix.max().item() > self.max_val:
                self.max_val = encoded_exp_matrix.max().item()    

        return all_neighborhoods
    
    def normalize_full_data(self):
        """
        Calls for the normalization function from utils to normalize all neighborhoods/samples
        based on the min and max values of the complete data split.
        """
        for spot_idx in self.neighborhoods.keys():
            encoded_exp_mt = self.neighborhoods[spot_idx]["encoded_exp_matrix"]
            self.neighborhoods[spot_idx]["encoded_exp_matrix"] = data_normalization(encoded_exp_mt, self.min_val, self.max_val)  


    def __getitem__(self, idx):
        """
        An item returns a dictionary with the following keys and values:
            - 'spot_id': string that corresponds to the id name of the main spot in the current sample.
            - 'exp_matrix': expression matrix with values from adata.layer[pred_layer]. Not encoded, nor normalized.
            - 'encoded_exp_matrix': same as "exp_matrix" but encoded and then normalized with the min and max values of whole data split.
            - 'patches': 

        """
        item = self.neighborhoods[str(idx)]

        return item

    def __len__(self):
        return len(self.adata)




class SpaREDData():
    def __init__(self, args, autoencoder, image_encoder_model, image_transforms):
        super().__init__()

        self.args = args
        self.dataset_name = args.dataset
        self.batch_size = args.batch_size
        self.num_neighs = args.num_neighs
        self.prediction_layer = args.pred_layer
        self.autoencoder = autoencoder
        self.image_encoder_model = image_encoder_model
        self.image_transforms = image_transforms

        # Load datasets (1024-gene adata, and original SpaRED adata)
        self.load_data()
        # Get average values for 1024-genes adata
        # Always the model work with this layer
        self.average_vals = torch.tensor(self.full_adata.var[f"c_t_log1p_avg_exp"]).unsqueeze(0)
        # Set split data and create data modules
        self.setup()
        self.train_data = stLDMDataset(self.args, self.spared_train, "train",
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        self.val_data = stLDMDataset(self.args, self.spared_val, "val", 
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        self.test_data = stLDMDataset(self.args, self.spared_test, "test", 
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        

    def load_data(self):
        self.adata_path = f"datasets/1024/{self.dataset_name}/adata.h5ad"
        self.full_adata = ad.read_h5ad(self.adata_path)
        # Check if test split is available
        self.test_data_available = True if 'test' in self.full_adata.obs['split'].unique() else False
        # Get number of genes in dataset
        self.n_genes = self.full_adata.n_vars

        # Load original SpaRED adata 
        self.original_adata_path = f"datasets/original/{self.dataset_name}/adata.h5ad"
        self.original_full_adata = ad.read_h5ad(self.original_adata_path)

    def setup(self):
        # Assign train/val/test datasets for use in dataloaders
        self.spared_train = self.full_adata[self.full_adata.obs["split"]=="train"] 
        self.spared_val = self.full_adata[self.full_adata.obs["split"]=="val"]
        self.spared_test = self.full_adata[self.full_adata.obs["split"]=="test"] if self.test_data_available else  self.full_adata[self.full_adata.obs["split"]=="val"]

    def train_dataloader(self):
        # item is a dictionary with keys ['spot_id', 'exp_matrix', 'exp_mask', 'encoded_exp_matrix', 'condition_matrix', 'condition_mask']
        # keys used during train: ['encoded_exp_matrix', 'condition_matrix', 'condition_mask']
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False) #, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)
    

