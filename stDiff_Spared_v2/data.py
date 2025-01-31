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
    def __init__(self, args, adata, split_name, spared_genes_names, genes_autoencoder, image_encoder, image_transforms):
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
        self.spared_genes_ids = spared_genes_names
        # Based on args, select the desired adata, original or 1024
        # If args.gene_autoencoder == None -> then the experiment doesn't involves 1024 adata
        self.adata = adata
        self.model_autoencoder = genes_autoencoder
        self.image_encoder = image_encoder

        if args.image_encoder == 'uni':
            self.image_transforms = transforms.Compose(image_transforms.transforms[-2:])
        else:
            self.image_transforms = image_transforms

        self.great_mask = self.build_general_mask()

        # Get original expression matrix based on selected prediction layer.
        self.expression_mtx = torch.tensor(self.adata.layers[self.pred_layer])
        #FIXME: esto no sirve para datasets de 32?
        self.all_st_data_shape = (self.expression_mtx.shape[0], 128) #-> DiT input is always 128
        # Calculate patch_features
        self.calculate_patch_embeddings()
        #Normalize patch_feature in a range (-1,1) 
        self.normalize_image_features()
        # Build and save each spot's neighborhood, and the min and max val of the data split
        self.min_val, self.max_val = np.inf, -np.inf 
        
        if args.num_neighs == -1:
            print(f'Construct {self.split_name} dataloader by spots using gene autoencoder: {self.args.gene_autoencoder} and  image encoder: {self.args.image_encoder}')
            self.spot_data = self.build_spot_data()
            self.DiT_input_dim = self.spot_data['0']['encoded_spot_exp'].shape # De aca tengo (128)
            self.image_features_dim = self.spot_data['0']['patches'].shape[0] # De aca tengo el 1024 de UNI
        else:
            print(f'Construct {self.split_name} dataloader by matrix using gene autoencoder: {self.args.gene_autoencoder} and image encoder: {self.args.image_encoder}')
            # Process to get neighboors
            self.adj_mat = None
            self.get_adjacency(self.args.num_neighs)
            self.neighborhoods = self.build_neighborhoods()
            self.DiT_input_dim = self.neighborhoods['0']['encoded_exp_matrix'].squeeze().shape   # De aca tengo (7,128)     
            self.image_features_dim = self.neighborhoods['0']['patches'].shape[-1] # De aca tengo el 1024 para UNI
            self.all_st_data_shape = [self.all_st_data_shape[0], self.DiT_input_dim[0], self.DiT_input_dim[1]]

        
        # Normalize data in every case
        if self.args.gene_autoencoder:
            print('Normalize input of DiT which is encoded gene expression')
            # The gene autoencoder exists
            self.normalize_full_encoded_data()
        else:
            print('Normalize input of DiT which is raw gene expression')
            # Use of raw st data 
            self.normalize_full_raw_data()



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

        features_path = os.path.join('Image_features', self.args.image_encoder, self.args.dataset, f'{self.split_name}.pt')
        if os.path.exists(features_path):
            print(f"The image features already exists in: {features_path}")
            self.patch_features = torch.load(features_path)

        else:
            print(f"Calculating image features for {self.args.dataset}/{self.split_name} using {self.args.image_encoder} model")          
            flat_patches = self.adata.obsm[f'patches_scale_1.0']
            patches = flat_patches.reshape((-1, 224, 224, 3))
            patches = np.array(patches) / 255
            patches_dataset = TransformTensorDataset(patches, self.image_transforms)
            dataloader = DataLoader(patches_dataset, batch_size=512, shuffle=False)
            patch_features = []

            for batch in tqdm(dataloader):
                batch = batch.to('cuda').float()                                  
                batch_output = self.image_encoder(batch)    
                patch_features.append(batch_output)

            self.patch_features = torch.cat(patch_features, dim=0)

            os.makedirs(os.path.join('Image_features', self.args.image_encoder, self.args.dataset), exist_ok=True)
            save_path = os.path.join('Image_features',  self.args.image_encoder, self.args.dataset, f'{self.split_name}.pt')
            torch.save(self.patch_features, save_path)
            print(f'Image features saved in: {save_path}')

    def build_general_mask(self):
        """
        Combines the mask from the adata.layers section to hide the values that were originally 
        artificially completed (i. e. completion through adaptive median or SpaCKLE), with a mask that 
        hides the columns of the genes that are not part of the "genes_to_keep" array. 
        
        This general mask is needed to compute post-decoding metrics.
        """

        # Build bool array, with True in the idxs corresponding to genes present in the SpaRED set
        mask_to_keep = self.adata.var['gene_ids'].isin(self.spared_genes_ids)
        # Get original mask (False in the values that were previously completed using SpaCKLE)
        original_mask = torch.tensor(self.adata.layers["mask"])
        new_mask = original_mask.clone()
        # Set columns corresponding to genes not in 'genes_to_keep' to False
        new_mask[:, ~mask_to_keep] = False
        # Add the modified mask to adata and return it
        self.adata.layers["general_mask"] = new_mask.cpu().numpy()

        return new_mask


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
            nn_exp = self.expression_mtx[self.adj_mat[idx,:]==1.]
            exp_matrix = torch.cat((spot_exp, nn_exp), dim=0).type('torch.FloatTensor')
            exp_matrix = exp_matrix.unsqueeze(0)
            nn_indices = torch.nonzero(self.adj_mat[idx,:], as_tuple=False)
            nn_indices = nn_indices.squeeze(1).tolist() #lista de 6 vecinos
            nn_indices = [idx] + nn_indices #Spot central + vecinos

            if self.model_autoencoder:
                # Encode neighborhood
                self.model_autoencoder.eval()
                with torch.no_grad():
                    encoded_exp_matrix = self.model_autoencoder.encoder(exp_matrix.to("cuda"))
                    
                # Get median imputation mask for idx spot and its nn
                spot_mask = self.great_mask[idx].unsqueeze(dim=0) #size 1xgenes(1024)
                nn_mask = self.great_mask[self.adj_mat[:,idx]==1.] #size 6xgenes(1024)
                great_mask = torch.cat((spot_mask, nn_mask), dim=0)

                all_neighborhoods[str(idx)] = {"spot_id": spot_name, 
                                            "exp_matrix": exp_matrix, 
                                            "encoded_exp_matrix": encoded_exp_matrix.squeeze(0).detach().cpu(),
                                            'patches': self.patch_features[nn_indices,:],
                                            "exp_mask": great_mask}
            else:
                all_neighborhoods[str(idx)] = {"spot_id": spot_name, 
                                            "exp_matrix": exp_matrix.squeeze(), 
                                            "encoded_exp_matrix": exp_matrix,
                                            'patches': self.patch_features[nn_indices,:]}

                # This variable is just for min and max calculation
                encoded_exp_matrix = exp_matrix

            # Set min and max values of the data split
            if encoded_exp_matrix.min().item() < self.min_val:
                self.min_val = encoded_exp_matrix.min().item()
            if encoded_exp_matrix.max().item() > self.max_val:
                self.max_val = encoded_exp_matrix.max().item()    
        return all_neighborhoods
    

    def build_spot_data(self):
        """
        Creates a dictionary of dictionaries, where each element/key corresponds to an individual spot in the
        adata, and each inner-dictionary/value corresponds to its own information.
        """
        all_spots_data = {}

        for idx, spot_name in enumerate(tqdm(self.adata.obs["unique_id"].unique())):
            # Get gt expression for idx spot and its nn
            spot_exp = self.expression_mtx[idx].unsqueeze(dim=0).unsqueeze(dim=0).type('torch.FloatTensor')
            
            if self.model_autoencoder:
                self.model_autoencoder.eval()
                with torch.no_grad():
                    encoded_spot_exp = self.model_autoencoder.encoder(spot_exp.to("cuda"))
                
                #Get median imputation mask for idx spot and its nn
                spot_mask = self.great_mask[idx].unsqueeze(dim=0) #size 1xgenes(1024)

                all_spots_data[str(idx)] = {"spot_id": spot_name, 
                                            "spot_expression": spot_exp.squeeze(), 
                                            "encoded_spot_exp": encoded_spot_exp.squeeze(),
                                            'patches': self.patch_features[idx,:],
                                            "exp_mask": spot_mask}
                
            else:
                all_spots_data[str(idx)] = {"spot_id": spot_name, 
                                            "spot_expression": spot_exp.squeeze(), 
                                            "encoded_spot_exp": spot_exp.squeeze(),
                                            'patches': self.patch_features[idx,:]}
                
                # This variable is just for min and max calculation
                encoded_spot_exp = spot_exp
                
            # Set min and max values of the data split
            if encoded_spot_exp.min().item() < self.min_val:
                self.min_val = encoded_spot_exp.min().item()
            if encoded_spot_exp.max().item() > self.max_val:
                self.max_val = encoded_spot_exp.max().item()  


        return all_spots_data
    
    def normalize_full_encoded_data(self):
        """
        Calls for the normalization function from utils to normalize all neighborhoods/samples
        based on the min and max values of the complete data split.
        """
        encoded_data_key = 'encoded_spot_exp' if self.args.num_neighs == -1 else 'encoded_exp_matrix'
        
        if encoded_data_key == 'encoded_exp_matrix':
            for spot_idx in self.neighborhoods.keys():
                encoded_exp_mt = self.neighborhoods[spot_idx][encoded_data_key]
                self.neighborhoods[spot_idx][encoded_data_key] = data_normalization(encoded_exp_mt, self.min_val, self.max_val)  
        else:
            for spot_idx in self.spot_data.keys():
                encoded_exp_spot = self.spot_data[spot_idx][encoded_data_key]
                self.spot_data[spot_idx][encoded_data_key] = data_normalization(encoded_exp_spot, self.min_val, self.max_val) 


    def normalize_full_raw_data(self):
        encoded_data_key = 'spot_expression' if self.args.num_neighs == -1 else 'exp_matrix'

        if encoded_data_key == 'exp_matrix':
            for spot_idx in self.neighborhoods.keys():
                encoded_exp_mt = self.neighborhoods[spot_idx][encoded_data_key]
                self.neighborhoods[spot_idx][encoded_data_key] = data_normalization(encoded_exp_mt, self.min_val, self.max_val)  
        else:
            for spot_idx in self.spot_data.keys():
                encoded_exp_spot = self.spot_data[spot_idx][encoded_data_key]
                self.spot_data[spot_idx][encoded_data_key] = data_normalization(encoded_exp_spot, self.min_val, self.max_val) 

    def normalize_image_features(self):
        features = self.patch_features
        self.patch_features = data_normalization(features, features.min(), features.max())


    def __getitem__(self, idx):
        """
        An item returns a dictionary with the following keys and values:
            - 'spot_id': string that corresponds to the id name of the main spot in the current sample.
            - 'exp_matrix': expression matrix with values from adata.layer[pred_layer]. Not encoded, nor normalized.
            - 'encoded_exp_matrix': same as "exp_matrix" but encoded and then normalized with the min and max values of whole data split.
            - 'patches': 

        """
        if self.args.num_neighs == -1:
            item = self.spot_data[str(idx)]
        
        else:
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
        # Sort genes in adatas
        self.sort_adatas()

        # Get average values for 1024-genes adata or 128-genes adata
        # Always work with this layer
        if self.autoencoder:
            self.average_vals = torch.tensor(self.full_adata.var[f"c_t_log1p_avg_exp"]).unsqueeze(0)
        else:
            self.average_vals = torch.tensor(self.original_full_adata.var[f"c_t_log1p_avg_exp"]).unsqueeze(0)

        # Set split data and create data modules
        self.setup()
        self.train_data = stLDMDataset(self.args, self.spared_train,  "train", self.spared_genes_array,
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        self.val_data = stLDMDataset(self.args, self.spared_val, "val", self.spared_genes_array,
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        self.test_data = stLDMDataset(self.args, self.spared_test, "test", self.spared_genes_array,
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        self.all_data = stLDMDataset(self.args, self.spared_all, "all", self.spared_genes_array,
                                        self.autoencoder, self.image_encoder_model, self.image_transforms)
        

    def load_data(self):
        self.adata_path = f"datasets/1024/{self.dataset_name}_1024.h5ad"
        self.full_adata = ad.read_h5ad(self.adata_path)
        # Check if test split is available
        self.test_data_available = True if 'test' in self.full_adata.obs['split'].unique() else False
        # Get number of genes in dataset
        self.n_genes = self.full_adata.n_vars

        # Load original SpaRED adata 
        self.original_adata_path = f"datasets/original/{self.dataset_name}.h5ad"
        self.original_full_adata = ad.read_h5ad(self.original_adata_path)

        # Get array of genes of interest
        self.spared_genes_array = self.original_full_adata.var['gene_ids'].unique()
        self.gene_comprobation = torch.tensor(np.isin(self.full_adata.var['gene_ids'].unique(), self.spared_genes_array))
        print(f"Genes from full adata that are part of the SpaRED list: {self.gene_comprobation.sum()}")


    def setup(self):
        # Assign train/val/test datasets for use in dataloaders
        # Use the 1024 adatas
        self.spared_train = self.full_adata[self.full_adata.obs["split"]=="train"] 
        self.spared_val = self.full_adata[self.full_adata.obs["split"]=="val"]
        self.spared_test = self.full_adata[self.full_adata.obs["split"]=="test"] if self.test_data_available else  self.full_adata[self.full_adata.obs["split"]=="val"]
        self.spared_all = self.full_adata

        # Original adatas (128 genes)
        if self.args.gene_autoencoder is None:
            self.spared_train = self.original_full_adata[self.original_full_adata.obs["split"]=="train"] 
            self.spared_val = self.original_full_adata[self.original_full_adata.obs["split"]=="val"]
            self.spared_test = self.original_full_adata[self.original_full_adata.obs["split"]=="test"] if self.test_data_available else  self.original_full_adata[self.original_full_adata.obs["split"]=="val"]
            self.spared_all = self.original_full_adata


    def sort_adatas(self):
        self.full_adata.var.reset_index(drop=True, inplace=True)
        self.original_full_adata.var.reset_index(drop=True, inplace=True)

        adata_1024_sorted = self.full_adata.copy()
        adata_original_sorted = self.original_full_adata.copy()
        
        # Sort genes by index
        adata_1024_sorted.var["original_index"] = adata_1024_sorted.var.index
        adata_1024_sorted.var = adata_1024_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

        adata_original_sorted.var["original_index"] = adata_original_sorted.var.index
        adata_original_sorted.var = adata_original_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

        #Get indices
        sorted_indices_1024 = adata_1024_sorted.var["original_index"].to_numpy()
        sorted_indices_1024 = [int(idx) for idx in sorted_indices_1024]
        
        sorted_indices_original = adata_original_sorted.var["original_index"].to_numpy()
        sorted_indices_original = [int(idx) for idx in sorted_indices_original]

        # Reorder all layers to match the new gene order
        for layer in self.full_adata.layers.keys():
            adata_1024_sorted.layers[layer] = self.full_adata.layers[layer][:, sorted_indices_1024]

        for layer in self.original_full_adata.layers.keys():
            adata_original_sorted.layers[layer] = self.original_full_adata.layers[layer][:, sorted_indices_original]

        self.full_adata = adata_1024_sorted
        self.original_full_adata = adata_original_sorted


    def train_dataloader(self):
        # item is a dictionary with keys ['spot_id', 'exp_matrix', 'exp_mask', 'encoded_exp_matrix', 'condition_matrix', 'condition_mask']
        # keys used during train: ['encoded_exp_matrix', 'condition_matrix', 'condition_mask']
        generator = torch.Generator(device='cuda')
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False, generator=generator) #, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)

    def all_dataloader(self):
        return DataLoader(self.all_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
    

