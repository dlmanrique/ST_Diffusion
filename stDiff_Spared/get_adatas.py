import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
from Transformer_encoder_decoder import *
from Transformer_simple import Transformer
from spared.datasets import get_dataset
from utils import *
import glob


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
current_device = torch.cuda.current_device()

#argparse
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args) #Not uses, maybe later usage

path_list = glob.glob("/home/dvegaa/ST_Diffusion/stDiff_Spared/spared_stdiff/processed_data/**/*.h5ad", recursive=True)

for path in path_list:
    if args.dataset in path:
        if "raw" in path:
            path_dataset = path
        
dataset = get_dataset(args.dataset)
adata = get_new_adatas(path_dataset, args)

dataset = get_dataset(args.dataset)
adata_128 = dataset.adata 
num_genes = adata_128.shape[1]
#adata = ad.read_h5ad(f'/home/dvegaa/ST_Diffusion/stDiff_Spared/adata_1024/{args.dataset}_1024.h5ad')
genes_evaluate = []
genes_128 = adata_128.var["gene_ids"].unique().tolist()
genes_1024 = adata.var["gene_ids"].unique().tolist()

#Get updated 1024 adata
adata = get_1204_adata(adata=adata_128, adata_1024=adata, genes=genes_128)
adata.write(f'/home/dvegaa/ST_Diffusion/stDiff_Spared/adata_1024/{args.dataset}_1024.h5ad')