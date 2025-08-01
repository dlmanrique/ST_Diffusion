import torch
from spared.datasets import get_dataset
from utils import *
import glob
import json

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
current_device = torch.cuda.current_device()

#argparse
parser = get_main_parser()
args = parser.parse_args()
#adata = get_dataset(args.dataset, visualize=False)

path_dataset = f"/home/dvegaa/ST_Diffusion/stDiff_Spared/processed_data/{args.dataset}/adata_raw.h5ad"

dir_path = os.path.dirname(path_dataset)
json_path = os.path.join(dir_path, "parameters.json")
with open(json_path, "r") as file:  # Replace with your actual file path
    data = json.load(file)
param_dict = data["param_dict"]

adata = get_complete_adatas(path_dataset, param_dict, args)

path_adata_original = f"/home/dvegaa/ST_Diffusion/stDiff_Spared/processed_data/{args.dataset}/adata.h5ad"
original_adata = ad.read_h5ad(path_adata_original)
genes = original_adata.var_names
adata_128 = adata[genes].copy()
save_path = "/media/SSD4/dvegaa/adatas_10000"

# Asegurar que el directorio existe
os.makedirs(save_path, exist_ok=True)

# Guardar el objeto AnnData en formato .h5ad
adata.write(os.path.join(save_path, f"adata_{args.dataset}.h5ad"))