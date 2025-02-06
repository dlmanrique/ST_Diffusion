"""from uni import get_encoder
import scanpy as sc
import os
import torch
import argparse
from tqdm import tqdm
import timm
from conch.open_clip_custom import create_model_from_pretrained



def extract_features(args):
    adata = sc.read_h5ad(os.path.join('datasets', args.dataset, 'adata.h5ad'))
    # Divide in different sets
    splits = adata.obs["split"].unique().tolist()
    breakpoint()
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
                                                  hf_auth_token="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")
    model = model.to('cuda')
    # Create folder if neccesary
    output_dir = os.path.join('CONCH', args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Dataset: {args.dataset}')
    
    for split in splits:
        adata_split_divided = adata[adata.obs["split"]==split]
        flat_patches = adata_split_divided.obsm[f'patches_scale_1.0']
        patches = flat_patches.reshape((-1, 224, 224, 3))
        patches = torch.tensor(patches)
        patches = patches.to(dtype=torch.float32)
        batches = torch.split(patches, args.batch)
        features_list = []
        for images in tqdm(batches):
            images = images.to('cuda')
            images = images.permute(0, 3, 1, 2)

            with torch.inference_mode():
                features = model.encode_image(images, proj_contrast=False, normalize=False)
                features_list.append(features)
        
        all_features = torch.cat(features_list, dim=0)
        torch.save(all_features, os.path.join(output_dir, f'{split}.pt'))
        print(f'Spot/Patches in {split}: {all_features.shape[0]}')



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Code for UNI features extraction')
    parser.add_argument('--dataset', type=str, default='villacampa_lung_organoid',  help='Dataset to use.')
    parser.add_argument('--batch', type=int, default=4096, help='Batch size for UNI inference')
    args = parser.parse_args()

    extract_features(args)"""

"""import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
breakpoint()
login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "../assets/ckpts/uni2-h/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
model = timm.create_model(
    pretrained=False, **timm_kwargs
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()"""

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
import os
from tqdm import tqdm
import scanpy as sc
import os
import torch
import argparse


# need to specify MLP layer and activation function for proper init
model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


def extract_features(args):
    adata = sc.read_h5ad(os.path.join('datasets', args.dataset, 'adata.h5ad'))
    # Divide in different sets
    splits = adata.obs["split"].unique().tolist()
    breakpoint()
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.eval()

    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model = model.to('cuda')
    # Create folder if neccesary
    output_dir = os.path.join('CONCH', args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Dataset: {args.dataset}')
    
    for split in splits:
        adata_split_divided = adata[adata.obs["split"]==split]
        flat_patches = adata_split_divided.obsm[f'patches_scale_1.0']
        patches = flat_patches.reshape((-1, 224, 224, 3))
        patches = torch.tensor(patches)
        patches = patches.to(dtype=torch.float32)
        batches = torch.split(patches, args.batch)
        features_list = []
        for images in tqdm(batches):
            images = images.to('cuda')
            images = images.permute(0, 3, 1, 2)

            with torch.inference_mode():
                features = model.encode_image(images, proj_contrast=False, normalize=False)
                features_list.append(features)
        
        all_features = torch.cat(features_list, dim=0)
        torch.save(all_features, os.path.join(output_dir, f'{split}.pt'))
        print(f'Spot/Patches in {split}: {all_features.shape[0]}')



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Code for UNI features extraction')
    parser.add_argument('--dataset', type=str, default='villacampa_lung_organoid',  help='Dataset to use.')
    parser.add_argument('--batch', type=int, default=4096, help='Batch size for UNI inference')
    args = parser.parse_args()

    extract_features(args)




