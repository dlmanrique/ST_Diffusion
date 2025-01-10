from uni import get_encoder
import scanpy as sc
import os
import torch
import argparse
from tqdm import tqdm

def extract_features(args):
    adata = sc.read_h5ad(os.path.join('datasets', args.dataset, 'adata.h5ad'))
    # Divide in different sets
    splits = adata.obs["split"].unique().tolist()

    # Initialize the UNI model
    model, _= get_encoder(enc_name='uni', device='cuda')

    # Create folder if neccesary
    output_dir = os.path.join('UNI', args.dataset)
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
                features = model(images)
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


