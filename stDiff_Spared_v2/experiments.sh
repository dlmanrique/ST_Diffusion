CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --dataset villacampa_lung_organoid --image_encoder virchow
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --dataset 10xgenomic_mouse_brain_sagittal_posterior --image_encoder virchow
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --dataset mirzazadeh_mouse_bone --image_encoder virchow
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --dataset villacampa_lung_organoid --image_encoder virchow2
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --dataset 10xgenomic_mouse_brain_sagittal_posterior --image_encoder virchow2
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --dataset mirzazadeh_mouse_bone --image_encoder virchow2