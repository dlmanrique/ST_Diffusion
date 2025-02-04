CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --num_neighs -1 --image_encoder shufflenet --dataset villacampa_mouse_brain --train False --test False --debbug_wandb True
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --num_neighs -1 --image_encoder shufflenet --dataset mirzazadeh_human_small_intestine --train False --test False --debbug_wandb True
CUDA_VISIBLE_DEVICES=2 python train_image_to_genes.py --num_neighs -1 --image_encoder shufflenet --dataset vicari_human_striatium --train False --test False --debbug_wandb True
