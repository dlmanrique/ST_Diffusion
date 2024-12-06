CUDA_VISIBLE_DEVICES=6 python main_2D.py --dataset vicari_human_striatium --prediction_layer c_t_deltas --batch_size 512
CUDA_VISIBLE_DEVICES=6 python main_2D.py --dataset erickson_human_prostate_cancer_p1 --prediction_layer c_t_deltas --batch_size 512
CUDA_VISIBLE_DEVICES=6 python main_2D.py --dataset mirzazadeh_mouse_brain --prediction_layer c_t_deltas --batch_size 512 --depth 6
CUDA_VISIBLE_DEVICES=6 python main_2D.py --dataset mirzazadeh_mouse_brain --prediction_layer c_t_deltas --batch_size 512