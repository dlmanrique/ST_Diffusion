CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset villacampa_lung_organoid --prediction_layer c_t_deltas --depth 6
CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset 10xgenomic_mouse_brain_sagittal_posterior --prediction_layer c_t_deltas --depth 6
CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset mirzazadeh_mouse_bone --prediction_layer c_t_deltas --depth 6
CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset villacampa_mouse_brain --prediction_layer c_t_deltas --depth 6
CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset mirzazadeh_human_small_intestine --prediction_layer c_t_deltas --depth 6