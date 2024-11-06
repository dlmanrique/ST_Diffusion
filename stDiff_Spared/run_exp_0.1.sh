CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset villacampa_lung_organoid --prediction_layer c_t_deltas --normalization_type 0-1
CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset villacampa_mouse_brain --prediction_layer c_t_deltas --normalization_type 0-1
CUDA_VISIBLE_DEVICES=5 python main_2D.py --dataset erickson_human_prostate_cancer_p1 --prediction_layer c_t_deltas --normalization_type 0-1