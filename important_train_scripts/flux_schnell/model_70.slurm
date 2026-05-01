#!/bin/bash
#SBATCH --job-name=flux_schnell
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/flux_schnell/model_70.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/flux_schnell/model_70.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=80gb
#SBATCH --export=NONE
#SBATCH --gres=gpu:H200:1

# >>> Conda setup >>>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train
# >>> Conda setup >>>


accelerate launch --mixed_precision bf16 --num_processes=1 --num_machines=1 --num_cpu_threads_per_process 1 /home/hd/hd_hd/hd_om233/SVD/flux/flux_schnell_train_var_guidance_it.py \
--ae /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/ae/ae.safetensors \
--apply_t5_attn_mask \
--blocks_to_swap 0 \
--clip_skip 1 \
--clip_l /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/clip/model.safetensors \
--dataset_config /home/hd/hd_hd/hd_om233/flux/dataset_config/laion2M_flux_compress_doubel_blocks.toml \
--discrete_flow_shift 3 \
--fp8_base \
--full_bf16 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing \
--guidance_scale 1.0 \
--highvram \
--huber_c 0.1 \
--huber_schedule "snr" \
--learning_rate 1.8e-6 \
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_70 \
--log_tracker_name tensorboard \
--log_with tensorboard \
--loss_type "l2" \
--lr_scheduler constant_with_warmup \
--lr_warmup_steps 4240 \
--max_data_loader_n_workers 6 \
--max_timestep 1000 \
--max_train_epochs 4 \
--mixed_precision bf16 \
--model_prediction_type raw \
--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
--optimizer_type adafactor \
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_70 \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors  \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors  \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_80/test-step00020000.safetensors  \
--save_every_n_steps 5000 \
--save_model_as safetensors \
--save_precision bf16 \
--sdpa \
--seed 42 \
--t5xxl /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/t5xxl/model.safetensors \
--t5xxl_max_token_length 512 \
--timestep_sampling "uniform" \
--sample_every_n_steps 2000 \
--sample_at_first \
--sample_prompts /home/hd/hd_hd/hd_om233/flux/dataset_config/sample_text_prompts.txt \
--sample_sampler euler_a \
--KD_flag \
--feature_loss_flag \
--final_loss_flag \
--original_loss_flag \
--single_feature_loss_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 \
--double_feature_loss_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
--compress_double_blocks 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0  \
--compress_single_blocks 0 33 30 2 5 13 22 1 15 18 8 19 16 17 12 14 10 6 9 26 \
--compress_double_blocks_new 4 15 7 17 6 10 11 1 5 16 14 8 12 9 13 0 3 18 \
--compress_single_blocks_new 27 17 20 22 33 34 28 32 29 16 23 15 \
--double_flag_img_attn \
--double_flag_txt_attn \
--double_flag_img_mlp \
--double_flag_txt_mlp \
--double_flag_img_mod \
--double_flag_txt_mod \
--double_flag_txt_proj \
--double_flag_img_proj \
--single_flag_mod \
--single_flag_mlp2 \
--single_flag_mlp \
--single_flag_attn \
--single_comp_mod 0.1004 0.092 0.0837 0.0754 0.0671 0.1599 0.1638 0.1166 0.0339 0.1591 0.0172 0.0089 0.143 0.1312 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 \
--single_comp_mlp2 0.1004 0.092 0.0837 0.0754 0.0671 0.1599 0.1638 0.1166 0.0339 0.1591 0.0172 0.0089 0.143 0.1312 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 \
--single_comp_attn 0.1004 0.092 0.0837 0.0754 0.0671 0.1599 0.1638 0.1166 0.0339 0.1591 0.0172 0.0089 0.143 0.1312 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 \
--single_comp_mlp 0.1004 0.092 0.0837 0.0754 0.0671 0.1599 0.1638 0.1166 0.0339 0.1591 0.0172 0.0089 0.143 0.1312 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 \
--double_comp_img_mod 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_img_mlp 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_img_attn 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_txt_mod 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_txt_mlp 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_txt_attn 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_txt_proj 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
--double_comp_img_proj 0.3929 0.3907 0.3522 0.3544 0.3613 0.3828 0.343 0.365 0.3731 0.2981 0.3108 0.294 0.3526 0.3513 0.3142 0.3232 0.2958 0.3315 \
    --single_comp_mod_new 0.1658 0.2564 0.1333 0.2662 0.1934 0.1008 0.09 0.0792 0.0683 0.1923 0.0467 0.0685 \
  --single_comp_mlp2_new 0.1658 0.2564 0.1333 0.2662 0.1934 0.1008 0.09 0.0792 0.0683 0.1923 0.0467 0.0685 \
  --single_comp_attn_new 0.1658 0.2564 0.1333 0.2662 0.1934 0.1008 0.09 0.0792 0.0683 0.1923 0.0467 0.0685 \
  --single_comp_mlp_new 0.1658 0.2564 0.1333 0.2662 0.1934 0.1008 0.09 0.0792 0.0683 0.1923 0.0467 0.0685 \
  --double_comp_img_mod_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_img_mlp_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_img_attn_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_txt_mod_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_txt_mlp_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_txt_attn_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_txt_proj_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
  --double_comp_img_proj_new 0.5411 0.5658 0.5371 0.5641 0.5597 0.5441 0.5097 0.4889 0.5228 0.5445 0.5246 0.5253 0.494 0.4648 0.5133 0.4568 0.4685 0.4069 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1