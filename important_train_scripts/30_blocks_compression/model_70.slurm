#!/bin/bash
#SBATCH --job-name=30_blocks_compression
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/paper_var_guidance_it/30_blocks_compression/model_70.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/paper_var_guidance_it/30_blocks_compression/model_70.txt

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


accelerate launch --mixed_precision bf16 --num_processes=1 --num_machines=1 --num_cpu_threads_per_process 1 /home/hd/hd_hd/hd_om233/SVD/flux/flux_train_var_guidance_it.py \
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
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_70 \
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
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_70 \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_80/test-step00020000.safetensors \
--save_every_n_steps 5000 \
--save_model_as safetensors \
--save_precision bf16 \
--sdpa \
--seed 42 \
--t5xxl /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/t5xxl/model.safetensors \
--t5xxl_max_token_length 512 \
--timestep_sampling "sigmoid" \
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
--compress_single_blocks 0 33 30 2 5 13 22 1 15 18 8 19 16 6 26 7 21 27 12 24 3 \
--compress_double_blocks_new 0 16 6 8 13 10 17 18 15 11 3 4 1 7 12 14 5 9 \
--compress_single_blocks_new 16 26 15 18 1 2 6 30 19 17 37 33 \
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
--single_comp_mod 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
--single_comp_mlp2 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
--single_comp_attn 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
--single_comp_mlp 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
--double_comp_img_mod 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_img_mlp 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_img_attn 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_txt_mod 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_txt_mlp 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_txt_attn 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_txt_proj 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
--double_comp_img_proj 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
    --single_comp_mod_new 0.2968 0.2812 0.2904 0.1802 0.1879 0.2098 0.2571 0.3046 0.133 0.1184 0.1117 0.2799 \
  --single_comp_mlp2_new 0.2968 0.2812 0.2904 0.1802 0.1879 0.2098 0.2571 0.3046 0.133 0.1184 0.1117 0.2799 \
  --single_comp_attn_new 0.2968 0.2812 0.2904 0.1802 0.1879 0.2098 0.2571 0.3046 0.133 0.1184 0.1117 0.2799 \
  --single_comp_mlp_new 0.2968 0.2812 0.2904 0.1802 0.1879 0.2098 0.2571 0.3046 0.133 0.1184 0.1117 0.2799 \
  --double_comp_img_mod_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_img_mlp_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_img_attn_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_txt_mod_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_txt_mlp_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_txt_attn_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_txt_proj_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
  --double_comp_img_proj_new 0.5321 0.5632 0.5573 0.569 0.5492 0.5029 0.529 0.4832 0.4978 0.5029 0.4923 0.4961 0.4136 0.4886 0.4555 0.4747 0.4991 0.4405 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1