#!/bin/bash
#SBATCH --job-name=var_guidance_it_model_2_Lite_comp
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/flux_train_var_guidance_it/model_2_Lite_comp.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/flux_train_var_guidance_it/model_2_Lite_comp.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
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
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_2_Lite_comp_2 \
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
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_2_Lite_comp_2 \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_80/test-step00020000.safetensors \
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
--compress_double_blocks 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 \
--compress_single_blocks 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 \
--compress_double_blocks_new 0 11 9 12 10 7 14 6 4 18 1 15 3 8 17 16 5 13 \
--compress_single_blocks_new 16 23 22 14 26 31 17 20 21 6 28 11 1 37 24 27 19 4 3 7 12 25 \
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
--single_comp_mod 0.515 0.485 0.41 0.395 0.275 0.26 0.2 0.185 0.155 0.14 0.125 0.11 0.095 0.08 0.065 0.05 0.035 \
--single_comp_mlp2 0.515 0.485 0.41 0.395 0.275 0.26 0.2 0.185 0.155 0.14 0.125 0.11 0.095 0.08 0.065 0.05 0.035 \
--single_comp_attn 0.515 0.485 0.41 0.395 0.275 0.26 0.2 0.185 0.155 0.14 0.125 0.11 0.095 0.08 0.065 0.05 0.035 \
--single_comp_mlp 0.515 0.485 0.41 0.395 0.275 0.26 0.2 0.185 0.155 0.14 0.125 0.11 0.095 0.08 0.065 0.05 0.035 \
--double_comp_img_mod 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_img_mlp 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_img_attn 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_txt_mod 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_txt_mlp 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_txt_attn 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_txt_proj 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
--double_comp_img_proj 0.53 0.5 0.47 0.455 0.44 0.425 0.38 0.365 0.35 0.335 0.32 0.305 0.29 0.245 0.23 0.215 0.17 \
    --single_comp_mod_new 0.1699 0.1623 0.1546 0.1393 0.2011 0.124 0.1164 0.1087 0.1011 0.1251 0.1726 0.0781 0.2564 0.1097 0.0552 0.0475 0.1599 0.0322 0.1319 0.0169 0.0737 0.0016 \
  --single_comp_mlp2_new 0.1699 0.1623 0.1546 0.1393 0.2011 0.124 0.1164 0.1087 0.1011 0.1251 0.1726 0.0781 0.2564 0.1097 0.0552 0.0475 0.1599 0.0322 0.1319 0.0169 0.0737 0.0016 \
  --single_comp_attn_new 0.1699 0.1623 0.1546 0.1393 0.2011 0.124 0.1164 0.1087 0.1011 0.1251 0.1726 0.0781 0.2564 0.1097 0.0552 0.0475 0.1599 0.0322 0.1319 0.0169 0.0737 0.0016 \
  --single_comp_mlp_new 0.1699 0.1623 0.1546 0.1393 0.2011 0.124 0.1164 0.1087 0.1011 0.1251 0.1726 0.0781 0.2564 0.1097 0.0552 0.0475 0.1599 0.0322 0.1319 0.0169 0.0737 0.0016 \
  --double_comp_img_mod_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_img_mlp_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_img_attn_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_txt_mod_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_txt_mlp_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_txt_attn_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_txt_proj_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
  --double_comp_img_proj_new 0.3 0.4657 0.4385 0.4433 0.4813 0.498 0.5152 0.5215 0.471 0.4887 0.3555 0.5138 0.5566 0.5403 0.5601 0.5926 0.5641 0.5991 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1


