#!/bin/bash
#SBATCH --job-name=30_blocks_compression
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/paper_var_guidance_it/30_blocks_compression/model_50.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/paper_var_guidance_it/30_blocks_compression/model_50.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --time=50:00:00
#SBATCH --mem=80gb
#SBATCH --export=NONE
#SBATCH --gres=gpu:H200:2

# >>> Conda setup >>>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train
# >>> Conda setup >>>


accelerate launch --mixed_precision bf16 --num_processes=2 --multi_gpu --num_machines=1 --num_cpu_threads_per_process 1 /home/hd/hd_hd/hd_om233/SVD/flux/flux_train_var_guidance_it.py \
--ae /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/ae/ae.safetensors \
--apply_t5_attn_mask \
--blocks_to_swap 0 \
--clip_skip 1 \
--clip_l /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/clip/model.safetensors \
--dataset_config /home/hd/hd_hd/hd_om233/flux/dataset_config/laion2M_flux_compress_doubel_blocks.toml \
--discrete_flow_shift 3 \
--fp8_base \
--full_bf16 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--guidance_scale 1.0 \
--highvram \
--huber_c 0.1 \
--huber_schedule "snr" \
--learning_rate 1.8e-6 \
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_50 \
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
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_50 \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_60/test-step00020000.safetensors \
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
--compress_single_blocks 0 33 30 2 5 13 22 1 15 18 8 19 16 6 26 7 21 27 12 24 3 17 37 28 14 23 20 31 4 25 \
--compress_double_blocks_new 5 7 0 13 12 18 8 14 15 16 11 6 3 9 \
--compress_single_blocks_new 4 19 31 25 21 5 20 37 10 2 16 28 3 23 1 12 \
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
--single_comp_mod 0.2867 0.2799 0.3046 0.2098 0.0671 0.2152 0.2794 0.1879 0.2904 0.3332 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.2451 0.3386 0.0902 0.0843 0.0784 0.1184 0.1117 0.2479 0.2343 0.2275 0.2002 0.1934 0.1798 0.173 \
--single_comp_mlp2 0.2867 0.2799 0.3046 0.2098 0.0671 0.2152 0.2794 0.1879 0.2904 0.3332 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.2451 0.3386 0.0902 0.0843 0.0784 0.1184 0.1117 0.2479 0.2343 0.2275 0.2002 0.1934 0.1798 0.173 \
--single_comp_attn 0.2867 0.2799 0.3046 0.2098 0.0671 0.2152 0.2794 0.1879 0.2904 0.3332 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.2451 0.3386 0.0902 0.0843 0.0784 0.1184 0.1117 0.2479 0.2343 0.2275 0.2002 0.1934 0.1798 0.173 \
--single_comp_mlp 0.2867 0.2799 0.3046 0.2098 0.0671 0.2152 0.2794 0.1879 0.2904 0.3332 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.2451 0.3386 0.0902 0.0843 0.0784 0.1184 0.1117 0.2479 0.2343 0.2275 0.2002 0.1934 0.1798 0.173 \
--double_comp_img_mod 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_img_mlp 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_img_attn 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_txt_mod 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_txt_mlp 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_txt_attn 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_txt_proj 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
--double_comp_img_proj 0.6947 0.6566 0.671 0.6251 0.665 0.6348 0.636 0.6911 0.6586 0.5972 0.6467 0.6382 0.6667 0.6275 0.6498 0.6287 0.563 0.5321 \
    --single_comp_mod_new 0.4461 0.4713 0.4882 0.463 0.5061 0.3805 0.461 0.3969 0.3162 0.4558 0.5087 0.4709 0.3425 0.4451 0.4126 0.3375 \
  --single_comp_mlp2_new 0.4461 0.4713 0.4882 0.463 0.5061 0.3805 0.461 0.3969 0.3162 0.4558 0.5087 0.4709 0.3425 0.4451 0.4126 0.3375 \
  --single_comp_attn_new 0.4461 0.4713 0.4882 0.463 0.5061 0.3805 0.461 0.3969 0.3162 0.4558 0.5087 0.4709 0.3425 0.4451 0.4126 0.3375 \
  --single_comp_mlp_new 0.4461 0.4713 0.4882 0.463 0.5061 0.3805 0.461 0.3969 0.3162 0.4558 0.5087 0.4709 0.3425 0.4451 0.4126 0.3375 \
  --double_comp_img_mod_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_img_mlp_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_img_attn_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_txt_mod_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_txt_mlp_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_txt_attn_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_txt_proj_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
  --double_comp_img_proj_new 0.8026 0.7828 0.71 0.8093 0.7655 0.7424 0.7647 0.775 0.7565 0.7618 0.7519 0.7735 0.7233 0.7241 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1