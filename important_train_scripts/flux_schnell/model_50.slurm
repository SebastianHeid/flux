#!/bin/bash
#SBATCH --job-name=flux_schnell
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/flux_schnell/model_60.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/flux_schnell/model_60.txt

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
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_50 \
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
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_50 \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors  \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors  \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_60/test-step00020000.safetensors  \
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
--compress_single_blocks 0 33 30 2 5 13 22 1 15 18 8 19 16 17 12 14 10 6 9 26 29 27 7 37 3 4 21 20 35 \
--compress_double_blocks_new 11 1 18 12 6 15 16 13 10 7 9 17 \
--compress_single_blocks_new 21 23 28 6 5 32 2 13 11 14 8 0 34 1 12 10 17 37 \
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
--single_comp_mod 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.3332 0.0172 0.0089 0.2874 0.2986 0.1252 0.3146 0.1014 0.0955 0.0896 0.2886 0.3899 0.3965 0.2403 0.2336 0.2303 0.2203 0.2137 0.3602 0.2037 \
--single_comp_mlp2 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.3332 0.0172 0.0089 0.2874 0.2986 0.1252 0.3146 0.1014 0.0955 0.0896 0.2886 0.3899 0.3965 0.2403 0.2336 0.2303 0.2203 0.2137 0.3602 0.2037 \
--single_comp_attn 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.3332 0.0172 0.0089 0.2874 0.2986 0.1252 0.3146 0.1014 0.0955 0.0896 0.2886 0.3899 0.3965 0.2403 0.2336 0.2303 0.2203 0.2137 0.3602 0.2037 \
--single_comp_mlp 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.3332 0.0172 0.0089 0.2874 0.2986 0.1252 0.3146 0.1014 0.0955 0.0896 0.2886 0.3899 0.3965 0.2403 0.2336 0.2303 0.2203 0.2137 0.3602 0.2037 \
--double_comp_img_mod 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_img_mlp 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_img_attn 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_txt_mod 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_txt_mlp 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_txt_attn 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_txt_proj 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
--double_comp_img_proj 0.6441 0.6465 0.6323 0.6101 0.6397 0.6472 0.6455 0.6597 0.6463 0.5692 0.6318 0.6293 0.6467 0.6295 0.626 0.5995 0.6091 0.5838 \
    --single_comp_mod_new 0.4889 0.3475 0.345 0.4031 0.382 0.3351 0.3829 0.4372 0.3276 0.5374 0.3319 0.3839 0.3102 0.3884 0.3922 0.3645 0.4953 0.4466 \
  --single_comp_mlp2_new 0.4889 0.3475 0.345 0.4031 0.382 0.3351 0.3829 0.4372 0.3276 0.5374 0.3319 0.3839 0.3102 0.3884 0.3922 0.3645 0.4953 0.4466 \
  --single_comp_attn_new 0.4889 0.3475 0.345 0.4031 0.382 0.3351 0.3829 0.4372 0.3276 0.5374 0.3319 0.3839 0.3102 0.3884 0.3922 0.3645 0.4953 0.4466 \
  --single_comp_mlp_new 0.4889 0.3475 0.345 0.4031 0.382 0.3351 0.3829 0.4372 0.3276 0.5374 0.3319 0.3839 0.3102 0.3884 0.3922 0.3645 0.4953 0.4466 \
  --double_comp_img_mod_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_img_mlp_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_img_attn_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_txt_mod_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_txt_mlp_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_txt_attn_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_txt_proj_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
  --double_comp_img_proj_new 0.7541 0.7352 0.7061 0.7454 0.7627 0.7519 0.7518 0.7492 0.7493 0.7378 0.7138 0.7416 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1