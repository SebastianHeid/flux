#!/bin/bash
#SBATCH --job-name=30_blocks_compression
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/paper_var_guidance_it/30_blocks_compression/model_40.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/paper_var_guidance_it/30_blocks_compression/model_40.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
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
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_40_cont \
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
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_40_cont \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_40/test-step00015000.safetensors \
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
--compress_single_blocks 0 33 30 2 5 13 22 1 15 18 8 19 16 6 26 7 21 27 12 24 3 17 37 28 14 23 20 31 4 25 10 9 11 34 36 32 \
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
--single_comp_mod 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--single_comp_mlp2 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--single_comp_attn 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--single_comp_mlp 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--double_comp_img_mod 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_img_mlp 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_img_attn 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_mod 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_mlp 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_attn 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_proj 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_img_proj 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1