#!/bin/bash
#SBATCH --job-name=var_guidance_it
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/flux_train_var_guidance_it/model_3_57.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/flux_train_var_guidance_it/model_3_57.txt

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
--logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_3_57 \
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
--output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_3_57 \
--output_name test \
--persistent_data_loader_workers \
--pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
--pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_2_73/test-step00020000.safetensors \
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
--compress_double_blocks 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \
--compress_single_blocks 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 21 20 11 24 \
--compress_double_blocks_new 0 9 10 7 18 1 12 8 4 11 17 14 5 3 6 15 13 16 \
--compress_single_blocks_new 21 24 27 25 31 37 19 16 28 26 15 4 6 8 22 11 17 34 12 14 7 20 \
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
--single_comp_mod 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.185 0.155 0.14 0.125 0.11 0.146448 0.199951 0.065 0.069078 0.101685 0.191658 0.179403 0.167147 0.142636 0.118126 0.10587 0.093615 0.081359 0.044593 0.007827 \
--single_comp_mlp2 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.185 0.155 0.14 0.125 0.11 0.146448 0.199951 0.065 0.069078 0.101685 0.191658 0.179403 0.167147 0.142636 0.118126 0.10587 0.093615 0.081359 0.044593 0.007827 \
--single_comp_attn 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.185 0.155 0.14 0.125 0.11 0.146448 0.199951 0.065 0.069078 0.101685 0.191658 0.179403 0.167147 0.142636 0.118126 0.10587 0.093615 0.081359 0.044593 0.007827 \
--single_comp_mlp 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.185 0.155 0.14 0.125 0.11 0.146448 0.199951 0.065 0.069078 0.101685 0.191658 0.179403 0.167147 0.142636 0.118126 0.10587 0.093615 0.081359 0.044593 0.007827 \
--double_comp_img_mod 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_img_mlp 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_img_attn 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_txt_mod 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_txt_mlp 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_txt_attn 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_txt_proj 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
--double_comp_img_proj 0.602799 0.608084 0.578074 0.579491 0.581644 0.563391 0.544418 0.564525 0.562204 0.527651 0.550332 0.51486 0.539195 0.537747 0.50969 0.509759 0.40028 0.4 \
    --single_comp_mod_new 0.274 0.2049 0.1983 0.1981 0.2927 0.2533 0.2979 0.3513 0.3149 0.3577 0.3457 0.197 0.2786 0.3093 0.331 0.2324 0.2816 0.1964 0.2486 0.3109 0.1961 0.2614 \
  --single_comp_mlp2_new 0.274 0.2049 0.1983 0.1981 0.2927 0.2533 0.2979 0.3513 0.3149 0.3577 0.3457 0.197 0.2786 0.3093 0.331 0.2324 0.2816 0.1964 0.2486 0.3109 0.1961 0.2614 \
  --single_comp_attn_new 0.274 0.2049 0.1983 0.1981 0.2927 0.2533 0.2979 0.3513 0.3149 0.3577 0.3457 0.197 0.2786 0.3093 0.331 0.2324 0.2816 0.1964 0.2486 0.3109 0.1961 0.2614 \
  --single_comp_mlp_new 0.274 0.2049 0.1983 0.1981 0.2927 0.2533 0.2979 0.3513 0.3149 0.3577 0.3457 0.197 0.2786 0.3093 0.331 0.2324 0.2816 0.1964 0.2486 0.3109 0.1961 0.2614 \
  --double_comp_img_mod_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_img_mlp_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_img_attn_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_txt_mod_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_txt_mlp_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_txt_attn_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_txt_proj_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
  --double_comp_img_proj_new 0.52 0.6078 0.6313 0.6401 0.6219 0.5199 0.6075 0.6504 0.6115 0.6298 0.6631 0.6492 0.6619 0.6647 0.6509 0.6347 0.6813 0.6855 \
--final_loss_weighting 10 \
--single_loss_weighting 0.05 \
--double_loss_weighting 0.1 \
--original_loss_weighting 1


