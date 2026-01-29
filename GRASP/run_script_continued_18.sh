#!/bin/bash
#SBATCH --job-name=GRASP_d_18_continued
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/output/GRASP_d_18_continued.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/train_script/error/GRASP_d_18_continued.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem=40gb
#SBATCH --export=NONE
#SBATCH --gres=gpu:H200:1
# >>> Conda setup >>>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_optuna
# >>> Conda setup >>>



cd /home/hd/hd_hd/hd_om233/SVD/flux/GRASP/
python -m grasp\
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
    --logging_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/GRASP/flux_comp_d_18 \
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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/GRASP/flux_comp_d_18 \
    --output_name test \
    --persistent_data_loader_workers \
    --pretrained_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --pruned_model_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/GRASP/flux_comp_d_15/compressed_model.safetensors \
    --pretrained_ref_model_name_or_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --save_every_n_steps 200 \
    --save_model_as safetensors \
    --save_precision bf16 \
    --sdpa \
    --seed 42 \
    --t5xxl /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/t5xxl/model.safetensors \
    --t5xxl_max_token_length 512 \
    --timestep_sampling "sigmoid" \
    --sample_every_n_steps 5000 \
    --sample_at_first \
    --sample_prompts None \
    --sample_sampler euler_a \
    --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 \
    --compress_double_blocks_new 18 1 0   \
    --double_flag_img_attn \
    --double_flag_txt_attn \
    --double_flag_img_mlp \
    --double_flag_txt_mlp \
    --double_flag_img_mod \
    --double_flag_txt_mod \
    --double_rank_img_mod 526  \
    --double_rank_img_mlp 491 \
    --double_rank_img_attn 307 \
    --double_rank_txt_mod 526 \
    --double_rank_txt_mlp 491 \
    --double_rank_txt_attn 307 \
