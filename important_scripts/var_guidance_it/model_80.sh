#!/bin/bash
#SBATCH --job-name=hpsv2_var_guidance_it_model_80
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_var_guidance_it_model_80.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_var_guidance_it_model_80.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=60:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1    


#--------------------
# CONDA SETUP
#--------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference_hpsv2_it.py \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/var_guidance_it/model_80 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_80/test-step00020000.safetensors \
    --double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 \
    --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 \
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