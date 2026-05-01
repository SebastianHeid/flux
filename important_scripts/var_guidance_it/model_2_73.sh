#!/bin/bash
#SBATCH --job-name=hpsv2_var_guidance_it_model_2_73_comp
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_var_guidance_it_model_2_73_comp.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_var_guidance_it_model_2_73_comp.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/var_guidance_it/model_2_73 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_2_73/test-step00020000.safetensors \
    --double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 21 20 11 24 \
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