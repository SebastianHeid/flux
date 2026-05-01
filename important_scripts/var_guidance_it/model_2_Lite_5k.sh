#!/bin/bash
#SBATCH --job-name=hpsv2_var_guidance_it_model_2_Lite_comp_5k
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_var_guidance_it_model_2_Lite_comp_5k.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_var_guidance_it_model_2_Lite_comp_5k.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/var_guidance_it/model_2_Lite_comp_5k \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
 --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_2_Lite_comp/test-step00005000.safetensors \
    --double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 20 21 11 24 27 4 7 25 \
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
    --single_comp_mod 0.515 0.485 0.41 0.395 0.275 0.26 0.2564 0.185 0.155 0.14 0.1599 0.1319 0.1726 0.2011 0.0737 0.1097 0.1251 0.1699 0.1623 0.1546 0.1393 0.124 0.1164 0.1087 0.1011 0.0781 0.0552 0.0475 0.0322 0.0169 0.0016 \
    --single_comp_mlp2 0.515 0.485 0.41 0.395 0.275 0.26 0.2564 0.185 0.155 0.14 0.1599 0.1319 0.1726 0.2011 0.0737 0.1097 0.1251 0.1699 0.1623 0.1546 0.1393 0.124 0.1164 0.1087 0.1011 0.0781 0.0552 0.0475 0.0322 0.0169 0.0016 \
    --single_comp_attn 0.515 0.485 0.41 0.395 0.275 0.26 0.2564 0.185 0.155 0.14 0.1599 0.1319 0.1726 0.2011 0.0737 0.1097 0.1251 0.1699 0.1623 0.1546 0.1393 0.124 0.1164 0.1087 0.1011 0.0781 0.0552 0.0475 0.0322 0.0169 0.0016 \
    --single_comp_mlp 0.515 0.485 0.41 0.395 0.275 0.26 0.2564 0.185 0.155 0.14 0.1599 0.1319 0.1726 0.2011 0.0737 0.1097 0.1251 0.1699 0.1623 0.1546 0.1393 0.124 0.1164 0.1087 0.1011 0.0781 0.0552 0.0475 0.0322 0.0169 0.0016 \
    --double_comp_img_mod 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_img_mlp 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_img_attn 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_txt_mod 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_txt_mlp 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_txt_attn 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_txt_proj 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \
    --double_comp_img_proj 0.5991 0.5926 0.5641 0.5601 0.5566 0.5403 0.5138 0.5215 0.5152 0.4887 0.498 0.471 0.4813 0.4657 0.4433 0.4385 0.3555 0.3 \