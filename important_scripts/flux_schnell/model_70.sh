#!/bin/bash
#SBATCH --job-name=hpsv2_flux_schnell
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_flux_schnell.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_flux_schnell.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/flux_schnell/model_70 \
      --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_70/test-step00020000.safetensors \
  --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors \
    --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 16 17 12 14 10 6 9 26 27 20 34 28 32 29 23 \
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
    --single_comp_mod 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.1591 0.0172 0.0089 0.2874 0.2986 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 0.1987 0.1897 0.1806 0.1776 0.1746 0.1716 0.1655 \
    --single_comp_mlp2 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.1591 0.0172 0.0089 0.2874 0.2986 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 0.1987 0.1897 0.1806 0.1776 0.1746 0.1716 0.1655 \
    --single_comp_attn 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.1591 0.0172 0.0089 0.2874 0.2986 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 0.1987 0.1897 0.1806 0.1776 0.1746 0.1716 0.1655 \
    --single_comp_mlp 0.1004 0.2587 0.0837 0.0754 0.0671 0.1599 0.3199 0.1166 0.1909 0.1591 0.0172 0.0089 0.2874 0.2986 0.1252 0.1133 0.1014 0.0955 0.0896 0.0836 0.1987 0.1897 0.1806 0.1776 0.1746 0.1716 0.1655 \
    --double_comp_img_mod 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_img_mlp 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_img_attn 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_txt_mod 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_txt_mlp 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_txt_attn 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_txt_proj 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \
    --double_comp_img_proj 0.5190 0.5265 0.4985 0.4846 0.5152 0.5166 0.5053 0.5161 0.5109 0.4355 0.4789 0.4705 0.5047 0.4900 0.4732 0.4659 0.4570 0.4684 \