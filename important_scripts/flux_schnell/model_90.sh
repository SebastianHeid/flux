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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/flux_schnell/model_90 \
      --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_90/test-step00020000.safetensors \
  --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors \
    --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 \
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
    --single_comp_mod 0.1004 0.092 0.0837 0.0754 0.0671 0.0588 0.0505 0.0422 0.0339 0.0255 0.0172 0.0089 \
    --single_comp_mlp2 0.1004 0.092 0.0837 0.0754 0.0671 0.0588 0.0505 0.0422 0.0339 0.0255 0.0172 0.0089 \
    --single_comp_attn 0.1004 0.092 0.0837 0.0754 0.0671 0.0588 0.0505 0.0422 0.0339 0.0255 0.0172 0.0089 \
    --single_comp_mlp 0.1004 0.092 0.0837 0.0754 0.0671 0.0588 0.0505 0.0422 0.0339 0.0255 0.0172 0.0089 \
    --double_comp_img_mod 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_img_mlp 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_img_attn 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_txt_mod 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_txt_mlp 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_txt_attn 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_txt_proj 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \
    --double_comp_img_proj 0.25 0.2417 0.2334 0.2251 0.2167 0.2084 0.2001 0.1918 0.1835 0.1752 0.1669 0.1586 0.1502 0.1419 0.1336 0.1253 0.117 0.1087 \