#!/bin/bash
#SBATCH --job-name=hpsv2_d_12_s_12_KD_only
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_d_12_s_12_KD_only.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_d_12_s_12_KD_only.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=60:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
#SBATCH --ntasks=1    


#--------------------
# CONDA SETUP
#--------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference_hpsv2.py \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/flux_comp_d_12_s_12_KD_only\
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_comp_d_12_s_12_KD_only/test-step00047500.safetensors \
    --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
    --double_flag_img_attn \
    --double_flag_txt_attn \
    --double_flag_img_mlp \
    --double_flag_txt_mlp \
    --double_flag_img_mod \
    --double_flag_txt_mod \
    --single_blocks_compress 19 24 12 22 26 31 10 20 23 29 15 25 \
    --single_flag_mod \
    --single_flag_mlp2 \
    --single_rank_mlp2 1024 \
    --single_rank_mod 512


