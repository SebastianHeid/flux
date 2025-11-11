#!/bin/bash
#SBATCH --job-name=Inference
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/Inference.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/Inference.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=4:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
#SBATCH --ntasks=1    

#--------------------
# JOB EXECUTION
#--------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --single_blocks 19 24 12 22 26 31 10 20 23 29 15 25 3 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/FastFlux/single_mod \
#   --single_flag_mod 



# 1) img_attn
# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks 13 14 10 11 16 12 15 3 \
#   --single_blocks 19 24 12 22 26 31 10 20 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp/double_all_red_rank \
#   --double_flag_img_attn \
#   --double_flag_txt_attn \
#   --double_flag_img_mlp \
#   --double_flag_txt_mlp \
#   --double_flag_img_mod \
#   --double_flag_txt_mod \
#   --single_flag_mlp2 \
#   --single_flag_mod \

#python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py --double_blocks 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp_new/double_img_attn --double_flag_img_attn

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1_train/comp_d_18_org_loss_25k \
#      --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/comp_d_18_org_loss/test-step00025000.safetensors \
#     --double_blocks 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18 1 0 \
#     --double_flag_img_attn \
#     --double_flag_txt_attn \
#     --double_flag_img_mlp \
#     --double_flag_txt_mlp \
#     --double_flag_img_mod \
#     --double_flag_txt_mod \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/new_images/no_training \
#   --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#      --single_blocks_compress  19 24 12 22 26 31 10 20 23 29 15 25 \
#     --single_flag_mod \
#     --single_flag_mlp2 \
#     --single_rank_mlp2 1024 \
#     --single_rank_mod 512 \
#   --double_blocks_compress  13 14 10 12 11 16 9 15 3 5 17 6 \
#     --double_flag_img_attn \
#     --double_flag_txt_attn \
#     --double_flag_img_mlp \
#     --double_flag_txt_mlp \
#     --double_flag_img_mod \
#     --double_flag_txt_mod \

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
  --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/new_images2/flux_comp_d_12_s_12_KD_only_2 \
  --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
  --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_comp_d_12_s_12_KD_only/test-step00047500.safetensors\
   --single_blocks_compress  19 24 12 22 26 31 10 20 23 29 15 25 \
    --single_flag_mod \
    --single_flag_mlp2 \
    --single_rank_mlp2 1024 \
    --single_rank_mod 512 \
  --double_blocks_compress  13 14 10 12 11 16 9 15 3 5 17 6 \
    --double_flag_img_attn \
    --double_flag_txt_attn \
    --double_flag_img_mlp \
    --double_flag_txt_mlp \
    --double_flag_img_mod \
    --double_flag_txt_mod \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/new_images2/flux_comp_d_12_s_12_org_loss_all_trainable \
#   --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_comp_d_12_s_12_org_loss_all_trainable/test-step00050000.safetensors \
#    --single_blocks_compress  19 24 12 22 26 31 10 20 23 29 15 25 \
#     --single_flag_mod \
#     --single_flag_mlp2 \
#     --single_rank_mlp2 1024 \
#     --single_rank_mod 512 \
#   --double_blocks_compress  13 14 10 12 11 16 9 15 3 5 17 6 \
#     --double_flag_img_attn \
#     --double_flag_txt_attn \
#     --double_flag_img_mlp \
#     --double_flag_txt_mlp \
#     --double_flag_img_mod \
#     --double_flag_txt_mod \


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/new_images2/flux_comp_d_12_s_12_KD_only \
#   --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_comp_d_12_s_12_KD_only/test-step00047500.safetensors \
#    --single_blocks_compress  19 24 12 22 26 31 10 20 23 29 15 25 \
#     --single_flag_mod \
#     --single_flag_mlp2 \
#     --single_rank_mlp2 1024 \
#     --single_rank_mod 512 \
#   --double_blocks_compress  13 14 10 12 11 16 9 15 3 5 17 6 \
#     --double_flag_img_attn \
#     --double_flag_txt_attn \
#     --double_flag_img_mlp \
#     --double_flag_txt_mlp \
#     --double_flag_img_mod \
#     --double_flag_txt_mod \


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/new_images2/flux_comp_d_12_s_12_KD_plus_org_loss_long \
#   --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_comp_d_12_s_12_KD_plus_org_loss_long/test-step00023500.safetensors \
#    --single_blocks_compress  19 24 12 22 26 31 10 20 23 29 15 25 \
#     --single_flag_mod \
#     --single_flag_mlp2 \
#     --single_rank_mlp2 1024 \
#     --single_rank_mod 512 \
#   --double_blocks_compress  13 14 10 12 11 16 9 15 3 5 17 6 \
#     --double_flag_img_attn \
#     --double_flag_txt_attn \
#     --double_flag_img_mlp \
#     --double_flag_txt_mlp \
#     --double_flag_img_mod \
#     --double_flag_txt_mod \
   

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/new_images2/flux_original \
#   --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors

   


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/rem_compress/s_8_4_d_12_0 \
#   --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#     --double_flag_img_attn \
#     --double_flag_txt_attn \
#     --double_flag_img_mlp \
#     --double_flag_txt_mlp \
#     --double_flag_img_mod \
#     --double_flag_txt_mod \
#     --single_blocks_compress  26 31 10 20 23 29 15 25\
#     --single_blocks 19 24 12 22  \
#     --single_flag_mod \
#     --single_flag_mlp2 \
#     --single_rank_mlp2 1024 \
#     --single_rank_mod 512

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mod3 \
#   --single_blocks 19 24 12 22 26 31 \
#   --single_flag_mod \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mod4 \
#   --single_blocks 19 24 12 22 26 31 10 20 \
#   --single_flag_mod \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mod5 \
#   --single_blocks 19 24 12 22 26 31 10 20 23 29  \
#   --single_flag_mod \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mod6 \
#   --single_blocks 19 24 12 22 26 31 10 20 23 29 15 25  \
#   --single_flag_mod \



#   python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mlp2_1 \
#   --single_blocks 19 24 \
#   --single_flag_mlp2 \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mlp2_2 \
#   --single_blocks 19 24 12 22 \
#   --single_flag_mlp2 \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mlp2_3 \
#   --single_blocks 19 24 12 22 26 31 \
#   --single_flag_mlp2 \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mlp2_4 \
#   --single_blocks 19 24 12 22 26 31 10 20 \
#   --single_flag_mlp2 \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mlp2_5 \
#   --single_blocks 19 24 12 22 26 31 10 20 23 29  \
#   --single_flag_mlp2 \

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/s_mlp2_6 \
#   --single_blocks 19 24 12 22 26 31 10 20 23 29 15 25  \
#   --single_flag_mlp2 \


# #python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py --double_blocks 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp1/double_img_attn_mlp_mod_txt_attn_mlp_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_img_mod --double_flag_txt_attn --double_flag_txt_mlp --double_flag_txt_mod
# # python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
# #   --single_blocks 19 24 12 22 26 31 10 20 \
# #   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp/single_l \
# #   --single_flag_attn \
# #   --single_flag_mlp \
# #   --single_flag_mlp2 \
# #   --single_flag_mod \


#   # python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   # --double_blocks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
#   # --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/exp/double_all_ext \
#   # --double_flag_img_attn \
#   # --double_flag_txt_attn \
#   # --double_flag_img_mlp \
#   # --double_flag_txt_mlp \
#   # --double_flag_img_mod \
#   # --double_flag_txt_mod 