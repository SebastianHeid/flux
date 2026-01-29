#!/bin/bash
#SBATCH --job-name=Inference
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/Inference.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/Inference.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=24:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1    

#--------------------
# JOB EXECUTION
#--------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

module purge
module load devel/cuda/12.1

export LD_LIBRARY_PATH=$MODULEPATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH


# img_mod + ALL four others

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/double_exp/double_all_3 \
#   --double_flag_img_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/double_exp/double_all_6 \
#   --double_flag_img_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/double_exp/double_all_9 \
#   --double_flag_img_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/double_exp/double_all_12 \
#   --double_flag_img_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/double_exp/double_all_15 \
#   --double_flag_img_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18 1 0 \
#   --output_dir /home/hd/hd_hd/hd_om233/SVD/flux/image/double_exp/double_all_18 \
#   --double_flag_img_mod --double_flag_img_attn --double_flag_img_mlp --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp



# # ---------------------- how strong to compress ----------------------------------

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/MJHQ_30k/original_model/ \


python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
  --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis_it_1k/original \
  --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \

    



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/MJHQ_30k/flux_train_Grasp/flux_d_18/ \
#   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_grasp/flux_comp_d_18/test-step00045000.safetensors \
#    --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18 1 0 \
#   --double_flag_img_attn \
#   --double_flag_txt_attn \
#   --double_flag_img_mlp \
#   --double_flag_txt_mlp \
#   --double_flag_img_mod \
#   --double_flag_txt_mod \
#   --double_rank_img_mod 526  \
#   --double_rank_img_mlp 491 \
#   --double_rank_img_attn 307 \
#   --double_rank_txt_mod 526 \
#   --double_rank_txt_mlp 491 \
#   --double_rank_txt_attn 307 \


  # --single_blocks_compress 27 31 30 25 6 8 3 10 15 0 33 29 \
  # --single_flag_mod \
  # --single_flag_mlp2 \
  # --single_rank_mlp2 1024 \
  # --single_rank_mod 512 \


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/3_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
 

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/4_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11  \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/5_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/6_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9  \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/7_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/8_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3  \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/9_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



