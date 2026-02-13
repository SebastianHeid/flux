#!/bin/bash
#SBATCH --job-name=Inference1
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/Inference1.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/Inference1.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=24:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
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




python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
  --double_blocks_compress 13   \
  --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/10_block \
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



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17  \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/11_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/12_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
 

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/13_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7  \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/14_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/15_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18  \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/16_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18 1 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/17_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18 1 0 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/18_block \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \



