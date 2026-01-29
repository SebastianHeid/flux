#!/bin/bash
#SBATCH --job-name=Inference_compressino_ratio
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/Inference_compressino_ratio.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/Inference_compressino_ratio.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=13:00:00 
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


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks_differently/12_blocks/1024_2048/ \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 1024  \
#   --double_rank_img_mlp 2048 \
#   --double_rank_img_attn 2048 \
#   --double_rank_txt_mod 1024 \
#   --double_rank_txt_mlp 2048 \
#   --double_rank_txt_attn 2048

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/512_1024/ \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 512  \
#   --double_rank_img_mlp 1024 \
#   --double_rank_img_attn 1024 \
#   --double_rank_txt_mod 512 \
#   --double_rank_txt_mlp 1024 \
#   --double_rank_txt_attn 1024


python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
  --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 1 \
  --single_blocks_compress 13 14 10 12 11 16 9 15 3 5 \
  --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/256_512/ \
  --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
  --double_rank_img_mod 256  \
  --double_rank_img_mlp 512 \
  --double_rank_img_attn 512 \
  --double_rank_txt_mod 256 \
  --double_rank_txt_mlp 512 \
  --double_rank_txt_attn 512 \
  --single_rank_mlp2 1024 \
  --single_rank_mod 512 \
  --single_flag_mlp2 \
  --single_flag_mod


# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/128_256/ \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 128  \
#   --double_rank_img_mlp 256 \
#   --double_rank_img_attn 256 \
#   --double_rank_txt_mod 128 \
#   --double_rank_txt_mlp 256 \
#   --double_rank_txt_attn 256

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/64_128/ \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 64  \
#   --double_rank_img_mlp 128 \
#   --double_rank_img_attn 128 \
#   --double_rank_txt_mod 64 \
#   --double_rank_txt_mlp 128 \
#   --double_rank_txt_attn 128

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/32_64 \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 32  \
#   --double_rank_img_mlp 64 \
#   --double_rank_img_attn 64 \
#   --double_rank_txt_mod 32 \
#   --double_rank_txt_mlp 64 \
#   --double_rank_txt_attn 64

#   python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/16_32 \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 16  \
#   --double_rank_img_mlp 32 \
#   --double_rank_img_attn 32 \
#   --double_rank_txt_mod 16 \
#   --double_rank_txt_mlp 32 \
#   --double_rank_txt_attn 32

# python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py \
#   --double_blocks_compress 13 14 10 12 11 16 9 15 3 5 17 6 \
#   --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation/12_blocks/8_16 \
#   --double_flag_img_mod --double_flag_img_attn  --double_flag_txt_attn --double_flag_img_mlp --double_flag_txt_mlp --double_flag_txt_mod --double_flag_img_mod \
#   --double_rank_img_mod 8  \
#   --double_rank_img_mlp 16 \
#   --double_rank_img_attn 16 \
#   --double_rank_txt_mod 8 \
#   --double_rank_txt_mlp 16 \
#   --double_rank_txt_attn 16


