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


python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference_iterative.py \
  --output_dir /home/hd/hd_hd/hd_om233/SVD/images/flux_schnell/compressed \
  --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
  --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_80/test-step00020000.safetensors \
  --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \
  --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 16 6 26 7 21 27 12 24 3 \
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
  --single_comp_mod 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
  --single_comp_mlp2 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
  --single_comp_attn 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
  --single_comp_mlp 0.1004 0.1954 0.1989 0.0754 0.0671 0.0588 0.0505 0.0422 0.1496 0.0255 0.0172 0.0089 0.1435 0.1375 0.1316 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 \
  --double_comp_img_mod 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_img_mlp 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_img_attn 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_txt_mod 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_txt_mlp 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_txt_attn 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_txt_proj 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315 \
  --double_comp_img_proj 0.3798 0.3819 0.3797 0.3383 0.3638 0.4016 0.3338 0.3795 0.3441 0.3082 0.3505 0.349 0.3224 0.3143 0.3463 0.3129 0.2489 0.3315

