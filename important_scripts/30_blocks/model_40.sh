#!/bin/bash
#SBATCH --job-name=hpsv2_paper_var_guidance_it_block_30
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_paper_var_guidance_it_block_30.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_paper_var_guidance_it_block_30.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=20G  
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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/paper_var_guidance_it/30_blocks_compression/model_40 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
  --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_40_cont/test-step00005000.safetensors \
  --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \
  --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 16 6 26 7 21 27 12 24 3 17 37 28 14 23 20 31 4 25 10 9 11 34 36 32 \
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
--single_comp_mod 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--single_comp_mlp2 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--single_comp_attn 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--single_comp_mlp 0.5544 0.5244 0.3046 0.4558 0.5892 0.5119 0.5459 0.6088 0.5411 0.5633 0.4103 0.4713 0.5087 0.5277 0.545 0.45 0.5061 0.3386 0.3375 0.4405 0.5585 0.4589 0.6149 0.4709 0.5069 0.4451 0.461 0.4882 0.4461 0.463 0.539 0.3973 0.3945 0.3808 0.3588 0.3231 \
--double_comp_img_mod 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_img_mlp 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_img_attn 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_mod 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_mlp 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_attn 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_txt_proj 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71 \
--double_comp_img_proj 0.8093 0.8407 0.8026 0.8195 0.665 0.7647 0.7565 0.7735 0.775 0.7424 0.8524 0.6382 0.7909 0.8477 0.7519 0.8185 0.7342 0.71