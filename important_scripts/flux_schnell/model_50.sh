#!/bin/bash
#SBATCH --job-name=hpsv2_flux_schnell
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_flux_schnell.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_flux_schnell.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=40:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1    


#--------------------
# CONDA SETUP
#--------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference_hpsv2_it.py \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/flux_schnell/model_50 \
      --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_schnell/model_50/test-step00020000.safetensors \
  --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/flux/flux_schnell/flux1-schnell.safetensors \
    --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 16 17 12 14 10 6 9 26 29 27 7 37 3 4 21 20 35 23 28 32 11 34 \
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
    --single_comp_mod 0.3839 0.2587 0.0837 0.3829 0.382 0.4372 0.3199 0.3884 0.1909 0.3332 0.3319 0.0089 0.2874 0.4953 0.3922 0.5374 0.3645 0.4031 0.0896 0.2886 0.3899 0.3965 0.2403 0.4466 0.2303 0.2203 0.4889 0.3602 0.2037 0.3475 0.345 0.3351 0.3276 0.3102 \
    --single_comp_mlp2 0.3839 0.2587 0.0837 0.3829 0.382 0.4372 0.3199 0.3884 0.1909 0.3332 0.3319 0.0089 0.2874 0.4953 0.3922 0.5374 0.3645 0.4031 0.0896 0.2886 0.3899 0.3965 0.2403 0.4466 0.2303 0.2203 0.4889 0.3602 0.2037 0.3475 0.345 0.3351 0.3276 0.3102 \
    --single_comp_attn 0.3839 0.2587 0.0837 0.3829 0.382 0.4372 0.3199 0.3884 0.1909 0.3332 0.3319 0.0089 0.2874 0.4953 0.3922 0.5374 0.3645 0.4031 0.0896 0.2886 0.3899 0.3965 0.2403 0.4466 0.2303 0.2203 0.4889 0.3602 0.2037 0.3475 0.345 0.3351 0.3276 0.3102 \
    --single_comp_mlp 0.3839 0.2587 0.0837 0.3829 0.382 0.4372 0.3199 0.3884 0.1909 0.3332 0.3319 0.0089 0.2874 0.4953 0.3922 0.5374 0.3645 0.4031 0.0896 0.2886 0.3899 0.3965 0.2403 0.4466 0.2303 0.2203 0.4889 0.3602 0.2037 0.3475 0.345 0.3351 0.3276 0.3102 \
    --double_comp_img_mod 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_img_mlp 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_img_attn 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_txt_mod 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_txt_mlp 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_txt_attn 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_txt_proj 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \
    --double_comp_img_proj 0.7492 0.7518 0.6323 0.6101 0.7416 0.6472 0.7519 0.7627 0.6463 0.7061 0.7378 0.6293 0.7493 0.7454 0.7541 0.7138 0.7352 0.5838 \