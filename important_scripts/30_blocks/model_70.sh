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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/paper_var_guidance_it/30_blocks_compression/model_70 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/30_blocks_compression/model_70/test-step00020000.safetensors \
  --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \
  --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 16 6 26 7 21 27 12 24 3 17 37 \
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
--single_comp_mod 0.1004 0.2799 0.3046 0.2098 0.0671 0.0588 0.0505 0.1879 0.2904 0.1802 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 0.1184 0.1117 \
--single_comp_mlp2 0.1004 0.2799 0.3046 0.2098 0.0671 0.0588 0.0505 0.1879 0.2904 0.1802 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 0.1184 0.1117 \
--single_comp_attn 0.1004 0.2799 0.3046 0.2098 0.0671 0.0588 0.0505 0.1879 0.2904 0.1802 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 0.1184 0.1117 \
--single_comp_mlp 0.1004 0.2799 0.3046 0.2098 0.0671 0.0588 0.0505 0.1879 0.2904 0.1802 0.0172 0.133 0.2968 0.2571 0.2812 0.1079 0.102 0.0961 0.0902 0.0843 0.0784 0.1184 0.1117 \
--double_comp_img_mod 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_img_mlp 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_img_attn 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_txt_mod 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_txt_mlp 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_txt_attn 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_txt_proj 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321 \
--double_comp_img_proj 0.5492 0.5632 0.4991 0.4923 0.529 0.569 0.4978 0.5573 0.4747 0.4832 0.4886 0.4961 0.5029 0.4555 0.5029 0.4405 0.4136 0.5321