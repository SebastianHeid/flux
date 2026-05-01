#!/bin/bash
#SBATCH --job-name=hpsv2_var_guidance_it_model_4_49
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_var_guidance_it_model_4_49.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_var_guidance_it_model_4_49.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/var_guidance_it/model_4_49 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_4_49/test-step00020000.safetensors \
    --double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 21 20 11 24 27 25 4 34 7 10 \
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
    --single_comp_mod 0.515 0.485 0.41 0.395 0.275 0.26 0.3232 0.4836 0.3251 0.3093 0.3994 0.2724 0.4723 0.3577 0.2486 0.4602 0.2786 0.3513 0.4844 0.331 0.3109 0.2927 0.2816 0.4889 0.4031 0.2324 0.3951 0.4128 0.4506 0.3283 0.1964 0.1961 0.2203 \
    --single_comp_mlp2 0.515 0.485 0.41 0.395 0.275 0.26 0.3232 0.4836 0.3251 0.3093 0.3994 0.2724 0.4723 0.3577 0.2486 0.4602 0.2786 0.3513 0.4844 0.331 0.3109 0.2927 0.2816 0.4889 0.4031 0.2324 0.3951 0.4128 0.4506 0.3283 0.1964 0.1961 0.2203 \
    --single_comp_attn 0.515 0.485 0.41 0.395 0.275 0.26 0.3232 0.4836 0.3251 0.3093 0.3994 0.2724 0.4723 0.3577 0.2486 0.4602 0.2786 0.3513 0.4844 0.331 0.3109 0.2927 0.2816 0.4889 0.4031 0.2324 0.3951 0.4128 0.4506 0.3283 0.1964 0.1961 0.2203 \
    --single_comp_mlp 0.515 0.485 0.41 0.395 0.275 0.26 0.3232 0.4836 0.3251 0.3093 0.3994 0.2724 0.4723 0.3577 0.2486 0.4602 0.2786 0.3513 0.4844 0.331 0.3109 0.2927 0.2816 0.4889 0.4031 0.2324 0.3951 0.4128 0.4506 0.3283 0.1964 0.1961 0.2203 \
    --double_comp_img_mod 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_img_mlp 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_img_attn 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_txt_mod 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_txt_mlp 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_txt_attn 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_txt_proj 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \
    --double_comp_img_proj 0.7967 0.6855 0.778 0.6631 0.71 0.7506 0.729 0.7674 0.7729 0.6219 0.7024 0.6714 0.7509 0.7429 0.7645 0.761 0.6393 0.7029 \