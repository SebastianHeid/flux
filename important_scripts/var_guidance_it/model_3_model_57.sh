#!/bin/bash
#SBATCH --job-name=hpsv2_var_guidance_it_model_3_57
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_var_guidance_it_model_3_57.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_var_guidance_it_model_3_57.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux2/flux/hpsv2/var_guidance_it/model_3_57 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux2/model_flux/flux/flux1-dev.safetensors \
  --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux2/flux/model_compression/flux_train_var_guidance_it/model_3_57/test-step00020000.safetensors \
    --double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 21 20 11 24 27 25 4 34 7 \
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
    --single_comp_mod 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.3457 0.155 0.3093 0.2979 0.11 0.3149 0.3577 0.2486 0.2533 0.2786 0.3513 0.179403 0.331 0.3109 0.2927 0.2816 0.274 0.2614 0.2324 0.2049 0.1983 0.1981 0.197 0.1964 0.1961 \
    --single_comp_mlp2 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.3457 0.155 0.3093 0.2979 0.11 0.3149 0.3577 0.2486 0.2533 0.2786 0.3513 0.179403 0.331 0.3109 0.2927 0.2816 0.274 0.2614 0.2324 0.2049 0.1983 0.1981 0.197 0.1964 0.1961 \
    --single_comp_attn 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.3457 0.155 0.3093 0.2979 0.11 0.3149 0.3577 0.2486 0.2533 0.2786 0.3513 0.179403 0.331 0.3109 0.2927 0.2816 0.274 0.2614 0.2324 0.2049 0.1983 0.1981 0.197 0.1964 0.1961 \
    --single_comp_mlp 0.515 0.485 0.41 0.395 0.275 0.26 0.22587 0.3457 0.155 0.3093 0.2979 0.11 0.3149 0.3577 0.2486 0.2533 0.2786 0.3513 0.179403 0.331 0.3109 0.2927 0.2816 0.274 0.2614 0.2324 0.2049 0.1983 0.1981 0.197 0.1964 0.1961 \
    --double_comp_img_mod 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_img_mlp 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_img_attn 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_txt_mod 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_txt_mlp 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_txt_attn 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_txt_proj 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \
    --double_comp_img_proj 0.6813 0.6855 0.6619 0.6631 0.6647 0.6504 0.6347 0.6509 0.6492 0.6219 0.6401 0.6115 0.6313 0.6298 0.6075 0.6078 0.5199 0.52 \