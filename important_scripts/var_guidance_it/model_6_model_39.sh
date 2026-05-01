#!/bin/bash
#SBATCH --job-name=hpsv2_var_guidance_it_model_6_39
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_var_guidance_it_model_6_39.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_var_guidance_it_model_6_39.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/var_guidance_it/model_6_39 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
   --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_6_39/test-step00020000.safetensors \
    --double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \
    --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 21 20 11 24 27 25 4 34 7 10 9 32 36 35 \
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
    --single_comp_mod 0.515 0.485 0.4777 0.395 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.5129 0.4821 0.5934 0.5205 0.4941 0.5745 0.4759 0.4999 0.4844 0.4602 0.4351 0.4924 0.3894 0.552 0.461 0.4364 0.5445 0.5741 0.4991 0.5154 0.3028 0.4302 0.4941 0.3878 0.4848 0.3145 0.3 \
    --single_comp_mlp2 0.515 0.485 0.4777 0.395 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.5129 0.4821 0.5934 0.5205 0.4941 0.5745 0.4759 0.4999 0.4844 0.4602 0.4351 0.4924 0.3894 0.552 0.461 0.4364 0.5445 0.5741 0.4991 0.5154 0.3028 0.4302 0.4941 0.3878 0.4848 0.3145 0.3 \
    --single_comp_attn 0.515 0.485 0.4777 0.395 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.5129 0.4821 0.5934 0.5205 0.4941 0.5745 0.4759 0.4999 0.4844 0.4602 0.4351 0.4924 0.3894 0.552 0.461 0.4364 0.5445 0.5741 0.4991 0.5154 0.3028 0.4302 0.4941 0.3878 0.4848 0.3145 0.3 \
    --single_comp_mlp 0.515 0.485 0.4777 0.395 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.5129 0.4821 0.5934 0.5205 0.4941 0.5745 0.4759 0.4999 0.4844 0.4602 0.4351 0.4924 0.3894 0.552 0.461 0.4364 0.5445 0.5741 0.4991 0.5154 0.3028 0.4302 0.4941 0.3878 0.4848 0.3145 0.3 \
    --double_comp_img_mod 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_img_mlp 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_img_attn 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_txt_mod 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_txt_mlp 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_txt_attn 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_txt_proj 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \
    --double_comp_img_proj 0.7967 0.8113 0.7936 0.7849 0.7587 0.7773 0.7983 0.7859 0.817 0.7593 0.7936 0.7304 0.8159 0.8042 0.8081 0.799 0.7484 0.7527 \