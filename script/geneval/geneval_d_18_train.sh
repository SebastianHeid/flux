#!/bin/bash
#SBATCH --job-name=geneval_comp_d_18_train
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/geneval_comp_d_18_train.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/geneval_comp_d_18_train.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=60:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
#SBATCH --ntasks=1    


#--------------------
# CONDA SETUP
#--------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference_geneval.py \
    --metadata_file /home/hd/hd_hd/hd_om233/ModelEvaluationBenchmarks/geneval/prompts/evaluation_metadata.jsonl \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/geneval/comp_d_18_train \
    --n_samples 4 \
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/comp_d_18/test-step00007500.safetensors\
    --double_blocks 13 14 10 12 11 16 9 15 3 5 17 6 4 7 8 18 1 0 \
    --double_flag_img_attn \
    --double_flag_txt_attn \
    --double_flag_img_mlp \
    --double_flag_txt_mlp \
    --double_flag_img_mod \
    --double_flag_txt_mod \

