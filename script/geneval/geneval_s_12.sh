#!/bin/bash
#SBATCH --job-name=geneval_comp_s_12
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/geneval_comp_s_12.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/geneval_comp_s_12.txt

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
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/geneval/comp_s_12 \
    --n_samples 4 \
    --ckpt_path_org /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors \
    --double_flag_txt_mod \
    --single_blocks 19 24 12 22 26 31 10 20 23 29 15 25 \
    --single_flag_mod \
    --single_flag_mlp2 \
    --single_rank_mlp2 1024 \
    --single_rank_mod 512
