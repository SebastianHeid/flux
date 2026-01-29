#!/bin/bash
#SBATCH --job-name=geneval_FluxMini
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/geneval_FluxMini.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/geneval_FluxMini.txt

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
conda activate flux-mini

python /home/hd/hd_hd/hd_om233/FluxKits/flux-mini/src/inference_geneval.py \
    --metadata_file /home/hd/hd_hd/hd_om233/ModelEvaluationBenchmarks/geneval/prompts/evaluation_metadata.jsonl \
    --n_samples 4 \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/geneval/FluxMini \

