#!/bin/bash
#SBATCH --job-name=geneval_FluxLite
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/geneval_FluxLite.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/geneval_FluxLite.txt

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
conda activate FluxLite

python /home/hd/hd_hd/hd_om233/SVD/flux/FluxLite_geneval.py \
    --metadata_file /home/hd/hd_hd/hd_om233/ModelEvaluationBenchmarks/geneval/prompts/evaluation_metadata.jsonl \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/geneval/FluxLite \
    --n_samples 4 \
  

