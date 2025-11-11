#!/bin/bash
#SBATCH --job-name=hpsv2_s_12
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/hpsv2_s_12.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/hpsv2_s_12.txt

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

python /home/hd/hd_hd/hd_om233/SVD/flux/FluxLite_hpsv2.py \
    --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/hpsv2/FluxLite \

