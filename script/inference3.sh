#!/bin/bash
#SBATCH --job-name=Inference2
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/Inference2.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/Inference2.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=18:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1    

#--------------------
# JOB EXECUTION
#--------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

module purge
module load devel/cuda/12.1

export LD_LIBRARY_PATH=$MODULEPATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH



BLOCKS=(19 24 12 22 26 31 10 20 23 29 15 25 3 18 27 30 28 21 33 32 36 8 14 2 11 16 13 35 17 5 34 1 0 37 6 9 4 7)

# Basis-Pfad für das Python-Skript
SCRIPT_PATH="/home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py"

# Basis-Ausgabepfad
OUTPUT_BASE="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/single_blocks"

# Feste Argumente
STATIC_ARGS="--single_flag_mod --single_flag_mlp2 --single_rank_mlp2 1024 --single_rank_mod 512"

# Variable für die aktuelle Blockliste
current_blocks=""
START_INDEX=15
# Schleife durch die Blöcke von 1 bis zur Gesamtzahl
for ((i=0; i<${#BLOCKS[@]}; i++)); do
    
    # Füge den aktuellen Block zur Liste hinzu
    current_blocks+="${BLOCKS[i]} "
    
    if (( i < START_INDEX )); then
        echo "Überspringe Durchlauf i=$i (Ziel: $START_INDEX)..."
        continue # Springt zur nächsten Iteration
    fi
    # Erstelle den dynamischen Ausgabe-Ordnernamen
    # (i+1, um bei 1_single_block statt 0_single_block zu beginnen)
    output_dir="${OUTPUT_BASE}/$((i+1))_single_block"
    
    # Erstelle den Ordner, falls er nicht existiert
    mkdir -p $output_dir
    
    echo "Führe Inferenz aus für $((i+1)) Blöcke: $current_blocks"
    
    # Führe den Befehl aus
    python $SCRIPT_PATH \
      --single_blocks_compress $current_blocks \
      --output_dir $output_dir \
      $STATIC_ARGS
      
    echo "----------------------------------------------------"
done

echo "Alle Inferenz-Durchläufe abgeschlossen."