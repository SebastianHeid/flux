#!/bin/bash
#SBATCH --job-name=Inference
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/Inference.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/Inference.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=20G  
#SBATCH --time=4:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:1,gpumem_per_gpu:20GB
#SBATCH --ntasks=1    

#--------------------
# JOB EXECUTION
#--------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixart

module purge
module load devel/cuda/12.1

export LD_LIBRARY_PATH=$MODULEPATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

#!/bin/bash

# --- KONFIGURATION ---
# Bitte passen Sie die Endnummer an, je nachdem, wie viele Blöcke Sie testen möchten.
# Beispiel: Wenn Sie .../1_block bis .../38_block testen wollen, setzen Sie END_BLOCK=38
START_BLOCK=1
END_BLOCK=38  # <-- Passen Sie diese Zahl nach Bedarf an

# --- Pfade (aus Ihrem Befehl) ---
PYTHON_SCRIPT="/home/hd/hd_hd/hd_om233/partially_removal/MasterThesis_Evaluation/evaluation_cmmd_datasets.py"
REF_DATASET="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/original"
SAVE_FILE="/home/hd/hd_hd/hd_om233/partially_removal/MasterThesis_Evaluation/logs/flux/block_compression_single_blocks.txt"
COMP_DATASET_BASE_PATH="/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_investigation_compress_blocks/single_blocks"

# --- Optional: Alte Log-Datei löschen ---
# (Kommentieren Sie diese Zeile aus, wenn Sie stattdessen an eine bestehende Datei anhängen möchten)
rm -f "$SAVE_FILE"
echo "Starte CMMD-Auswertung von Block $START_BLOCK bis $END_BLOCK..."

# --- Ausführungsschleife ---
for (( i=$START_BLOCK; i<=$END_BLOCK; i++ ))
do
    # 1. Den dynamischen Pfad für den aktuellen Block erstellen
    CURRENT_COMP_DATASET="${COMP_DATASET_BASE_PATH}/${i}_single_block"

    # 2. Prüfen, ob der Ordner existiert, bevor das Skript gestartet wird
    if [ ! -d "$CURRENT_COMP_DATASET" ]; then
        echo "Warnung: Ordner nicht gefunden, überspringe: $CURRENT_COMP_DATASET"
        # Optional: Einen Vermerk in der Log-Datei hinterlassen
        echo "--- Ordner ${i}_single_block NICHT GEFUNDEN ---" >> "$SAVE_FILE"
        echo "" >> "$SAVE_FILE"
        continue # Springe zum nächsten Schleifendurchlauf
    fi

    echo "Bewerte jetzt: ${i}_single_block"

    # 3. Eine Kopfzeile in die Log-Datei schreiben, um den Score zuzuordnen
    echo "--- Ergebnisse für: ${i}_single_block ---" >> "$SAVE_FILE"

    # 4. Den Python-Befehl ausführen
    python "$PYTHON_SCRIPT" \
      --ref_dataset "$REF_DATASET" \
      --comp_dataset "$CURRENT_COMP_DATASET" \
      --save_file "$SAVE_FILE"

    # 5. Eine Leerzeile für bessere Lesbarkeit in der Log-Datei
    echo "" >> "$SAVE_FILE"
done

echo "Alle Auswertungen abgeschlossen. Ergebnisse sind in $SAVE_FILE"