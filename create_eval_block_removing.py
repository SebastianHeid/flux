import os

def generate_slurm_scripts():
    # --- ZIELVERZEICHNIS (Kombiniert) ---
    output_script_dir = "/home/hd/hd_hd/hd_om233/SVD/flux/script/block_analysis/removal/original_model"
    os.makedirs(output_script_dir, exist_ok=True)

    # --- GEMEINSAME KONFIGURATION ---
    base_output_path = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/1k_removal/original_model"
    base_log_path = "/home/hd/hd_hd/hd_om233/SVD/flux/1k_removal/original_model"
    ckpt = "/gpfs/bwfor/work/ws/hd_om233-flux/model_flux/flux/flux1-dev.safetensors"
    prompts = "/home/hd/hd_hd/hd_om233/partially_removal/block_eval/1k_prompts.json"

    # Statische Kompressions-Parameter
    comp_args = (
    )

    # --- Hilfsfunktion für Slurm Header & Template ---
    def write_script(block_id, is_double):
        prefix = "double" if is_double else "single"
        short = "DB" if is_double else "SB"
        block_arg = "--double_block_list" if is_double else "--single_block_list"
        
        content = f"""#!/bin/bash
#SBATCH --job-name={short}_an_{block_id}
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/{prefix}_block_{block_id}.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/{prefix}_block_{block_id}.txt
#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=4:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train
module purge
module load devel/cuda/12.1

export LD_LIBRARY_PATH=$MODULEPATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_block_analysis_svd.py \\
  {block_arg} {block_id} \\
  --prompt_path {prompts} \\
  --ckpt_path {ckpt} \\
  --output_dir {base_output_path}/ \\
  --logdir {base_log_path}/{prefix}_block_{block_id}.txt \\
  --double_flag_img_attn \\
  --double_flag_txt_attn \\
  --double_flag_img_mlp \\
  --double_flag_txt_mlp \\
  --double_flag_img_mod \\
  --double_flag_txt_mod \\
  --double_flag_txt_proj \\
  --double_flag_img_proj \\
  --single_flag_mod \\
  --single_flag_mlp2 \\
  --single_flag_mlp \\
  --single_flag_attn \\
  
"""
        filename = f"job_{prefix}_block_{block_id:02d}.sh"
        with open(os.path.join(output_script_dir, filename), "w") as f:
            f.write(content)

    # --- GENERIERUNG ---
    for i in range(19): # Double Blocks
        write_script(i, True)
    
    for i in range(38): # Single Blocks
        write_script(i, False)

    print(f"Erfolg! Insgesamt 57 Skripte wurden in {output_script_dir} erstellt.")

if __name__ == "__main__":
    generate_slurm_scripts()