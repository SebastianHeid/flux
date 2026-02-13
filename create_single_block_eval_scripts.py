import os

def generate_slurm_scripts():
    # --- ZIELVERZEICHNIS (Kombiniert) ---
    output_script_dir = "/home/hd/hd_hd/hd_om233/SVD/flux/script/block_analysis/block_30/model_80_100"
    os.makedirs(output_script_dir, exist_ok=True)

    # --- GEMEINSAME KONFIGURATION ---
    base_output_path = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis_paper_it/block_30/model_80_100"
    base_log_path = "/home/hd/hd_hd/hd_om233/SVD/flux/block_analysis_it/paper/30_blocks/model_80_100"
    ckpt = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/20_blocks_compression/model_80/test-step00020000.safetensors"
    prompts = "/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json"

    # Statische Kompressions-Parameter
    comp_args = (
        "--double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \\\n"
        "  --single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 28 3 26 12 37 6 34 31 27 16 21 23 14 11 24 10 20 17 \\\n"
        "--single_comp_mod 0.0702 0.0575 0.1528 0.1446 \\\n"
        "--single_comp_mlp2 0.0702 0.0575 0.1528 0.1446 \\\n"
        "--single_comp_attn 0.0702 0.0575 0.1528 0.1446 \\\n"
        "--single_comp_mlp 0.0702 0.0575 0.1528 0.1446 \\\n"
        "--double_comp_img_mod 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_img_mlp 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_img_attn 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_txt_mod 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_txt_mlp 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_txt_attn 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_txt_proj 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506 \\\n"
        "--double_comp_img_proj 0.4241 0.4077 0.4091 0.4168 0.4619 0.3904 0.4119 0.3958 0.3532 0.3629 0.4207 0.3911 0.3748 0.3301 0.3418 0.3395 0.2413 0.3506"
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

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_block_analysis_svd_it.py \\
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
  --single_comp_mod_new 0.6 \\
  --single_comp_mlp2_new 0.6 \\
  --single_comp_attn_new 0.6 \\
  --single_comp_mlp_new 0.6 \\
  --double_comp_img_mod_new 0.6 \\
  --double_comp_img_mlp_new 0.6 \\
  --double_comp_img_attn_new 0.6 \\
  --double_comp_txt_mod_new 0.6 \\
  --double_comp_txt_mlp_new 0.6 \\
  --double_comp_txt_attn_new 0.6 \\
  --double_comp_txt_proj_new 0.6 \\
  --double_comp_img_proj_new 0.6 \\
  {comp_args}
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