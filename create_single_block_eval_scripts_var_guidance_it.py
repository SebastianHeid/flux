import os

def generate_slurm_scripts():
    # --- ZIELVERZEICHNIS (Kombiniert) ---
    output_script_dir = "/home/hd/hd_hd/hd_om233/SVD/flux/script/block_analysis/block_40/model_30"
    os.makedirs(output_script_dir, exist_ok=True)

    # --- GEMEINSAME KONFIGURATION ---
    base_output_path = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis_paper_it/block_40/model_30"
    base_log_path = "/home/hd/hd_hd/hd_om233/SVD/flux/block_analysis_paper_it/block_40/model_30"
    ckpt = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/40_blocks_compression/model_30/test-step00020000.safetensors"
    prompts = "/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json"

    # Statische Kompressions-Parameter
    comp_args = (
        "--double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \\\n"
        "--single_blocks_compress 0 33 30 2 5 13 22 1 15 18 8 19 28 3 26 12 37 6 34 31 27 16 21 23 14 11 24 10 20 17 32 25 7 9 4 36 35 \\\n"
        "--single_comp_mod 0.6864 0.6732 0.653 0.6616 0.6114 0.5922 0.7006 0.5272 0.5974 0.6812 0.6274 0.7133 0.6322 0.6111 0.5753 0.6472 0.7222 0.4733 0.4962 0.4862 0.5956 0.7055 0.7301 0.6871 0.6655 0.4672 0.7157 0.6174 0.6427 0.4521 0.5912 0.5641 0.5734 0.5704 0.6112 0.3596 0.2761 \\\n"
        "--single_comp_mlp2 0.6864 0.6732 0.653 0.6616 0.6114 0.5922 0.7006 0.5272 0.5974 0.6812 0.6274 0.7133 0.6322 0.6111 0.5753 0.6472 0.7222 0.4733 0.4962 0.4862 0.5956 0.7055 0.7301 0.6871 0.6655 0.4672 0.7157 0.6174 0.6427 0.4521 0.5912 0.5641 0.5734 0.5704 0.6112 0.3596 0.2761 \\\n"
        "--single_comp_attn 0.6864 0.6732 0.653 0.6616 0.6114 0.5922 0.7006 0.5272 0.5974 0.6812 0.6274 0.7133 0.6322 0.6111 0.5753 0.6472 0.7222 0.4733 0.4962 0.4862 0.5956 0.7055 0.7301 0.6871 0.6655 0.4672 0.7157 0.6174 0.6427 0.4521 0.5912 0.5641 0.5734 0.5704 0.6112 0.3596 0.2761 \\\n"
        "--single_comp_mlp 0.6864 0.6732 0.653 0.6616 0.6114 0.5922 0.7006 0.5272 0.5974 0.6812 0.6274 0.7133 0.6322 0.6111 0.5753 0.6472 0.7222 0.4733 0.4962 0.4862 0.5956 0.7055 0.7301 0.6871 0.6655 0.4672 0.7157 0.6174 0.6427 0.4521 0.5912 0.5641 0.5734 0.5704 0.6112 0.3596 0.2761 \\\n"
        "--double_comp_img_mod 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_img_mlp 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_img_attn 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_txt_mod 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_txt_mlp 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_txt_attn 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_txt_proj 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029 \\\n"
        "--double_comp_img_proj 0.8866 0.8668 0.8808 0.7899 0.8632 0.8736 0.866 0.8774 0.8756 0.8514 0.8035 0.7387 0.8653 0.8758 0.8655 0.8699 0.8429 0.8029"
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