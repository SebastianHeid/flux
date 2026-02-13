import os

def generate_slurm_scripts():
    # --- ZIELVERZEICHNIS (Kombiniert) ---
    output_script_dir = "/home/hd/hd_hd/hd_om233/SVD/flux/script/block_analysis/first_attempt/model_5_43"
    os.makedirs(output_script_dir, exist_ok=True)

    # --- GEMEINSAME KONFIGURATION ---
    base_output_path = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis_it/model_5_43"
    base_log_path = "/home/hd/hd_hd/hd_om233/SVD/flux/block_analysis_it/model_5_43"
    ckpt = "/gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/flux_train_var_guidance_it/model_5_43/test-step00020000.safetensors"
    prompts = "/home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json"

    # Statische Kompressions-Parameter
    comp_args = (
        "--double_blocks_compress 13 16 5 17 3 8 15 6 14 18 7 4 10 11 12 9 1 0 \\\n"
        "  --single_blocks_compress 0 33 30 2 5 13 1 15 18 8 19 3 28 26 12 37 6 16 23 22 14 31 17 21 20 11 24 27 25 4 34 7 10 9 32 36  \\\n"
        "--single_comp_mod 0.5150 0.4850 0.4100 0.3950 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.4857 0.4480 0.4723 0.5205 0.4941 0.4602 0.2786 0.4999 0.4844 0.4353 0.4351 0.2927 0.2816 0.4889 0.4031 0.4364 0.3951 0.4828 0.4506 0.3920 0.1964 0.1961 0.4941 0.3878 0.3390 0.3145 \\\n"
        "--single_comp_mlp2 0.5150 0.4850 0.4100 0.3950 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.4857 0.4480 0.4723 0.5205 0.4941 0.4602 0.2786 0.4999 0.4844 0.4353 0.4351 0.2927 0.2816 0.4889 0.4031 0.4364 0.3951 0.4828 0.4506 0.3920 0.1964 0.1961 0.4941 0.3878 0.3390 0.3145 \\\n"
        "--single_comp_attn 0.5150 0.4850 0.4100 0.3950 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.4857 0.4480 0.4723 0.5205 0.4941 0.4602 0.2786 0.4999 0.4844 0.4353 0.4351 0.2927 0.2816 0.4889 0.4031 0.4364 0.3951 0.4828 0.4506 0.3920 0.1964 0.1961 0.4941 0.3878 0.3390 0.3145 \\\n"
        "--single_comp_mlp 0.5150 0.4850 0.4100 0.3950 0.4322 0.4747 0.5113 0.4836 0.5786 0.3093 0.4857 0.4480 0.4723 0.5205 0.4941 0.4602 0.2786 0.4999 0.4844 0.4353 0.4351 0.2927 0.2816 0.4889 0.4031 0.4364 0.3951 0.4828 0.4506 0.3920 0.1964 0.1961 0.4941 0.3878 0.3390 0.3145 \\\n"
        "--double_comp_img_mod 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_img_mlp 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_img_attn 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_txt_mod 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_txt_mlp 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_txt_attn 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_txt_proj 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029 \\\n"
        "--double_comp_img_proj 0.7967 0.8113 0.7936 0.7074 0.7587 0.7773 0.7290 0.7674 0.7729 0.7593 0.7597 0.6985 0.8019 0.7429 0.7645 0.7610 0.7484 0.7029"
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