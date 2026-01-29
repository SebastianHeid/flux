import os

# Konfiguration - Neuer Pfad für die Double-Block Analyse
output_script_dir = "/home/hd/hd_hd/hd_om233/SVD/flux/script/block_analysis/double_block_analysis/model_90"
os.makedirs(output_script_dir, exist_ok=True)

# Template für die .sh Datei
template = """#!/bin/bash
#SBATCH --job-name=DB_analysis_{block_id}
#SBATCH --output=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_out/double_block_{block_id}.txt
#SBATCH --error=/home/hd/hd_hd/hd_om233/SVD/flux/script/std_error/double_block_{block_id}.txt

#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --mem=40G  
#SBATCH --time=4:00:00 
#SBATCH --export=NONE
#SBATCH --gres=gpu:A100:1

#--------------------
# JOB EXECUTION
#--------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux_train

module purge
module load devel/cuda/12.1

export LD_LIBRARY_PATH=$MODULEPATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python /home/hd/hd_hd/hd_om233/SVD/flux/flux_block_analysis_svd_it.py \\
  --double_block_list {block_id} \\
  --prompt_path /home/hd/hd_hd/hd_om233/partially_removal/100_prompts_laion.json \\
  --ckpt_path /gpfs/bwfor/work/ws/hd_om233-flux/flux/model_compression/paper_var_guidance_it/20_blocks_compression/model_90/test-step00020000.safetensors \\
  --output_dir /gpfs/bwfor/work/ws/hd_om233-flux/flux/image/block_analysis_paper_it/block_20/model_90/ \\
  --logdir /home/hd/hd_hd/hd_om233/SVD/flux/block_analysis_it_1k/double_block_{block_id}.txt \\
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
  --double_blocks_compress 13 16 5 3 17 8 15 6 14 18 7 4 10 12 11 9 1 0 \\
  --single_blocks_compress 0 33 \\
--single_comp_mod 0.0702 0.0575 \\
--single_comp_mlp2 0.0702 0.0575 \\
--single_comp_attn 0.0702 0.0575 \\
--single_comp_mlp 0.0702 0.0575 \\
--double_comp_img_mod 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_img_mlp 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_img_attn 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_txt_mod 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_txt_mlp 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_txt_attn 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_txt_proj 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
--double_comp_img_proj 0.3 0.2872 0.2745 0.2617 0.2489 0.2362 0.2234 0.2106 0.1979 0.1851 0.1724 0.1596 0.1468 0.1341 0.1213 0.1085 0.0958 0.083 \\
"""

# Generiere die 19 Dateien (Index 0 bis 18)
for i in range(19):
    file_content = template.format(block_id=i)
    file_path = os.path.join(output_script_dir, f"job_double_block_{i:02d}.sh")
    
    with open(file_path, "w") as f:
        f.write(file_content)

print(f"Fertig! 19 Skripte wurden im Ordner '{output_script_dir}' erstellt.")