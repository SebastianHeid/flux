import itertools

# Define all flags, in fixed order
flags_img = ["double_flag_img_attn", "double_flag_img_mlp", "double_flag_img_mod"]
flags_txt = ["double_flag_txt_attn", "double_flag_txt_mlp", "double_flag_txt_mod"]
flags = flags_img + flags_txt

base = (
    "python /home/hd/hd_hd/hd_om233/SVD/flux/flux_minimal_inference.py "
    "--double_blocks 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"
)
out_root = "/home/hd/hd_hd/hd_om233/SVD/flux/image/exp_new"

lines = ["#!/bin/bash", "# All combinations with img-first naming convention"]

for i in range(1, len(flags) + 1):
    for combo in itertools.combinations(flags, i):
        # Split into img and txt groups, preserving fixed order
        img = [f for f in flags_img if f in combo]
        txt = [f for f in flags_txt if f in combo]
        ordered_combo = img + txt

        # Build readable name: double_img_attn_mlp_mod_txt_attn_mlp_mod
        name_parts = []
        if img:
            name_parts.append("img_" + "_".join(x.split("_")[-1] for x in img))
        if txt:
            name_parts.append("txt_" + "_".join(x.split("_")[-1] for x in txt))
        name = "_".join(name_parts)
        outdir = f"{out_root}/double_{name}"

        # Construct command
        flags_str = " ".join(f"--{f}" for f in ordered_combo)
        cmd = f"{base} --output_dir {outdir} {flags_str}"
        lines.append(cmd)

# Write explicit script
with open("/home/hd/hd_hd/hd_om233/SVD/flux/script/run_all_combinations.sh", "w") as f:
    f.write("\n".join(lines))
