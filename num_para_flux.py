from library import flux_models, flux_utils
loading_device="cpu"
ckpt_path = "/export/scratch/sheid/.cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/flux1-dev.safetensors"
is_schnell, model = flux_utils.load_flow_model(ckpt_path, None, loading_device)

for idx in range(38):
    block = model.single_blocks[idx]
    print("Number of parameters of single block ", idx, ": ", sum(p.numel() for p in block.parameters()))
    
for idx in range(19):
    block = model.double_blocks[idx]
    print("Number of parameters of double block ", idx, ": ", sum(p.numel() for p in block.parameters()))