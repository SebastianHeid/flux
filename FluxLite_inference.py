import torch
from diffusers import FluxPipeline

base_model_id = "Freepik/flux.1-lite-8B-alpha"
torch_dtype = torch.bfloat16
device = "cuda"

# Load the pipe
model_id = "Freepik/flux.1-lite-8B-alpha"
pipe = FluxPipeline.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)

# Calculate the number of parameters
total_params = sum(p.numel() for p in pipe.transformer.parameters())
trainable_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)

print(f"Total parameters in Transformer: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# # Inference
# prompts = [
#     "A photograph of a majestic Bengal tiger in a lush jungle, with soft sunlight filtering through the canopy, detailed fur, and sharp focus on its eyes.",
#     "photo of peaceful winter landscape, serene winter scenery, snow-covered path, leafless trees, overcast sky, winter forest, frozen stream, icy water, subtle blue hues, delicate snow textures, soft light, gentle snowfall, quiet atmosphere, calming environment, natural setting, tranquil riverside, bare branches, rustic road, snow-dusted bushes, delicate frost, seasonal beauty, detailed winter flora, tranquil nature scene, cold season ambiance, muted colors, soft textures",
#     "A close-up portrait of an elderly man with a weathered face, showing every wrinkle and detail, against a simple, dark background, shot with a shallow depth of field.",
#    "A bustling city street in the heart of a modern metropolis, filled with people walking on sidewalks, cars and buses in traffic, neon signs and billboards glowing, skyscrapers towering above, reflections on wet asphalt, dynamic lighting and cinematic atmosphere, photographed at street level during rush hour."
#    "A candid photo of a person laughing, with a genuine expression, in a cozy coffee shop, with warm, inviting lighting and a soft focus on the background.",
#    "portrait of a joker like the joker in batman, he is wearing the joker outfit and makeup. He holds poker cards in his hand, glitch effects cinematic lighting, film scene, optimized lighting, ray tracing, sharpened image, film grain, super high resolution 8k ",
#    "Ultra realistic photographyMale lion roaring in front of a savanna tree National Geographic Photo, sundowner, aggressiv, Sony \u03b17 III, F 1.2 v 5",
#    "The sharp dressed black guy sits at a table in a dimly lit jazz club, his crisp black suit perfectly tailored to his athletic frame. He wears a sleek silver watch on his wrist that catches the light as he moves. Beside him sits his stunning white wife, her blonde hair swept up in an elegant bun, wearing a formfitting black dress that accentuates her curves. As they watch the band play, the mans foot taps in time to the music while his wife sways gently in her seat. The atmosphere is lively yet intimate, the perfect backdrop for a night out on the town. The jazz musicians on stage are equally stylish, their suits and instruments gleaming under the dim lights. The black guy leans in to whisper something in his wifes ear, a smile spreading across her face. They clink their glasses together in a toast, enjoying the moment as the jazz music fills the room.",
#     "a full body photo portrait of a Mexican beautiful girl during the Mexican revolution in 1914 after a battle, she has glowing eyes and dark hair, she is wearing ammo belts, ultra realistic, cinematic lighting, dust particles, light particles, professional portrait, hyper detailed, 8k, sony a7iii, sigma lens, professional color grade ",
# ]

# guidance_scale = 3.5  # Keep guidance_scale at 3.5
# n_steps = 50
# seed = 42

# with torch.inference_mode():
#     for idx, prompt in enumerate(prompts):
#         image = pipe(
#             prompt=prompt,
#             generator=torch.Generator(device="cpu").manual_seed(seed),
#             num_inference_steps=n_steps,
#             guidance_scale=guidance_scale,
#             height=1024,
#             width=1024,
#         ).images[0]
#         image.save("/home/hd/hd_hd/hd_om233/SVD/images/FluxLite/image_"+str(idx)+".png")
