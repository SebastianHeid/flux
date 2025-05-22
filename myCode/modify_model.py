from myCode.place_holder_blocks import Identity, IdentityD


def modify_model(model, double_blocks, single_blocks):
    # Example modification: replace a specific layer with an Identity layer
    for idx in double_blocks:
        model.double_blocks[idx] = IdentityD(hidden_size=int(3072),mlp_ratio= 4.0, num_heads=int(24))
    for idx in single_blocks:
        model.single_blocks[idx] = Identity(hidden_size=int(3072), num_heads=int(24))
    return model