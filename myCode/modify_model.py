from typing import Tuple, Union

import torch
import torch.nn as nn
from myCode.place_holder_blocks import (
    DoubleStreamBlockPruned,
    Identity,
    IdentityD,
    SingleStreamBlockPruned,
)


def modify_model(model,
                 double_blocks,
                 single_blocks, 
                 single_blocks_comp = [],
                 double_blocks_comp = [],
                 single_flag_attn=False,
                 single_flag_mlp=False,
                 single_flag_mlp2=False,
                 single_flag_mod=False,
                 single_rank_mod=256,
                 single_rank_mlp2=512,
                 single_rank_attn=512,
                 single_rank_mlp=512,
                 double_flag_img_attn=False,
                 double_flag_txt_attn=False,
                 double_flag_img_mlp=False,
                 double_flag_txt_mlp=False,
                 double_flag_img_mod=False,
                 double_flag_txt_mod=False,
                 double_rank_img_mod=256,
                 double_rank_img_mlp=512,
                 double_rank_img_attn=512,
                 double_rank_txt_mod=256,
                 double_rank_txt_mlp=512,
                 double_rank_txt_attn=512 
                 ):
    # Example modification: replace a specific layer with an Identity layer
    for idx in double_blocks:
        model.double_blocks[idx] = IdentityD(hidden_size=int(3072),mlp_ratio= 4.0, num_heads=int(24))
    for idx in single_blocks:
        model.single_blocks[idx] = Identity(hidden_size=int(3072), num_heads=int(24))
    print("Modify singel blocks: ", single_blocks)
    print(double_blocks)
    for idx in single_blocks_comp:
        model.single_blocks[idx] = SingleStreamBlockPruned(model.single_blocks[idx],
                                                           rank_attn=single_rank_attn,
                                                           rank_mlp=single_rank_mlp,
                                                           rank_mlp2=single_rank_mlp2,
                                                           rank_mod=single_rank_mod,
                                                           flag_mod=single_flag_mod,
                                                           flag_attn=single_flag_attn,
                                                           flag_mlp=single_flag_mlp,
                                                           flag_mlp2=single_flag_mlp2)
    for idx in double_blocks_comp:
        model.double_blocks[idx] = DoubleStreamBlockPruned( model.double_blocks[idx],
                                                            rank_attn_img=double_rank_img_attn,
                                                            rank_attn_txt=double_rank_txt_attn,
                                                            rank_img_mlp_in=double_rank_img_mlp,
                                                            rank_img_mlp_out=double_rank_img_mlp,
                                                            rank_txt_mlp_in=double_rank_txt_mlp,
                                                            rank_txt_mlp_out=double_rank_txt_mlp,
                                                            rank_img_mod=double_rank_img_mod,
                                                            rank_txt_mod=double_rank_txt_mod,
                                                            flag_img_attn=double_flag_img_attn,
                                                            flag_txt_attn=double_flag_txt_attn,
                                                            flag_img_mlp=double_flag_img_mlp,
                                                            flag_txt_mlp=double_flag_txt_mlp,
                                                            flag_img_mod=double_flag_img_mod,
                                                            flag_txt_mod=double_flag_txt_mod)
    return model