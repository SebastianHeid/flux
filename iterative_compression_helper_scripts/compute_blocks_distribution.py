import numpy as np 

def compute_preserve_rank( in_features, out_features, compression_ratio: float):
        k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
        return k



in_features=1152
out_attn_features = 1152
out_mlp_features=1152*4

s = []
old_comp  = 0.5, 0.53, 0, 0.44, 0.455, 0, 0, 0.425, 0.47, 0, 0, 0, 0, 0, 0.365, 0, 0, 0, 0.35, 0.38, 0, 0, 0, 0, 0, 0.335, 0, 0.32, 0, 0, 0.305, 0.29, 0.245, 0, 0, 0.215, 0.23, 0.41, 0.515, 0.485
print(len(old_comp))
for idx in range(len(old_comp)):
    if (0.5 - idx * 0.015 < 0) or idx > 40:
        print("Negative comp after step:", idx)
        break
    compression_ratio = 1- (1-old_comp[idx]) * (1- (0.5 - idx * 0.015))
    s.append(round(compression_ratio,5))

print(len(s))
print(s)
    #compute_preserve_rank( in_features, out_features, compression_ratio)
    
new_comp = [s[i] - old_comp[i] for i in range(len(s))]

print("total new compression: ", sum(new_comp))



attn_rank = [compute_preserve_rank(in_features, out_attn_features, e) for e in s]
mlp_rank = [compute_preserve_rank(in_features, out_mlp_features, e) for e in s]

print("Attn: ", attn_rank)
print("MLP: ", mlp_rank)