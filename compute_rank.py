
in_features = 4096
out_features = 12288
compression_ratio=0.6

k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
print(k)