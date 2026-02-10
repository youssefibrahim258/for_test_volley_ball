import numpy as np

path = r"Y:\02_DL\200_projects\for_test_volley_ball\data_set\features\image-level\resnet\7\38025.npz"

data = np.load(path)

features = data['features']   # (12, 2048)

# Max Pooling across players
pooled_feature = np.max(features, axis=0)

print("Original shape:", features.shape)
print("Pooled shape:", pooled_feature.shape)

data.close()

print(pooled_feature)


# Y:\02_DL\200_projects\for_test_volley_ball\scripts\test_feature.py
# python -m scripts.test_feature