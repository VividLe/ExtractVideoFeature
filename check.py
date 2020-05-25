import numpy as np


file1 = '/disk2/yangle/features/chunk_size_16_stride_1_cv2.INTER_LINEAR_pool/video_validation_0000154.npz'
data1 = np.load(file1)
feature1 = data1['feature']

file2 = '/disk2/yangle/features/video_validation_0000154.npz'
data2 = np.load(file2)
feature2 = data2['feature']

diff = abs(feature1 - feature2)
print(diff.min(), diff.max())
