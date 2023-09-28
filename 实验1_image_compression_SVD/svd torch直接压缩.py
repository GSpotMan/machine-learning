import numpy as np
import matplotlib.pyplot as plt
import torch
# 读取图片
imag = plt.imread('butterfly.bmp')

tensor_image = torch.tensor(imag)
reshaped_image = tensor_image.reshape(-1,437)
float_image = reshaped_image.float()
U, S, V = torch.svd(float_image)

# 误差
errors=[]

# 按照奇异值数量倒序进行图片展示，每次减少25个奇异值
for k in range(len(S), 0, -25):
    # 分解
    
    compressed_S = np.diag(S[:k])
    compressed_U = U[:, :k]
    compressed_V= V[:k, :]


    # 计算
    reconstructed_array = np.dot(compressed_U, np.dot(compressed_S, compressed_V))
    

    # 回滚
    reconstructed_image_array = reconstructed_array.reshape(imag.shape)
    reconstructed_image = reconstructed_image_array.astype(np.uint8)
    
    # 误差分析
    diff = imag - reconstructed_image
    mse = np.mean(np.square(diff))
    errors.append(mse)

    # 图片展示,如需要可以将注释打开
    plt.imshow(imag)
    plt.show()
    
    plt.imshow(reconstructed_image)
    plt.show()

# 误差分析
# plt.plot(range(len(S), 0, -25),errors)
# plt.xlabel('奇异值数',fontproperties='SimHei')
# plt.ylabel('均方误差',fontproperties='SimHei')
# plt.show()
