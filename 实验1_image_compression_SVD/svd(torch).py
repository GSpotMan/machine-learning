# 做到中途发现不行，因为使用pytorch的话是把一个通道放在一列，最后reshape成了n*3列矩阵，对应三个奇异值


import numpy as np
import  torch
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片
# image = plt.imread('butterfly.bmp')
image = Image.open('butterfly.bmp')

# 转换为PyTorch张量
image_tensor = torch.tensor(np.array(image))
image_tensor = image_tensor.float()
# 将图像张量调整为二维形状（高度 x 宽度, 通道）
reshaped_tensor = image_tensor.view(-1,image_tensor.shape[2])
# 分解
u, s, v=torch.svd(reshaped_tensor)
# 误差
errors=[]

print(s.shape)

# # 按照奇异值数量倒序进行图片展示，每次减少25个奇异值
# for k in range(len(U_red), 0, -25):
#     # 分解三个深度
#     compressed_S_red = np.diag(S_red[:k])
#     compressed_S_green = np.diag(S_green[:k])
#     compressed_S_blue = np.diag(S_blue[:k])
#
#     # UV分解
#     compressed_U_red = U_red[:, :k]
#     compressed_V_red = V_red[:k, :]
#     compressed_U_green = U_green[:, :k]
#     compressed_V_green = V_green[:k, :]
#     compressed_U_blue = U_blue[:, :k]
#     compressed_V_blue = V_blue[:k, :]
#
#     # 计算
#     compressed_red_channel = np.dot(compressed_U_red, np.dot(compressed_S_red, compressed_V_red))
#     compressed_green_channel = np.dot(compressed_U_green, np.dot(compressed_S_green, compressed_V_green))
#     compressed_blue_channel = np.dot(compressed_U_blue, np.dot(compressed_S_blue, compressed_V_blue))
#
#     # 合并三个深度
#     compressed_imag = np.stack([compressed_red_channel, compressed_green_channel, compressed_blue_channel], axis=2)
#     compressed_imag = compressed_imag.astype(np.uint8)
#
#     # 误差分析
#     diff = imag - compressed_imag
#     mse = np.mean(np.square(diff))
#     errors.append(mse)
#
#     # 图片展示,如需要可以将注释打开
#     # plt.imshow(imag)
#     # plt.show()
#     #
#     # plt.imshow(compressed_imag)
#     # plt.show()
#
# # 误差分析
# plt.plot(range(len(U_red), 0, -25),errors)
# plt.xlabel('奇异值数（k）',fontproperties='SimHei')
# plt.ylabel('均方方差（mse）',fontproperties='SimHei')
# plt.show()
