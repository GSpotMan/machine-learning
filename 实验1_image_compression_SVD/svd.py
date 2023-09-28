import numpy as np
import matplotlib.pyplot as plt

# 读取图片
imag = plt.imread('butterfly.bmp')

# 图片深度是3，拆分为深度1的3个矩阵分别进行分解
red_channel = imag[:, :, 0]
green_channel = imag[:, :, 1]
blue_channel = imag[:, :, 2]

# 进行分解
U_red, S_red, V_red = np.linalg.svd(red_channel)
U_green, S_green, V_green = np.linalg.svd(green_channel)
U_blue, S_blue, V_blue = np.linalg.svd(blue_channel)

# 误差
errors=[]

# 按照奇异值数量倒序进行图片展示，每次减少25个奇异值
for k in range(len(U_red), 0, -25):
    # 分解三个深度
    compressed_S_red = np.diag(S_red[:k])
    compressed_S_green = np.diag(S_green[:k])
    compressed_S_blue = np.diag(S_blue[:k])

    # UV分解
    compressed_U_red = U_red[:, :k]
    compressed_V_red = V_red[:k, :]
    compressed_U_green = U_green[:, :k]
    compressed_V_green = V_green[:k, :]
    compressed_U_blue = U_blue[:, :k]
    compressed_V_blue = V_blue[:k, :]

    # 计算
    compressed_red_channel = np.dot(compressed_U_red, np.dot(compressed_S_red, compressed_V_red))
    compressed_green_channel = np.dot(compressed_U_green, np.dot(compressed_S_green, compressed_V_green))
    compressed_blue_channel = np.dot(compressed_U_blue, np.dot(compressed_S_blue, compressed_V_blue))

    # 合并三个深度
    compressed_imag = np.stack([compressed_red_channel, compressed_green_channel, compressed_blue_channel], axis=2)
    compressed_imag = compressed_imag.astype(np.uint8)

    # 误差分析
    diff = imag - compressed_imag
    mse = np.mean(np.square(diff))
    errors.append(mse)

    #图片展示,如需要可以将注释打开


    plt.imshow(compressed_imag)
    plt.show()

# # 误差分析
# plt.plot(range(len(U_red), 0, -25),errors)
# plt.xlabel('奇异值数',fontproperties='SimHei')
# plt.ylabel('均方误差',fontproperties='SimHei')
# plt.show()