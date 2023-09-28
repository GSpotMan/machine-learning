import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

# 读取图片
imag = plt.imread('butterfly.bmp')

image_array = np.array(imag)
# 将图像重塑为二维矩阵
reshaped_array = imag.reshape(-1, 437)

# 误差
errors=[]

for n_components in range(437, 0, -25):
    # 创建PCA对象，指定要保留的主成分数量
    pca = PCA(n_components=n_components)

    # 执行PCA降维
    compressed_array = pca.fit_transform(reshaped_array)

    # 重构压缩后的数据
    reconstructed_array = pca.inverse_transform(compressed_array)

    # 将数据形状转换回图像尺寸
    reconstructed_image_array = reconstructed_array.reshape(imag.shape)


    # 将重构的数据转换回图像
    reconstructed_image = reconstructed_image_array.astype(np.uint8)

    reconstruction_error = np.mean(np.square(reconstructed_image - image_array))
    errors.append(reconstruction_error)

    #图片展示，需要则打开
    plt.imshow(reconstructed_image)
    plt.show()

plt.plot(range(437, 0, -25),errors)
plt.xlabel('保存的维数',fontproperties='SimHei')
plt.ylabel('均方误差',fontproperties='SimHei')
plt.show()