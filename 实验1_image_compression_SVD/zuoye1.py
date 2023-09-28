import numpy as np
import matplotlib.pyplot as plt


#reading and converting the image
#在此处填入合适的内容
pic = open("butterfly.bmp")

# decomposing the image using singular value decomposition
#在此处填入合适的内容


# Using different number of singular values (diagonal of S) to compress and
# reconstruct the image
dispEr = []
numSVals = []
for N in np.arange(5,300+25,25).reshape(-1):
    # store the singular values in a temporary var
    C = S
    # discard the diagonal values not required for compression
    C[np.arange[N + 1,end()+1],:] = 0
    C[:,np.arange[N + 1,end()+1]] = 0
    # Construct an Image using the selected singular values
    D = U * C * np.transpose(V)
    # display and compute error
    figure
    buffer = sprintf('Image output using %d singular values',N)
    imshow(uint8(D))
    plt.title(buffer)
    error = sum(sum((inImageD - D) ** 2))
    # store vals for display
    dispEr = np.array([[dispEr],[error]])
    numSVals = np.array([[numSVals],[N]])

# dislay the error graph
figure
plt.title('Error in compression')
plt.plot(numSVals,dispEr)
grid('on')
plt.xlabel('Number of Singular Values used')
plt.ylabel('Error between compress and original image')