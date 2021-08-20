import pywt
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

image = imread(os.path.join(".", "data", "IMG_4522.jpg"))
imageGray = np.mean(image, -1)

fig, axs = plt.subplots(3, 2)
axs[0,0].axis('off')
axs[0,0].imshow(imageGray.astype('uint8'), cmap='gray_r')
axs[0,0].set_title("original")
axs[0,1].axis('off')

n = 4
w = "db1"

coeffs = pywt.wavedec2(imageGray, wavelet=w, level=n)

array, slices = pywt.coeffs_to_array(coeffs)

sorted = np.sort(np.abs(array.reshape(-1)))

pos = [ [1,0], [1, 1], [2, 0], [2, 1]]
i = 0
for keep in [0.1, 0.05, 0.01, 0.005]:
    # element at keep point
    limit = sorted[int(np.floor((1-keep)*len(sorted)))]

    # filter the array with limit value
    indices = np.abs(array) > limit

    # threshold small indices
    filtered = array * indices

    # invert
    coeffs_filtered = pywt.array_to_coeffs(filtered, slices, output_format='wavedec2')

    approx = pywt.waverec2(coeffs_filtered, wavelet=w)

    r = pos[i][0]
    c = pos[i][1]

    print("plotting at r=", r, ", c=", c)
    axs[r,c].axis('off')
    axs[r,c].imshow(approx.astype('uint8'), cmap='gray_r')
    axs[r,c].set_title("keep "+str(keep))

    i = i + 1
    
plt.show()