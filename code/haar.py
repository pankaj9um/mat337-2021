import pywt
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

t = np.linspace(-20,20,1000)

sample = []
for x in t:
    sample.append(pow(x, 3))

fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
axs[0,0].plot(t,sample,color="blue",label="original")
axs[0,1].axis('off')
axs[0,0].set_title("original")

n = 5
w = "haar"

coeffs = pywt.wavedec(sample, wavelet=w, level=n)
array, slices = pywt.coeffs_to_array(coeffs)
sorted = np.sort(np.abs(array.reshape(-1)))

colors = ["red", "yellow", "green", "orange"]
pos = [ [1,0], [1, 1], [2, 0], [2, 1]]
i = 0
for keep in [0.9, 0.5, 0.1, 0.05]:
    # element at keep point
    limit = sorted[int(np.floor((1-keep)*len(sorted)))]

    # filter the array with limit value
    indices = np.abs(array) > limit

    # threshold small indices
    filtered = array * indices

    # invert
    coeffs_filtered = pywt.array_to_coeffs(filtered, slices, output_format='wavedec')
    approx = pywt.waverec(coeffs_filtered, wavelet=w)

    r = pos[i][0]
    c = pos[i][1]

    print("plotting at r=", r, ", c=", c)
    axs[r,c].plot(t,approx, color=colors[i])
    axs[r,c].set_title("keep "+str(keep))

    i = i + 1
    
plt.show()