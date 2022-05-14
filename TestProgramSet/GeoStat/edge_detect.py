# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/20/2022 10:44 AM
@file: edge_detect.py
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import roberts, sobel, scharr, prewitt

# cv2
img_cv = cv.imread("D:/MyProject/Python/PycharmProjects/DataProcessing/Microtremor/"
                   "TestProgramSet/GeoStat/picture.jpg", 0)
edge_cv = cv.Canny(img_cv, 100, 200)
plt.subplot(121)
plt.imshow(img_cv, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(edge_cv, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])
plt.show()

# skimage
im_rgb = imread("D:/MyProject/Python/PycharmProjects/DataProcessing/Microtremor/"
                "TestProgramSet/GeoStat/picture.jpg")
im_gray = rgb2gray(im_rgb)
edge_roberts = roberts(im_gray)
edge = feature.canny(im_gray, sigma=1)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].imshow(im_rgb, cmap='gray')
ax[0].set_title('image', fontsize=12)
ax[1].imshow(edge_roberts, cmap='gray')
ax[1].set_title('Canny filter', fontsize=12)
for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()
