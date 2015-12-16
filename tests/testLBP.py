from localbinarypatterns import LocalBinaryPatterns
import cv2
import numpy as np

import matplotlib.pyplot as plt

def to32F(np_histogram_output):
    return np_histogram_output.astype('float32')

desc = LocalBinaryPatterns(24, 8)

def extactLBP(inputimg):
	image = cv2.imread(inputimg)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return desc.describe(gray)

image1 = extactLBP("face-1450033798.jpg")
image2 = extactLBP("face-1450033813.jpg")
image3 = extactLBP("face-1449002401.jpg")

#plt.hist(image1)
#plt.hist(image2)

plt.subplot(221), plt.hist(image1)
plt.subplot(222), plt.hist(image2)
plt.subplot(223), plt.hist(image3)
#plt.show()
print cv2.compareHist(to32F(image2), to32F(image1), 2)
#print cv2.compareHist(np_hist_to_cv(image1), np_hist_to_cv(image2) , 0)
