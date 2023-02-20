######################################################################
### VC i PSIV                                                      ###
### Lab 0 (basat en material de Gemma Rotger)                      ###
######################################################################


# Hello! Welcome to the computer vision LAB. 
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt


## PROBLEM 1 (+0.5) --------------------------------------------------
# TODO. READ THE CAMERAMAN IMAGE. 
img_cameraman = cv2.imread('./img/cameraman.jpg')

## PROBLEM 2 (+0.5) --------------------------------------------------
# TODO: SHOW THE CAMERAMAN IMAGE
cv2.imshow('image', img_cameraman)
# 

## PROBELM 3 (+2.0) --------------------------------------------------
# TODO. Negative effect using a double for instruction

t = time.time()
# Your code goes here
im_neg = np.copy(img_cameraman)
height, width, _ = im_neg.shape

for i in range(0, height - 1):
    for j in range(0, width - 1):
        # Agafem el valor del píxel actual
        pixel = im_neg[i, j]

        # Canviem el color de cada paràmetre del píxel RGB i restem 255
        pixel[0] = 255 - pixel[0]
        pixel[1] = 255 - pixel[1]
        pixel[2] = 255 - pixel[2]

        # Substituïm el valor
        im_neg[i, j] = pixel

elapsed = time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')
plt.imshow(im_neg, 'gray')
plt.show()

# TODO. Negative effect using a vectorial instruction
cv2.imshow('image', img_cameraman)
t = time.time()
im_neg2 = 255 - img_cameraman
elapsed = time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')
plt.figure(2)
plt.imshow(im_neg2, 'gray')
plt.show()

# You sould see that results in figures 1 and 2 are the same but times
# are much different.

## PROBLEM 4 (+2.0) --------------------------------------------------

# TODO. Give some color (red, green or blue)
# r = ...
# g = ...
# b = ...

# im_col = np.zeros...
# im_col[:,:,0]=...
# ...
# plt.imshow(im_col)
# plt.show()

# im_col = np.dstack...
# plt.imshow(im_col)
# plt.show()


## PROBLEM 5 (+1.0) --------------------------------------------------

# cv2.imwrite ...
# cv2.imwrite ...
# cv2.imwrite ...

## PROBLEM 6 (+1.0) --------------------------------------------------

# lin128=im...
# plt.plot ...
# plt.show()

# lin128rgb=im...
# plt.plot ...
# plt.show()

## PROBLEM 7 (+2) ----------------------------------------------------

# TODO. Compute the histogram.
t=time.time()
# hist,bins = np.histogram...
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')
# plt.plot ...
# plt.show()

t=time.time()
# h=zeros(1,256);
# for ...
# plt.plot ...
# plt.show()
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

## PROBLEM 8 Binarize the image text.png (+1) ------------------------

# TODO. Read the image
# imtext = ...
# plt.imshow(imtext)
# plt.show()
# hist,bins = np.histogram...
# plt.plot...
# plt.show()

# TODO. Define 3 different thresholds
# th1 = ...
# th2 = ...
# th3 = ...

# TODO. Apply the 3 thresholds 5 to the image
# threshimtext1 = ...
# threshimtext2 = ...
# threshimtext3 = ...

# TODO. Show the original image and the segmentations in a subplot
fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0,0].remove()
ax[0,1].imshow(imtext)
ax[0,1].set_title('Original image')
ax[0,2].remove()
ax[1,0].imshow(threshimtext1)
ax[1,1].imshow(threshimtext2)
ax[1,2].imshow(threshimtext3)
plt.show()


## THE END -----------------------------------------------------------
# Well done, you finished this lab! Now, remember to deliver it 
# properly on Caronte.

# File name:
# lab0_NIU.zip 
# (put matlab file lab0.m and python file lab0.py in the same zip file)
# Example lab0_1234567.zip

















