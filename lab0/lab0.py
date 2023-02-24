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

elapsed = time.time() - t
print('Elapsed time is ' + str(elapsed) + ' seconds')
plt.imshow(im_neg, 'gray')
plt.show()

# TODO. Negative effect using a vectorial instruction
# cv2.imshow('image', img_cameraman)
t = time.time()
im_neg2 = 255 - img_cameraman
elapsed = time.time() - t
print('Elapsed time is ' + str(elapsed) + ' seconds')
plt.figure(2)
plt.imshow(im_neg2, 'gray')
plt.show()

# You sould see that results in figures 1 and 2 are the same but times
# are much different.

## PROBLEM 4 (+2.0) --------------------------------------------------

# TODO. Give some color (red, green or blue)
r = img_cameraman[:, :, 2]
g = im_neg2[:, :, 1]
b = img_cameraman[:, :, 0]

# im_col = np.zeros(img_cameraman.shape, dtype="uint8")
# im_col[:, :, 0] = b
# im_col[:, :, 1] = g
# im_col[:, :, 2] = r
# plt.imshow(im_col)
# plt.show()

im_col = np.dstack((b, g, r))
plt.imshow(im_col)
plt.show()

## PROBLEM 5 (+1.0) --------------------------------------------------


cv2.imwrite('imagenPNG.png', im_col)
cv2.imwrite('imagenBMP.bmp', im_col)
cv2.imwrite('imagenTIF.tif', im_col)
cv2.imwrite('imagenJPG.jpg', im_col)
# cv2.imwrite ...
# cv2.imwrite ...

## PROBLEM 6 (+1.0) --------------------------------------------------

lin128 = img_cameraman[127, :, :]
mean = np.mean(lin128)
plt.axhline(y=mean, color='r', linestyle='-')
plt.plot(lin128)
plt.show()

plt.clf()
lin128rgb = im_col[127, :, :]
mean2 = np.mean(lin128rgb)
plt.axhline(y=mean2, color='r', linestyle='-')
plt.plot(lin128rgb)
plt.show()

## PROBLEM 7 (+2) ----------------------------------------------------

# TODO. Compute the histogram.
# cv2.imshow('image', img_cameraman)
img_cameraman_grey = cv2.cvtColor(img_cameraman, cv2.COLOR_BGR2GRAY)
t = time.time()
# hist, bins = np.histogram(img_cameraman_grey, bins=256, range=(0, 1))
plt.title("Histograma de Cameraman")
plt.xlabel("Valor de gris")
plt.ylabel("Nombre de pixels")

plt.hist(img_cameraman_grey.ravel(), 256)
plt.show()
elapsed = time.time() - t
print('Elapsed time is ' + str(elapsed) + ' seconds')
# print(hist)



t = time.time()
h = np.zeros((1, 256))
height, width = img_cameraman_grey.shape
for i in range(0, height):
    for j in range(0, width):
        # Agafem el valor del píxel actual
        pixel = img_cameraman_grey[i, j]
        h[0][pixel] = h[0][pixel] + 1
plt.plot(h[0])
plt.show()
elapsed = time.time() - t
print('Elapsed time is ' + str(elapsed) + ' seconds')

## PROBLEM 8 Binarize the image text.png (+1) ------------------------

# TODO. Read the image
imtext = cv2.imread('./img/alice.jpg')
plt.imshow(imtext)
plt.show()
imtext_grey = cv2.cvtColor(imtext, cv2.COLOR_BGR2GRAY)
plt.hist(imtext.ravel(), 256)
plt.show()

# TODO. Define 3 different thresholds
th1 = 200
th2 = 150
th3 = 230


# TODO. Apply the 3 thresholds 5 to the image

threshimtext1 = np.copy(imtext_grey)
threshimtext2 = np.copy(imtext_grey)
threshimtext3 = np.copy(imtext_grey)
threshimtext1 = np.where(threshimtext1 > th1, 1, 0)
threshimtext2 = np.where(threshimtext2 > th2, 1, 0)
threshimtext3 = np.where(threshimtext3 > th3, 1, 0)

"""height, width = imtext_grey.shape
for i in range(0, height):
    for j in range(0, width):
        # Agafem el valor del píxel actual
        pixel = imtext_grey[i, j]
        if pixel > th1:
            threshimtext1[i, j] = 1
        else:
            threshimtext1[i, j] = 0
        if pixel > 150:
            threshimtext2[i, j] = 1
        else:
            threshimtext2[i, j] = 0
        if pixel > 230:
            threshimtext3[i, j] = 1
        else:
            threshimtext3[i, j] = 0"""

# TODO. Show the original image and the segmentations in a subplot
fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0, 0].remove()
ax[0, 1].imshow(imtext)
ax[0, 1].set_title('Original image')
ax[0, 2].remove()
ax[1, 0].imshow(threshimtext1)
ax[1, 1].imshow(threshimtext2)
ax[1, 2].imshow(threshimtext3)
plt.show()

## THE END -----------------------------------------------------------
# Well done, you finished this lab! Now, remember to deliver it 
# properly on Caronte.

# File name:
# lab0_NIU.zip 
# (put matlab file lab0.m and python file lab0.py in the same zip file)
# Example lab0_1234567.zip
#
