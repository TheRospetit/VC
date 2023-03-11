######################################################################
### VC i PSIV                                                      ###
### Lab 0 (basat en material de Gemma Rotger)                      ###
######################################################################
import os
# Hello! Welcome to the computer vision LAB.
import  itertools
import cv2
import numpy as np
import skimage
from skimage import io
from matplotlib import pyplot as plt

## PROBLEM 1  --------------------------------------------------
# TODO LEER LAS IMAGENES DE LA CARPETA INPUT DE HIGHWAY
files = os.listdir('./highway/input')   # CARGAMOS TODAS LAS IMÁGENES DE LA CARPETA

# TODO SELECCIONAR LAS IMAGENES PARA TRAIN Y TEST | DEL FICHERO 'in001051.jpg' al 'in001350.jpg' | 150 TRAIN - 150 TEST
# SEPARAMOS LAS IMAGENES PRECARGADAS ANTERIORMENTE
files_train = files[1050:1200]  # TRAIN
files_test = files[1200:1350]   # TEST

train = []
# train_rgb = []
for archivo in files_train:
    path = './highway/input/' + archivo
    image = skimage.io.imread(path, as_gray=True)     # LEO CADA IMAGEN EN GRIS
    # imageRGB = skimage.io.imread(path, as_gray=False)     # LEO CADA IMAGEN EN RGB
    train.append(image)             # GUARDO LOS VALORES DE CADA IMAGEN EN UN ARRAY
    # train_rgb.append(imageRGB)

test = []
for archivo in files_test:
    path = './highway/input/' + archivo
    image = skimage.io.imread(path, as_gray=True)  # LEO CADA IMAGEN EN GRIS
    # imageRGB = skimage.io.imread(path, as_gray=False)     # LEO CADA IMAGEN EN RGB
    test.append(image)


## PROBLEM 2 (+0.5)  --------------------------------------------------
# TODO: CALCULAR LA MEDIA Y LA DESVIACIÓN ESTÁNDAR
# CONSEGUIMOS LAS DIMENSIONES DE LA PRIMERA IMAGEN PARA ASÍ GENERAR LA NUEVA QUE SERÁ EL "FONDO"
# (SUPONGO QUE TODAS TIENEN EL MISMO TAMAÑO)
mean_train = np.mean(train, axis=0)
std_train = np.std(train, axis=0)


plt.imshow(mean_train, cmap='gray') # MOSTRAMOS LA IMAGEN GENERADA CON LA MEDIA
plt.show()


plt.imshow(std_train, cmap='gray')  # MOSTRAMOS LA IMAGEN GENERADA CON LA DESVIACIÓN ESTÁNDAR
plt.show()

"""mean_trainR = np.mean(train_rgb[:, :, 2])
mean_trainG = np.mean(train_rgb[:, :, 1])
mean_trainB = np.mean(train_rgb[:, :, 0])
mean_trainRGB = cv2.cvtColor(mean_train, cv2.COLOR_GRAY2RGB)
plt.imshow(mean_trainRGB)
plt.show()"""
# 

## PROBELM 3 (+1.0) --------------------------------------------------
# TODO. FRAGMENTAR COCHES RESTANDO EL MODELO DEL FONDO

imagenes_a_mirar = 1
threshold = 0.2
for image in train:
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.figure(2)
    bin = abs(image - mean_train)    # resto el fondo a la imagen
    plt.imshow(bin, cmap='gray')
    plt.show()
    bin = bin > threshold   # Binarizo la imagen
    plt.imshow(bin, cmap='gray')
    plt.show()
    if imagenes_a_mirar == 0:
        break
    else:
        imagenes_a_mirar -= 1



## PROBELM 4 (+1.0) &  PROBELM 5 (+2.0)--------------------------------------------------
# TODO. FRAGMENTAR COCHES CON UN MODELO MÁS ELABORADO
out = []
mean_test = np.mean(test, axis=0)

height, width = train[0].shape
size = (width, height)

alpha = 0.5
beta = 0.05

imagenes_a_mirar = 1

kernel = np.ones((3, 3), np.uint8)

video_bg_train = cv2.VideoWriter('resultsBgTrain.mp4', 0, 30, size)
video_bg_test = cv2.VideoWriter('resultsBgTest.mp4', 0, 30, size)

i = 0

for image in test:
    # plt.imshow(image, cmap='gray')
    # plt.show()
    i += 1
    image_name_train = './outputs/train/outputImage' + str(i) + '.png'
    image_name_test = './outputs/test/outputImage' + str(i) + '.png'

    binTrain = abs(image - mean_train)    # resto el fondo a la imagen
    binTest = abs(image - mean_test)
    # plt.imshow(bin, cmap='gray')
    # plt.show()

    # TODO OPEN Y CLOSE DE LA IMAGEN CON TAL DE TRATARLA Y ELIMINAR SONIDO
    # REALIZO UN OPEN PARA ELIMINAR EL SONIDO GENERADO POR LAS OJAS
    binTrain = cv2.morphologyEx(binTrain, cv2.MORPH_OPEN, kernel)

    # REALIZO UNOS CUANTOS CLOSE CON TAL DE AMPLIAR LOS COCHES Y ELIMINAR LOS ERRORES DE LA LUNA DELANTERA
    binTrain = cv2.morphologyEx(binTrain, cv2.MORPH_CLOSE, kernel)
    binTrain = cv2.morphologyEx(binTrain, cv2.MORPH_CLOSE, kernel)
    binTrain = cv2.morphologyEx(binTrain, cv2.MORPH_CLOSE, kernel)

    # OPEN TEST
    binTest = cv2.morphologyEx(binTest, cv2.MORPH_OPEN, kernel)
    # CLOSE TEST
    binTest = cv2.morphologyEx(binTest, cv2.MORPH_CLOSE, kernel)
    binTest = cv2.morphologyEx(binTest, cv2.MORPH_CLOSE, kernel)
    binTest = cv2.morphologyEx(binTest, cv2.MORPH_CLOSE, kernel)


    binTrain = binTrain > threshold * alpha + beta   # Binarizo la imagen
    binTest = binTest > threshold * alpha + beta  # Binarizo la imagen
    out.append(binTest)

    plt.imsave(image_name_train, np.uint8(binTrain * 255)) # Guardo la imagen obtenida
    plt.imsave(image_name_test, np.uint8(binTest * 255))  # Guardo la imagen obtenida

    video_bg_train.write(cv2.imread(image_name_train)) # Creo el video leyendo la imagen que acabo de guardar en la carpeta
    video_bg_test.write(cv2.imread(image_name_test))

    ### TODO NO ACABA DE FUNCIONAR LA EROSION Y DILATACION DESPUÉS DE BINARIZAR
    ## REALIZO UN OPENING CON TAL DE ELIMINAR EL SONIDO GENERADO POR LAS HOJAS EN LA IMAGEN
    # bin = cv2.erode(image, kernel, iterations=1)  # EROSIÓN
    # bin = cv2.dilate(bin, kernel, iterations=1) # DILATACIÓN
    # NO DEJA HACER EL OPENING DIRECTAMENTE
    # bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)

    if imagenes_a_mirar != 0:
        imagenes_a_mirar -= 1
        plt.imshow(binTrain, cmap='gray')
        plt.figure(2)
        plt.imshow(binTest, cmap='gray')
        plt.show()

## PROBELM 6 (+1.0) --------------------------------------------------
# TODO. EVALUA TUS RESULTADOS
# ADQUIERO LAS IMAGENES DE GT
groundtruth_test = os.listdir('./highway/groundtruth')
groundtruth_test = groundtruth_test[1200:1350]

out = np.uint8(out * 255)

# CARGO LAS IMAGENES DEL GT
gt = []
for archivo in groundtruth_test:
    path = './highway/groundtruth/' + archivo
    image = skimage.io.imread(path, as_gray=True)  # LEO CADA IMAGEN EN GRIS
    # imageRGB = skimage.io.imread(path, as_gray=False)     # LEO CADA IMAGEN EN RGB
    gt.append(image)

accuracys = []
for (test_image, ground_image) in zip(out, gt):
    accuracy = np.mean(ground_image == (test_image * 255))
    accuracys.append(accuracy)

accuracy = np.mean(accuracys) * 100
print('Accuracy: = ' + str(accuracy) + '%')


## PROBELM 8 (+1.0) --------------------------------------------------
# TODO. VELOCIDAD?
"""Si obtenemos tenemos la frecuencia de disparo de la cámara (el timpo que hay entre imágenes)
 y sabemos la distáncia de la carretera si también usamos el número de imagenes en las que aparece 
 podemos calcular la velocidad cómo distancia/tiempo dónde tiempo sería n.Imagenes * freq."""

## THE END -----------------------------------------------------------
# Well done, you finished this lab! Now, remember to deliver it 
# properly on Caronte.

# File name:
# lab1_NIU.zip
# (put matlab file lab0.m and python file lab1.py in the same zip file)
# Example lab1_1234567.zip
#
