# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%                              VC                                %%%
# %%%                             Lab 2                              %%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
# Hello! Welcome to the computer vision LAB.
import itertools
import cv2
import numpy as np
from scipy import signal
from scipy import ndimage
import skimage
from skimage import io
from matplotlib import pyplot as plt


def correlacion_cruzada(matriz1, matriz2):
    # Calcular la correlación cruzada
    corr = signal.correlate2d(matriz1, matriz2, mode='same')

    # Encontrar todos los valores máximos
    max_vals = np.where(corr == np.max(corr))

    # Encontrar la posición más cercana al centro de la matriz
    centro = np.array(matriz1.shape) // 2
    distancias = [np.sqrt((i - centro[0]) ** 2 + (j - centro[1]) ** 2) for i, j in zip(max_vals[0], max_vals[1])]
    idx_max = np.argmin(distancias)
    max_pos = (max_vals[0][idx_max], max_vals[1][idx_max])

    return max_pos


## PROBLEM 1  --------------------------------------------------
# TODO LEER LAS IMAGENES DE LA CARPETA
files = os.listdir('./imatges/petites')

# TODO. CARGAMOS UNA PRIMERA IMAGEN PARA HACER TODAS LAS PRUEBAS
for imagen in files:
    path = './imatges/petites/' + imagen
    image = skimage.io.imread(path, as_gray=False)
    # image = skimage.io.imread('peppers.png', as_gray=False)

    # TODO. SEPARO LA IMAGEN EN TRES COMPONENTES
    height, width = image.shape
    tercera_parte = height // 3  # Si le pongo los dos divisores lo que hago es que pille el int entero y no el float
    R = image[:tercera_parte, :]
    G = image[tercera_parte:tercera_parte * 2, :]
    B = image[tercera_parte * 2:, :]

    # plt.figure('R')
    # plt.imshow(R)
    # plt.figure('G')
    # plt.imshow(G)
    # plt.figure('B')
    # plt.imshow(B)
    # plt.show()

    # TODO. REESTRUCTURO LAS IMAGENES PARA QUE TODAS TENGAN EL MISMO TAMAÑO PARA CUANDO LAS TENGA QUE FUSIONAR
    height1, width1 = R.shape
    height2, width2 = G.shape
    height3, width3 = B.shape

    height_img = min(height1, height2, height3)
    R = R[-height_img:, :]
    G = G[-height_img:, :]
    B = B[-height_img:, :]

    # TODO. RECORTO UN PORCENTAJE LOS BORDES DE LA IMAGEN PARA EVITAR PROBLEMAS
    """rec_height = int(height_img * 0.1)
    rec_width = int(width1 * 0.1)
    R = R[rec_height:-rec_height, rec_width:-rec_width]
    G = G[rec_height:-rec_height, rec_width:-rec_width]
    B = B[rec_height:-rec_height, rec_width:-rec_width]"""

    plt.figure('Inicial')
    rgb_img = np.dstack((R, G, B))
    plt.imshow(rgb_img)
    # plt.figure('R')
    # plt.imshow(R)
    # plt.figure('G')
    # plt.imshow(G)
    # plt.figure('B')
    # plt.imshow(B)
    # plt.show()

    # TODO. BUSCO EL PUNTO CON MAYOR CORRELACIÓN EL CUAL ESTÉ MÁS CERCA DEL CENTRO DE LA MATRIZ
    corr_rg = correlacion_cruzada(R, G)
    corr_rb = correlacion_cruzada(R, B)

    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B (85, 98)
    shift_g = [corr_rg[0] - R.shape[0] // 2, corr_rg[1] - R.shape[1] // 2]
    shift_b = [corr_rb[0] - R.shape[0] // 2, corr_rb[1] - R.shape[1] // 2]

    # Desplazamos los canales G y B
    # g_shifted = np.roll(np.roll(G, shift_g[0], axis = 0), shift_g[1], axis=1)
    g_shifted = ndimage.shift(G, shift_g)
    # b_shifted = np.roll(np.roll(B, shift_b[0], axis = 0), shift_b[1], axis=1)
    b_shifted = ndimage.shift(B, shift_b)

    # Juntamos los tres canales en una sola imagen
    plt.figure('R')
    plt.imshow(R)
    plt.figure('G')
    plt.imshow(g_shifted)
    plt.figure('B')
    plt.imshow(b_shifted)
    plt.figure('final')
    rgb_new = np.dstack((R, g_shifted, b_shifted))
    plt.imshow(rgb_new)
    plt.show()

