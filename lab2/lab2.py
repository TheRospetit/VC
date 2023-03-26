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
from scipy.signal import fftconvolve
from skimage import io
from PIL import ImageChops
from matplotlib import pyplot as plt


def correlacion_cruzada(matriz1, matriz2):
    # Calcular la correlación cruzada
    corr = signal.correlate2d(matriz2, matriz1, mode='same') # Tienen que ser al revés en G se usa R como máscara
    # Encontrar todos los valores máximos
    max_vals = np.where(corr == np.max(corr))

    # Encontrar la posición más cercana al centro de la matriz
    centro = np.array(matriz1.shape) // 2   # 80 95
    distancias = [np.sqrt((i - centro[0]) ** 2 + (j - centro[1]) ** 2) for i, j in zip(max_vals[0], max_vals[1])]
    idx_max = np.argmin(distancias)
    max_pos = (max_vals[0][idx_max], max_vals[1][idx_max])

    return max_pos


def normxcorr2(template, image, mode="full"):
    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    max_vals = np.where(out == np.max(out))
    out = []
    out.append(max_vals[0][0] // 2)
    out.append(max_vals[1][0] // 2)
    return out




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

    # TODO. REESTRUCTURO LAS IMAGENES PARA QUE TODAS TENGAN EL MISMO TAMAÑO PARA CUANDO LAS TENGA QUE FUSIONAR
    height1, width1 = R.shape
    height2, width2 = G.shape
    height3, width3 = B.shape

    height_img = min(height1, height2, height3)
    R = R[-height_img:, :]
    G = G[-height_img:, :]
    B = B[-height_img:, :]

    # TODO. RECORTO UN PORCENTAJE LOS BORDES DE LA IMAGEN PARA EVITAR PROBLEMAS
    rec_height = int(height_img * 0.1)
    rec_width = int(width1 * 0.1)
    R = R[rec_height:-rec_height, rec_width:-rec_width]
    G = G[rec_height:-rec_height, rec_width:-rec_width]
    B = B[rec_height:-rec_height, rec_width:-rec_width]
    height_fin, width_fin = R.shape

    """plt.figure('Inicial')
    rgb_img = np.dstack((R, G, B))
    plt.imshow(rgb_img)

    # TODO. BUSCO EL PUNTO CON MAYOR CORRELACIÓN EL CUAL ESTÉ MÁS CERCA DEL CENTRO DE LA MATRIZ
    corr_rg = correlacion_cruzada(R, G)
    corr_rb = correlacion_cruzada(R, B)

    corr_rg_norm = normxcorr2(R, G)
    corr_rb_norm = normxcorr2(R, B)

    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B Im1 -> (85, 100)
    shift_g = [corr_rg[0] - R.shape[0] // 2, corr_rg[1] - R.shape[1] // 2]
    shift_b = [corr_rb[0] - R.shape[0] // 2, corr_rb[1] - R.shape[1] // 2]

    shift_g_norm = [corr_rg_norm[0] - R.shape[0] // 2, corr_rg_norm[1] - R.shape[1] // 2]
    shift_b_norm = [corr_rb_norm[0] - R.shape[0] // 2, corr_rb_norm[1] - R.shape[1] // 2]

    # TODO Desplazamos los canales G y B
    #g_shifted = np.roll(np.roll(G, shift_g[0]//2, axis=0), shift_g[1]//2, axis=1)
    M = np.float32([[1, 0, shift_g[0]], [0, 1, shift_g[1]]])
    g_shifted = cv2.warpAffine(G, M, (width_fin, height_fin))
    M = np.float32([[1, 0, shift_g_norm[0]], [0, 1, shift_g_norm[1]]])
    g_shifted_norm = np.roll(np.roll(G, shift_g_norm[0] // 2, axis=0), shift_g_norm[1] // 2, axis=1)
    # g_shifted = ndimage.shift(G, shift_g)
    # g_shifted = ImageChops.offset(G, shift_g[0], shift_g[1])
    # b_shifted = np.roll(np.roll(B, shift_b[0]//2, axis=0), shift_b[1]//2, axis=1)
    M = np.float32([[1, 0, -shift_b[0]],[0, 1, shift_b[1]]])
    b_shifted = cv2.warpAffine(B,M, (width_fin, height_fin))
    b_shifted_norm = np.roll(np.roll(B, shift_b_norm[0] // 2, axis=0), shift_b_norm[1] // 2, axis=1)

    # Juntamos los tres canales en una sola imagen
    #plt.figure('R')
    #plt.imshow(R)
    #plt.figure('G')
    #plt.imshow(g_shifted)
    #plt.figure('B')
    #plt.imshow(b_shifted)
    plt.figure('Correlació')
    rgb_new = np.dstack((R, g_shifted, b_shifted))
    plt.imshow(rgb_new)
    plt.figure('Correlació Normalitzada')
    rgb_new_norm = np.dstack((R, g_shifted_norm, b_shifted_norm))
    plt.imshow(rgb_new_norm)
    plt.show()"""

    # TODO TEST
    matriz = np.random.randint(0, 256, size=(height_fin, width_fin))
    matriz_movimiento1 = np.copy(matriz)
    matriz_movimiento2 = np.copy(matriz)

    # Aplicar un movimiento de 2 a la derecha y 3 hacia abajo a la primera matriz
    matriz_movimiento1 = np.roll(matriz_movimiento1, shift=2, axis=1)
    matriz_movimiento1 = np.roll(matriz_movimiento1, shift=3, axis=0)

    # Aplicar un movimiento de 5 a la izquierda y 1 hacia arriba a la segunda matriz
    matriz_movimiento2 = np.roll(matriz_movimiento2, shift=-5, axis=1)
    matriz_movimiento2 = np.roll(matriz_movimiento2, shift=-1, axis=0)

    matriz_recortada = matriz[5:-5, 5:-5]
    matriz_movimiento1_recortada = matriz_movimiento1[5:-5, 5:-5]
    matriz_movimiento2_recortada = matriz_movimiento2[5:-5, 5:-5]

    corr_rg = correlacion_cruzada(matriz_recortada, matriz_movimiento1_recortada)
    corr_rb = correlacion_cruzada(matriz_recortada, matriz_movimiento2_recortada)

    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B Im1 -> (85, 100)
    shift_g = [corr_rg[0] - R.shape[0] // 2, corr_rg[1] - R.shape[1] // 2]
    shift_b = [corr_rb[0] - R.shape[0] // 2, corr_rb[1] - R.shape[1] // 2]

    corr_rg_norm = normxcorr2(matriz_recortada, matriz_movimiento1_recortada)
    corr_rb_norm = normxcorr2(matriz_recortada, matriz_movimiento2_recortada)

    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B Im1 -> (85, 100)
    shift_g_norm = [corr_rg_norm[0] - R.shape[0] // 2, corr_rg_norm[1] - R.shape[1] // 2]
    shift_b_norm = [corr_rb_norm[0] - R.shape[0] // 2, corr_rb_norm[1] - R.shape[1] // 2]

    height_fina, width_fina = matriz_movimiento1_recortada.shape
    M = np.float32([[1, 0, (shift_g[0])], [0, 1, (shift_g[1])]])
    plt.figure('g_s')
    g_shifted = cv2.warpAffine(G, M, (width_fin, height_fin))
    plt.imshow(g_shifted)
    plt.show()
    M = np.float32([[1, 0, (shift_b[0])], [0, 1, (shift_b[1])]])
    b_shifted = cv2.warpAffine(matriz_movimiento2_recortada, M, (width_fina, height_fina))
    plt.figure('Correlació')
    rgb_new = np.dstack((matriz_recortada, g_shifted, b_shifted))
    plt.imshow(rgb_new)
    plt.show()


