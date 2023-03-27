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


def correlacion_cruzada(matriz1, matriz2): # La correlación debería ser lo mismo que la convolución con la máscara rotada 180 grados pero la función correlate2d() ya se encarga de ello
    # Calcular la correlación cruzada
    corr = signal.correlate2d(matriz2, matriz1, mode='same') # Tienen que ser al revés en G se usa R como máscara

    # Crea una figura 3D
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Crea las coordenadas X e Y para la matriz
    x, y = np.meshgrid(np.arange(corr.shape[0]), np.arange(corr.shape[1]))
    # Aplana la matriz y convierte los valores en una lista
    z = corr.flatten()
    # Crea una barra de color para los valores de la matriz
    colores = plt.cm.jet(z / np.amax(z))
    # Dibuja la matriz en 3D
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, color=colores)
    # Configura los límites de los ejes
    ax.set_xlim(0, corr.shape[0])
    ax.set_ylim(0, corr.shape[1])
    ax.set_zlim(0, np.amax(z))
    # Muestra la figura
    path = './outputs/zz-01_Correlacio.png'
    plt.savefig(path)"""
    # plt.show()
    # conv = signal.convolve2d(matriz2, np.flip(np.flip(matriz1, 0), 1))
    # Encontrar todos los valores máximos
    max_vals = np.where(corr == np.max(corr))
    # max_vals_conv = np.where(conv == np.max(conv))

    # Encontrar la posición más cercana al centro de la matriz
    centro = np.array(matriz1.shape) // 2   # 80 95
    distancias = [np.sqrt((i - centro[0]) ** 2 + (j - centro[1]) ** 2) for i, j in zip(max_vals[0], max_vals[1])]
    # distancias2 = [np.sqrt((i - centro[0]) ** 2 + (j - centro[1]) ** 2) for i, j in zip(max_vals_conv[0], max_vals_conv[1])]
    idx_max = np.argmin(distancias)
    # idx_max2 = np.argmin(distancias2)
    max_pos = (max_vals[0][idx_max], max_vals[1][idx_max])
    # max_pos2 = (max_vals_conv[0][idx_max2], max_vals_conv[1][idx_max2])

    return max_pos


def normxcorr2(image, template, mode="full"):
    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Para hacer la correlación paso a paso debemos hacer la convolución pero rotando la máscara primero de izquierda a derecha u luego de arriba a abajo
    # Y después de esta rotación hacer la convolución
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

    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Crea las coordenadas X e Y para la matriz
    x, y = np.meshgrid(np.arange(out.shape[0]), np.arange(out.shape[1]))
    # Aplana la matriz y convierte los valores en una lista
    z = out.flatten()
    # Crea una barra de color para los valores de la matriz
    colores = plt.cm.jet(z / np.amax(z))
    # Dibuja la matriz en 3D
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, color=colores)
    # Configura los límites de los ejes
    ax.set_xlim(0, out.shape[0])
    ax.set_ylim(0, out.shape[1])
    ax.set_zlim(0, np.amax(z))
    # Muestra la figura
    path = './outputs/zz-02_Correlacio_Norm.png'
    plt.savefig(path)"""

    max_vals = np.where(out == np.max(out))
    out = []
    out.append(max_vals[0][0] // 2)
    out.append(max_vals[1][0] // 2)

    return out


def fourier_transform(máscara, imagen): # En Fourier una convolución o correlación tenemos que transformar las señales a fourier, multiplicarlas y seguidamente hacer la trnsformada inversa
    # Calcular la correlación cruzada
    fft_mask = np.fft.fft2(máscara)
    fft_image = np.fft.fft2(imagen)
    fft_conjunto = fft_image * np.conj(fft_mask)
    fft_conjunto = np.fft.fftshift(np.fft.ifft2(fft_conjunto))
    # plt.figure('FFT')
    """fft = np.abs(np.fft.ifft2(fft_image * np.conj(fft_mask)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Crea las coordenadas X e Y para la matriz
    x, y = np.meshgrid(np.arange(fft.shape[0]), np.arange(fft.shape[1]))
    # Aplana la matriz y convierte los valores en una lista
    z = fft.flatten()
    # Crea una barra de color para los valores de la matriz
    colores = plt.cm.jet(z / np.amax(z))
    # Dibuja la matriz en 3D
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, color=colores)
    # Configura los límites de los ejes
    ax.set_xlim(0, fft.shape[0])
    ax.set_ylim(0, fft.shape[1])
    ax.set_zlim(0, np.amax(z))
    # Muestra la figura
    path = './outputs/zz-03_FFT.png'
    plt.savefig(path)"""
    # plt.imshow(fft)
    # plt.show()
    shape = np.unravel_index(np.argmax(fft_conjunto), fft_conjunto.shape)


    return shape

def fourier_norm(máscara, imagen): # En Fourier una convolución o correlación tenemos que transformar las señales a fourier, multiplicarlas y seguidamente hacer la trnsformada inversa
    # Calcular la correlación cruzada
    fft_mask = np.fft.fft2(máscara)
    fft_mask = fft_mask/abs(fft_mask)
    fft_image = np.fft.fft2(imagen)
    fft_image = fft_image / abs(fft_image)
    fft_conjunto = fft_image * np.conj(fft_mask)
    fft_conjunto = np.fft.fftshift(np.fft.ifft2(fft_conjunto))
    # plt.figure('FFT')
    """fft = np.abs(np.fft.ifft2(fft_image * np.conj(fft_mask)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Crea las coordenadas X e Y para la matriz
    x, y = np.meshgrid(np.arange(fft.shape[0]), np.arange(fft.shape[1]))
    # Aplana la matriz y convierte los valores en una lista
    z = fft.flatten()
    # Crea una barra de color para los valores de la matriz
    colores = plt.cm.jet(z / np.amax(z))
    # Dibuja la matriz en 3D
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, color=colores)
    # Configura los límites de los ejes
    ax.set_xlim(0, fft.shape[0])
    ax.set_ylim(0, fft.shape[1])
    ax.set_zlim(0, np.amax(z))
    # Muestra la figura
    path = './outputs/zz-04_FFT_NORM.png'
    plt.savefig(path)"""
    # plt.imshow(fft)
    # plt.show()
    shape = np.unravel_index(np.argmax(fft_conjunto), fft_conjunto.shape)

    return shape


## PROBLEM 1  --------------------------------------------------
# TODO LEER LAS IMAGENES DE LA CARPETA
files = os.listdir('./imatges/petites') # Puede ser ./imatges/petites o ./imatges/grans

# TODO. CARGAMOS UNA PRIMERA IMAGEN PARA HACER TODAS LAS PRUEBAS
i = 1
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

    # TODO. BUSCO EL PUNTO CON MAYOR CORRELACIÓN EL CUAL ESTÉ MÁS CERCA DEL CENTRO DE LA MATRIZ
    corr_rg = correlacion_cruzada(R, G)
    corr_rb = correlacion_cruzada(R, B)

    corr_rg_norm = normxcorr2(R, G)
    corr_rb_norm = normxcorr2(R, B)

    fft_rg = fourier_transform(R, G)
    fft_rb = fourier_transform(R, B)

    fft_rg_norm = fourier_transform(R, G)
    fft_rb_norm = fourier_transform(R, B)


    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B Im1 -> (85, 100)
    shift_g = [corr_rg[0] - R.shape[0] // 2, corr_rg[1] - R.shape[1] // 2]
    shift_b = [corr_rb[0] - R.shape[0] // 2, corr_rb[1] - R.shape[1] // 2]

    shift_g_norm = [corr_rg_norm[0] - R.shape[0] // 2, corr_rg_norm[1] - R.shape[1] // 2]
    shift_b_norm = [corr_rb_norm[0] - R.shape[0] // 2, corr_rb_norm[1] - R.shape[1] // 2]

    shift_fft_rg = [fft_rg[0] - R.shape[0] // 2, fft_rg[1] - R.shape[1] // 2]
    shift_fft_rb = [fft_rb[0] - R.shape[0] // 2, fft_rb[1] - R.shape[1] // 2]

    shift_fft_rg_norm = [fft_rg_norm[0] - R.shape[0] // 2, fft_rg_norm[1] - R.shape[1] // 2]
    shift_fft_rb_norm = [fft_rb_norm[0] - R.shape[0] // 2, fft_rb_norm[1] - R.shape[1] // 2]

    # TODO Desplazamos los canales G
   
    M = np.float32([[1, 0, shift_g[0]], [0, 1, shift_g[1]]])
    g_shifted = cv2.warpAffine(G, M, (width_fin, height_fin))
    
    M = np.float32([[1, 0, shift_g_norm[0]], [0, 1, shift_g_norm[1]]])
    g_shifted_norm = cv2.warpAffine(G, M, (width_fin, height_fin))

    M = np.float32([[1, 0, shift_fft_rg[0]], [0, 1, shift_fft_rg[1]]])
    g_shifted_fft = cv2.warpAffine(G, M, (width_fin, height_fin))

    M = np.float32([[1, 0, shift_fft_rg_norm[0]], [0, 1, shift_fft_rg_norm[1]]])
    g_shifted_fft_norm = cv2.warpAffine(G, M, (width_fin, height_fin))

    # TODO Desplazamos los canales b
    
    M = np.float32([[1, 0, shift_b[0]],[0, 1, shift_b[1]]])
    b_shifted = cv2.warpAffine(B,M, (width_fin, height_fin))
    
    M = np.float32([[1, 0, shift_b_norm[0]],[0, 1, shift_b_norm[1]]])
    b_shifted_norm = cv2.warpAffine(B,M, (width_fin, height_fin))

    M = np.float32([[1, 0, shift_fft_rb[0]], [0, 1, shift_fft_rb[1]]])
    b_shifted_fft = cv2.warpAffine(B, M, (width_fin, height_fin))

    M = np.float32([[1, 0, shift_fft_rb_norm[0]], [0, 1, shift_fft_rb_norm[1]]])
    b_shifted_fft_norm = cv2.warpAffine(B, M, (width_fin, height_fin))

    # Juntamos los tres canales en una sola imagen
    #plt.figure('R')
    #plt.imshow(R)
    #plt.figure('G')
    #plt.imshow(g_shifted)
    #plt.figure('B')
    #plt.imshow(b_shifted)

    # TODO. MOSTRAMOS LAS IMAGENES QUE HEMOS ACABADO DE FORMAR
    plt.figure('Inicial')
    rgb_img = np.dstack((R, G, B))
    # path = './outputs/ ' + str(i) +'_01_inicial_color.png'
    plt.imshow(rgb_img)
    # plt.imsave(path, rgb_img)
    plt.figure('Correlació')
    rgb_corr = np.dstack((R, g_shifted, b_shifted))
    # path = './outputs/ ' + str(i) +'_02__correlacio_color.png'
    plt.imshow(rgb_corr)
    # plt.imsave(path, rgb_corr)
    plt.figure('Correlació Normalitzada')
    rgb_corr_norm = np.dstack((R, g_shifted_norm, b_shifted_norm))
    # path = './outputs/ ' + str(i) +'_03__correlacio_norm_color.png'
    plt.imshow(rgb_corr_norm)
    # plt.imsave(path, rgb_corr_norm)
    plt.figure('Transformada Fourier')
    rgb_fft = np.dstack((R, g_shifted_fft, b_shifted_fft))
    # path = './outputs/ ' + str(i) +'_04__fft_color.png'
    plt.imshow(rgb_fft)
    # plt.imsave(path, rgb_fft)
    plt.figure('Transformada Fourier Normalitzada')
    rgb_fft_norm = np.dstack((R, g_shifted_fft_norm, b_shifted_fft_norm))
    # path = './outputs/ ' + str(i) +'_05__fft_norm_color.png'
    plt.imshow(rgb_fft_norm)
    # plt.imsave(path, rgb_fft_norm)
    plt.show()
    i += 1

    # TODO TEST
    """matriz = np.random.randint(0, 255, size=(height_fin, width_fin))

    # Crear dos copias de la matriz original
    matriz_movimiento1 = np.copy(matriz)
    matriz_movimiento2 = np.copy(matriz)

    # Aplicar un movimiento de 2 a la derecha y 3 hacia abajo a la primera matriz
    matriz_movimiento1 = np.roll(matriz_movimiento1, shift=3, axis=1)
    matriz_movimiento1 = np.roll(matriz_movimiento1, shift=5, axis=0)

    # Aplicar un movimiento de 5 a la izquierda y 1 hacia arriba a la segunda matriz
    matriz_movimiento2 = np.roll(matriz_movimiento2, shift=-5, axis=1)
    matriz_movimiento2 = np.roll(matriz_movimiento2, shift=-1, axis=0)

    # Recortar 5 píxeles de cada borde de las tres matrices
    matriz_recortada = matriz[5:-5, 5:-5]
    matriz_movimiento1_recortada = matriz_movimiento1[5:-5, 5:-5]
    matriz_movimiento2_recortada = matriz_movimiento2[5:-5, 5:-5]

    corr_rg = correlacion_cruzada(matriz_recortada, matriz_movimiento1_recortada)

    corr_rb = correlacion_cruzada(matriz_recortada, matriz_movimiento2_recortada)

    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B Im1 -> (85, 100)
    shift_g = [corr_rg[0] - matriz_recortada.shape[0] // 2, corr_rg[1] - matriz_recortada.shape[1] // 2]
    shift_b = [corr_rb[1] - matriz_recortada.shape[1] // 2, corr_rb[0] - matriz_recortada.shape[0] // 2] # INVERITDO

    corr_rg_norm = normxcorr2(matriz_recortada, matriz_movimiento1_recortada)
    corr_rb_norm = normxcorr2(matriz_recortada, matriz_movimiento2_recortada)

    # TODO. MUEVO CADA MATRIZ A LA POSICIÓN QUE LE TOQUE

    # Calculamos cuánto tenemos que desplazar los canales G y B Im1 -> (85, 100)
    shift_g_norm = [corr_rg_norm[0] - matriz_recortada.shape[0] // 2, corr_rg_norm[1] - matriz_recortada.shape[1] // 2]
    shift_b_norm = [corr_rb_norm[0] - matriz_recortada.shape[0] // 2, corr_rb_norm[1] - matriz_recortada.shape[1] // 2]

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
    plt.show()"""


