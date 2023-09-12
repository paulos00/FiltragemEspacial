import numpy as np
import cv2

# Carregue a imagem de entrada em escala de cinza
imagem = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Máscara de média
kernel_media = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]) / 9

# Máscara Gaussiana
kernel_gaussiano = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]]) / 16

# Máscara Laplaciana
kernel_laplaciano = np.array([[0,  1, 0],
                              [1, -4, 1],
                              [0,  1, 0]])

# Máscaras de Sobel
kernel_sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

kernel_sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

# Aplicar a convolução com as máscaras
imagem_media = cv2.filter2D(imagem, -1, kernel_media)
imagem_gaussiana = cv2.filter2D(imagem, -1, kernel_gaussiano)
imagem_laplaciana = cv2.filter2D(imagem, -1, kernel_laplaciano)
imagem_sobel_x = cv2.filter2D(imagem, -1, kernel_sobel_x)
imagem_sobel_y = cv2.filter2D(imagem, -1, kernel_sobel_y)
gradiente = cv2.add(imagem_sobel_x, imagem_sobel_y)
imagem_laplaciana_somada = cv2.add(imagem, imagem_laplaciana)

# Exibir as imagens resultantes
cv2.imshow('Média', imagem_media)
cv2.imshow('Gaussiana', imagem_gaussiana)
cv2.imshow('Laplaciana', imagem_laplaciana)
cv2.imshow('Sobel X', imagem_sobel_x)
cv2.imshow('Sobel Y', imagem_sobel_y)
cv2.imshow('Gradiente', gradiente)
cv2.imshow('Laplaciana somada', imagem_laplaciana_somada)

cv2.waitKey(0)
cv2.destroyAllWindows()
