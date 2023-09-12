import cv2
import numpy as np
from scipy.signal import convolve2d

# Carregando uma imagem de exemplo
imagem = cv2.imread('exemplo.jpg', cv2.IMREAD_GRAYSCALE)

# Máscara da Média
media_mask = np.ones((3, 3), np.float32) / 9
media_result_cv2 = cv2.filter2D(imagem, -1, media_mask)
media_result_scipy = convolve2d(imagem, media_mask, mode='same')

# Máscara Gaussiana
gauss_mask = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16
gauss_result_cv2 = cv2.filter2D(imagem, -1, gauss_mask)
gauss_result_scipy = convolve2d(imagem, gauss_mask, mode='same')

# Máscara Laplaciana
laplacian_mask = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float32)
laplacian_result_cv2 = cv2.filter2D(imagem, -1, laplacian_mask)
laplacian_result_scipy = convolve2d(imagem, laplacian_mask, mode='same')

# Máscara Sobel X
sobel_x_mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
sobel_x_result_cv2 = cv2.filter2D(imagem, -1, sobel_x_mask)
sobel_x_result_scipy = convolve2d(imagem, sobel_x_mask, mode='same')

# Máscara Sobel Y
sobel_y_mask = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=np.float32)
sobel_y_result_cv2 = cv2.filter2D(imagem, -1, sobel_y_mask)
sobel_y_result_scipy = convolve2d(imagem, sobel_y_mask, mode='same')

# Gradiente (Sobel X + Sobel Y)
gradient_result_cv2 = sobel_x_result_cv2 + sobel_y_result_cv2
gradient_result_scipy = sobel_x_result_scipy + sobel_y_result_scipy

# Laplaciano somado à imagem original
laplacian_result_cv2_combined = cv2.add(imagem, laplacian_result_cv2)
laplacian_result_scipy_combined = imagem + laplacian_result_scipy

# Exibindo os resultados
cv2.imshow('Original', imagem)
cv2.imshow('Média (OpenCV)', media_result_cv2)
cv2.imshow('Média (SciPy)', media_result_scipy)
cv2.imshow('Gaussiana (OpenCV)', gauss_result_cv2)
cv2.imshow('Gaussiana (SciPy)', gauss_result_scipy)
cv2.imshow('Laplaciana (OpenCV)', laplacian_result_cv2)
cv2.imshow('Laplaciana (SciPy)', laplacian_result_scipy)
cv2.imshow('Sobel X (OpenCV)', sobel_x_result_cv2)
cv2.imshow('Sobel X (SciPy)', sobel_x_result_scipy)
cv2.imshow('Sobel Y (OpenCV)', sobel_y_result_cv2)
cv2.imshow('Sobel Y (SciPy)', sobel_y_result_scipy)
cv2.imshow('Gradiente (OpenCV)', gradient_result_cv2)
cv2.imshow('Gradiente (SciPy)', gradient_result_scipy)
cv2.imshow('Laplaciano + Original (OpenCV)', laplacian_result_cv2_combined)
cv2.imshow('Laplaciano + Original (SciPy)', laplacian_result_scipy_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()

