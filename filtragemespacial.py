import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para aplicar as operações de convolução e exibir em uma janela separada
def aplicar_convolucoes_e_mostrar_resultados(imagem_path, titulo):
    # Carregando a imagem
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

    # Máscaras
    mascara_media = np.ones((3, 3), np.float32) / 9
    mascara_gaussiana = np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=np.float32) / 16
    mascara_laplaciana = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=np.float32)
    mascara_sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32)
    mascara_sobel_y = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=np.float32)

    # Aplicando as convoluções
    resultado_media = cv2.filter2D(imagem, -1, mascara_media)
    resultado_gaussiana = cv2.filter2D(imagem, -1, mascara_gaussiana)
    resultado_laplaciana = cv2.filter2D(imagem, -1, mascara_laplaciana)
    resultado_sobel_x = cv2.filter2D(imagem, -1, mascara_sobel_x)
    resultado_sobel_y = cv2.filter2D(imagem, -1, mascara_sobel_y)
    gradiente = resultado_sobel_x + resultado_sobel_y
    laplaciano_somado = imagem + resultado_laplaciana

    # Criando uma figura para exibir os resultados
    plt.figure(figsize=(15, 8))

    # Imagem Original
    plt.subplot(3, 4, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Original')

    # Máscara Média
    plt.subplot(3, 4, 2)
    plt.imshow(resultado_media, cmap='gray')
    plt.title('Média')

    # Máscara Gaussiana
    plt.subplot(3, 4, 3)
    plt.imshow(resultado_gaussiana, cmap='gray')
    plt.title('Gaussiana')

    # Máscara Laplaciana
    plt.subplot(3, 4, 4)
    plt.imshow(resultado_laplaciana, cmap='gray')
    plt.title('Laplaciano')

    # Máscara Sobel X
    plt.subplot(3, 4, 5)
    plt.imshow(resultado_sobel_x, cmap='gray')
    plt.title('Sobel X')

    # Máscara Sobel Y
    plt.subplot(3, 4, 6)
    plt.imshow(resultado_sobel_y, cmap='gray')
    plt.title('Sobel Y')

    # Gradiente (Sobel X + Sobel Y)
    plt.subplot(3, 4, 7)
    plt.imshow(gradiente, cmap='gray')
    plt.title('Gradiente (Sobel X + Sobel Y)')

    # Laplaciano somado à imagem original
    plt.subplot(3, 4, 8)
    plt.imshow(laplaciano_somado, cmap='gray')
    plt.title('Laplaciano + Original')

    # Configurando o título da janela
    plt.suptitle(titulo, fontsize=16)
    plt.show()

# Aplicar as operações de convolução para três imagens distintas em janelas separadas
aplicar_convolucoes_e_mostrar_resultados('cameraman.tif', 'Cameraman')
aplicar_convolucoes_e_mostrar_resultados('biel.png', 'Biel')
aplicar_convolucoes_e_mostrar_resultados('lena.jpg', 'Lena')
