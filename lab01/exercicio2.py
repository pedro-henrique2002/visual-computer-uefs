import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image import selecionar_e_ler_imagem

# 1. Carregar a imagem original
img_original = selecionar_e_ler_imagem()

# 2. Criar uma cópia para não alterar a original
img_modificada = img_original.copy()

# 3. Desenhar um quadrado preto (0,0,0) no canto superior esquerdo (50x50 pixels)
# Em matrizes: imagem[altura , largura] -> imagem[y1:y2 , x1:x2]
img_modificada[0:50, 0:50] = [0, 0, 0]

# 4. "Explodir" o canal Vermelho no centro da imagem
# Primeiro, vamos achar o centro da imagem
altura, largura, _ = img_modificada.shape
centro_y, centro_x = altura // 2, largura // 2
tamanho = 50 # define um quadrado de 100x100 (50 pra cada lado)

# No OpenCV (BGR), o canal Vermelho é o índice 2.
# Vamos pegar uma região central e setar o canal 2 para o valor máximo (255)
img_modificada[centro_y-tamanho : centro_y+tamanho, 
               centro_x-tamanho : centro_x+tamanho, 
               2] = 255

# 5. Exibir o resultado comparativo
plt.figure(figsize=(12, 6))

# Original
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
plt.axis('off')

# Modificada
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_modificada, cv2.COLOR_BGR2RGB))
plt.title('Modificada (Quadrado e Centro Vermelho)')
plt.axis('off')

plt.show()