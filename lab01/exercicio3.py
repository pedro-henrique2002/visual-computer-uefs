import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image import selecionar_e_ler_imagem

# 1. Selecionar e carregar a imagem
img_bgr = selecionar_e_ler_imagem()

# 2. Converter a imagem original para Tons de Cinza (Grayscale)
# O OpenCV usa a fórmula de pesos (R:29%, G:58%, B:11%) automaticamente
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 3. Binarização Manual (Thresholding)
# Criamos uma nova imagem baseada na regra:
# Se o pixel > 127, vira 255 (Branco)
# Se o pixel <= 127, vira 0 (Preto)
# O np.where funciona como um "if/else" para a matriz inteira de uma vez
img_binaria = np.where(img_gray > 127, 255, 0).astype(np.uint8)

# 4. Exibição dos resultados
plt.figure(figsize=(15, 5))

# Imagem Original (Convertida para RGB apenas para exibição correta no Matplotlib)
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.title('1. Original Colorida')
plt.axis('off')

# Imagem em Tons de Cinza
plt.subplot(1, 3, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('2. Tons de Cinza')
plt.axis('off')

# Imagem Binarizada (Preto e Branco)
plt.subplot(1, 3, 3)
plt.imshow(img_binaria, cmap='gray')
plt.title('3. Binarizada (Limiar 127)')
plt.axis('off')

plt.tight_layout()
plt.show()