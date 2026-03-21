import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.image import selecionar_e_ler_imagem

# A função já retorna a imagem lida pelo cv2.imread
img_bgr = selecionar_e_ler_imagem()

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Cria a nova imagem com as mesmas dimensões (img.shape)
# O dtype='uint8' é fundamental para que o computador entenda como imagem (0-255)
img_100 = np.ones(img.shape, dtype="uint8") * 100

img_soma = img + img_100

img_add = cv2.add(img, img_100)

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('1. Visualização RGB (Correta))')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_soma)
plt.title('2. Visualização Soma com Operação')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_add)
plt.title('2. Visualização Soma com cv2.add()')
plt.axis('off')

plt.tight_layout()
plt.show()