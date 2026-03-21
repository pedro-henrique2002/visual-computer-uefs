import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.image import selecionar_e_ler_imagem

# A função já retorna a imagem lida pelo cv2.imread
img_bgr = selecionar_e_ler_imagem()

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Defina os limites (AZUL vibrante)
# [H, S, V]
limite_inferior = np.array([100, 50, 50])
limite_superior = np.array([130, 255, 255])

# Cria a máscara (Pixels dentro do intervalo viram 255, fora viram 0)
mascara_objeto = cv2.inRange(img_hsv, limite_inferior, limite_superior)

# Aplica a máscara na imagem colorida original
# src1 e src2 são a mesma imagem, a mágica acontece no parâmetro mask
objeto_colorido = cv2.bitwise_and(img, img, mask=mascara_objeto)

# --- PASSO 5: Preparar o fundo Preto e Branco ---
# Primeiro, invertemos a máscara (o que era preto vira branco e vice-versa)
# Isso serve para "apagar" o objeto da imagem de fundo
mascara_inv = cv2.bitwise_not(mascara_objeto)

# Convertemos a imagem cinza de volta para 3 canais (RGB) 
# Ela continua parecendo cinza, mas agora tem o 'shape' (H, W, 3)
fundo_pb_3canais = cv2.cvtColor(img_cinza, cv2.COLOR_GRAY2RGB)

# Aplicamos a máscara invertida no fundo PB (criamos um "buraco negro" onde está o objeto)
fundo_com_buraco = cv2.bitwise_and(fundo_pb_3canais, fundo_pb_3canais, mask=mascara_inv)

# Somamos o (Objeto Colorido no fundo preto) + (Fundo PB com buraco no objeto)
sin_city = cv2.add(objeto_colorido, fundo_com_buraco)

plt.figure(figsize=(15, 5)) # Aumenta um pouco a janela para ver melhor

plt.subplot(2, 2, 1)
plt.imshow(img) 
plt.title('1. Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_cinza, cmap= 'gray') 
plt.title('2. Escala de Cinza')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mascara_objeto, cmap= 'gray') 
plt.title('3. Mascara')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(sin_city, cmap= 'gray') 
plt.title('4. Mascara colorida')
plt.axis('off')

plt.tight_layout()
plt.show()