import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image import selecionar_e_ler_imagem

# A função já retorna a imagem lida pelo cv2.imread
cinza_fundo_bgr = selecionar_e_ler_imagem()
cinza_objeto_bgr = selecionar_e_ler_imagem()

#Converte da escala de BGR para escala de cinza (3 canais de cores pra 1 canal só)
cinza_fundo = cv2.cvtColor(cinza_fundo_bgr, cv2.COLOR_BGR2GRAY)
cinza_objeto = cv2.cvtColor(cinza_objeto_bgr, cv2.COLOR_BGR2GRAY)

#criação da mascara fazendo a diferença absoluta de pixel a pixel
mascara = cv2.absdiff(cinza_fundo, cinza_objeto)
#Prova de que temos só 1 canal
print(cinza_objeto.shape)
print(cinza_fundo.shape)
print(mascara.shape)

# 1. Calcular a média de todos os pixels da máscara
# Como a máscara é o resultado do absdiff, a média representa 
# o "quão diferente", em média, cada pixel está do fundo original.
media = np.mean(mascara)

# 2. Exibir o valor da média (opcional, mas bom para você ajustar o limiar)
print(f"Média de diferença: {media:.2f}")

# 3. Verificar o limiar (Threshold)
limiar = 5.0
if media > limiar:
    print("ALERTA: Mudança detectada no cenário!")
else:
    print("Cenário estático.")

plt.figure(figsize=(15, 5)) # Aumenta um pouco a janela para ver melhor

plt.subplot(1, 3, 1)
# ADICIONADO: cmap='gray'
plt.imshow(cinza_fundo, cmap='gray') 
plt.title('1. Fundo (Cinza)')
plt.axis('off')

plt.subplot(1, 3, 2)
# ADICIONADO: cmap='gray'
plt.imshow(cinza_objeto, cmap='gray') 
plt.title('2. Objeto (Cinza)')
plt.axis('off')

plt.subplot(1, 3, 3)
# ADICIONADO: cmap='gray'
plt.imshow(mascara, cmap='gray') 
plt.title('3. Diferença (cv2.absdiff)')
plt.axis('off')

plt.tight_layout()
plt.show()