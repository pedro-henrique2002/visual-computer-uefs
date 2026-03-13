import cv2
import matplotlib.pyplot as plt
from utils.image import selecionar_e_ler_imagem

# 1. Carregar as bibliotecas (cv2 e plt já importados acima)

# 2. Fazer o upload/seleção da foto usando sua função customizada
# A função já retorna a imagem lida pelo cv2.imread
img_bgr = selecionar_e_ler_imagem()

# 3. Ler a imagem e a exibir (Passo 4 do PDF)
# Vamos exibir primeiro do jeito que o OpenCV leu (BGR)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_bgr)
plt.title('1. Visualização BGR (Incorreta)')
print("Exibindo imagem em BGR... Note como as cores estão trocadas.")

# 4. Verificar e Corrigir (Passo 5 e 6 do PDF)
# O OpenCV lê como Blue-Green-Red. O Matplotlib espera Red-Green-Blue.
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Exibir a imagem corrigida
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title('2. Visualização RGB (Corrigida)')
print("Exibindo imagem em RGB... Agora as cores estão certas!")

plt.show()