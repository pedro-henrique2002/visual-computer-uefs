import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys

def selecionar_e_ler_imagem():
    """
    Abre uma janela de seleção de arquivo do sistema operacional e 
    retorna a imagem lida pelo OpenCV no formato padrão (BGR).
    """
    # Cria uma janela raiz oculta para o Tkinter
    root = tk.Tk()
    root.withdraw()
    # Garante que a janela de seleção apareça na frente de outros programas
    root.attributes("-topmost", True)

    print("Aguardando seleção de imagem...")
    caminho = filedialog.askopenfilename(
        title="Selecione uma imagem para os exercícios",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    
    # Fecha a instância do tkinter para liberar memória
    root.destroy()

    # Se o usuário cancelar a seleção, encerra o programa sem erros
    if not caminho:
        print("Ação cancelada pelo usuário. Encerrando...")
        sys.exit()

    # OpenCV lê a imagem. Se falhar (arquivo corrompido, etc), retorna None
    imagem = cv2.imread(caminho)
    if imagem is None:
        print("Erro: Não foi possível carregar o arquivo selecionado.")
        sys.exit()
    
    return imagem

def main():
    # Passo Inicial: Selecionar a imagem que será usada em todos os exercícios
    img_original_bgr = selecionar_e_ler_imagem()

    # =========================================================================
    # EXERCÍCIO 1: Diferença BGR vs RGB e Exibição
    # =========================================================================
    print("\nExecutando Exercício 1: BGR vs RGB...")
    
    # O OpenCV lê imagens no padrão BGR (Blue, Green, Red).
    # O Matplotlib e a maioria dos softwares esperam RGB (Red, Green, Blue).
    img_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Exercício 1: Entendendo BGR vs RGB")
    
    # Exibição da imagem sem conversão (Cores trocadas)
    plt.subplot(1, 2, 1)
    plt.imshow(img_original_bgr)
    plt.title('1. Visualização BGR (Incorreta)\nNote como Vermelho e Azul se invertem')
    
    # Exibição da imagem com conversão (Cores naturais)
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title('2. Visualização RGB (Correta)\nPadrão esperado pelo Matplotlib')
    plt.show()

    # =========================================================================
    # EXERCÍCIO 2: Manipulação de Matriz via NumPy
    # =========================================================================
    print("Executando Exercício 2: Manipulação de Matriz...")
    
    # Criamos uma cópia para não alterar a imagem original lida no início
    img_modificada = img_original_bgr.copy()

    # Desenhar um quadrado preto (0,0,0) no canto superior esquerdo (50x50 pixels)
    # Slicing de matriz: imagem[y_inicial : y_final, x_inicial : x_final]
    img_modificada[0:50, 0:50] = [0, 0, 0]

    # "Explodir" o canal Vermelho no centro da imagem
    # Pegamos as dimensões (altura, largura, canais) da matriz
    h, w, _ = img_modificada.shape
    cy, cx = h // 2, w // 2 # Coordenadas do centro (divisão inteira)
    t = 50 # Definimos um raio de 50 pixels para o quadrado central

    # No OpenCV (BGR), o índice 2 é o canal Vermelho.
    # Setamos o valor 255 (intensidade máxima) apenas para esse canal na região
    img_modificada[cy-t : cy+t, cx-t : cx+t, 2] = 255

    plt.figure(figsize=(12, 6))
    plt.suptitle("Exercício 2: Manipulação Direta de Pixels (NumPy)")
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_modificada, cv2.COLOR_BGR2RGB))
    plt.title('Modificada\nQuadrado Preto (0,0) e Centro Vermelho')
    plt.show()

    # =========================================================================
    # EXERCÍCIO 3: Tons de Cinza e Binarização (Thresholding Manual)
    # =========================================================================
    print("Executando Exercício 3: Binarização...")
    
    # 1. Converter para Tons de Cinza (Grayscale)
    # Usa a fórmula de luminosidade: (0.299*R + 0.587*G + 0.114*B)
    img_gray = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Criar imagem binarizada (Preto e Branco puro) baseado no Limiar 127
    # Se o valor do pixel for > 127, vira 255 (Branco). Caso contrário, 0 (Preto).
    # O np.where processa toda a matriz de forma otimizada.
    img_binaria = np.where(img_gray > 127, 255, 0).astype(np.uint8)

    plt.figure(figsize=(15, 5))
    plt.suptitle("Exercício 3: Processo de Binarização (Thresholding)")
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB))
    plt.title('1. Original Colorida')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_gray, cmap='gray') # Necessário informar cmap='gray' para 1 canal
    plt.title('2. Tons de Cinza\nIntensidade de 0 a 255')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_binaria, cmap='gray')
    plt.title('3. Binarizada\nLimiar Manual em 127')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    print("\nLaboratório concluído com sucesso!")

# Ponto de entrada padrão para scripts Python
if __name__ == "__main__":
    main()