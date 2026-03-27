import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import numpy as np

def selecionar_e_ler_imagem():
    """
    Abre janela de seleção e retorna a imagem lida pelo OpenCV (BGR).
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    caminho = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    root.destroy()

    if not caminho:
        print("Ação cancelada.")
        sys.exit()

    imagem = cv2.imread(caminho)
    
    if imagem is None:
        print("Erro ao carregar imagem.")
        sys.exit()

    return imagem

def binarizacao_automatica_adaptativa(img_cinza):
    # 1. Analisar a Estatística da Imagem (O Histograma na prática)
    media = np.mean(img_cinza)
    desvio_padrao = np.std(img_cinza)
    altura, largura = img_cinza.shape
    
    print(f"--- Análise Estatística ---")
    print(f"Brilho Médio: {media:.2f}")
    print(f"Desvio Padrão (Contraste): {desvio_padrao:.2f}")

    # 2. Definir o Tamanho do Bloco Base (5% da menor dimensão)
    # Isso garante que em fotos de alta resolução o bloco não seja pequeno demais
    bloco_base = int(min(altura, largura) * 0.05)
    
    # 3. Ajustar de acordo com o "Aperto" do Histograma
    # Se o desvio padrão for baixo (< 30), o histograma está espremido.
    # Precisamos de um bloco maior para 'enxergar' a diferença.
    if desvio_padrao < 30:
        print("Diagnóstico: Histograma ESPREMIDO (Baixo Contraste).")
        bloco_final = bloco_base * 2  # Dobramos o bloco para suavizar o ruído
        c_final = 1                   # C pequeno para não perder detalhes fracos
    else:
        print("Diagnóstico: Histograma EQUILIBRADO (Bom Contraste).")
        bloco_final = bloco_base
        c_final = 5                   # C maior para limpar melhor as sombras

    # O Bloco deve ser sempre ÍMPAR e no mínimo 3
    if bloco_final % 2 == 0: bloco_final += 1
    if bloco_final < 3: bloco_final = 3

    print(f"Parâmetros Sugeridos: Bloco={bloco_final}, C={c_final}")

    # 4. Aplicar a binarização com os parâmetros calculados
    bin_auto = cv2.adaptiveThreshold(
        img_cinza, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        bloco_final, c_final
    )

    return bin_auto

def main():

    # 1° Carregar a imagem
    img_colorida = selecionar_e_ler_imagem()

    # 2° Converter para escala de cinza
    img_cinza = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)

    # 3. Gerar o Histograma Original
    # calcHist(imagem, canais, máscara, número de bins, intervalo)
    hist_original = cv2.calcHist([img_cinza], [0], None, [256], [0, 256])

    # Exibir imagem e histograma lado a lado
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_cinza, cmap='gray')
    plt.title("Imagem em Tons de Cinza")

    plt.subplot(1, 2, 2)
    plt.plot(hist_original)
    plt.title("Histograma Original")
    plt.xlim([0, 256])
    plt.show()

    # Aplicar a equalização
    img_equalizada = cv2.equalizeHist(img_cinza)

    # Gerar novo histograma
    hist_equalizado = cv2.calcHist([img_equalizada], [0], None, [256], [0, 256])

    # Comparação Visual
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='blue', label='Original')
    plt.plot(hist_equalizado, color='red', label='Equalizado')
    plt.legend()
    plt.title("Comparação de Histogramas")

    plt.subplot(1, 2, 2)
    plt.imshow(img_equalizada, cmap='gray')
    plt.title("Imagem após Equalização")
    plt.show()

    # A. Método de Otsu (Global)
    # O Otsu retorna o limiar calculado (ret) e a imagem binarizada
    print(f"cv2.THRESH_BINARY_INV = {cv2.THRESH_BINARY_INV} | cv2.THRESH_OTSU {cv2.THRESH_OTSU}")
    ret, bin_otsu = cv2.threshold(img_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # B. Método Adaptativo (Local)
    # 11 é o tamanho do bloco (vizinhança), 2 é uma constante subtraída da média
    print(f"cv2.THRESH_BINARY_INV = {cv2.THRESH_BINARY_INV} | cv2.ADAPTIVE_THRESH_GAUSSIAN_C {cv2.ADAPTIVE_THRESH_GAUSSIAN_C}")
    #bin_adapt = cv2.adaptiveThreshold(img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY_INV, 31, 2)
    bin_adapt = binarizacao_automatica_adaptativa(img_cinza)
    
    # Exibição para o relatório
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(bin_otsu, cmap='gray')
    plt.title(f"Otsu (Limiar: {ret})")

    plt.subplot(1, 2, 2)
    plt.imshow(bin_adapt, cmap='gray')
    plt.title("Adaptativa (Resistente a sombras)")
    plt.show()

    # No OpenCV, após a binarização:
    # Objeto (Branco) = 255
    # Fundo (Preto) = 0

    # 1. Contar pixels do objeto
    pixels_objeto = cv2.countNonZero(bin_adapt)

    # 2. Total de pixels da imagem
    total_pixels = bin_adapt.size # largura * altura

    # 3. Cálculo da porcentagem
    porcentagem_area = (pixels_objeto / total_pixels) * 100

    print(f"Total de pixels: {total_pixels}")
    print(f"Pixels do objeto: {pixels_objeto}")
    print(f"O objeto ocupa {porcentagem_area:.2f}% da cena.")

    

# --- BLOCO PRINCIPAL (MAIN) ---
if __name__ == "__main__":
    print("Iniciando Roteiro de atividade - Avaliativa 01")
    
    
    main()

    print("\n--- Atividade Finalizada ---")
