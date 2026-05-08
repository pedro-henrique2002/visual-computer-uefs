import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys

# --- FUNÇÃO DE SELEÇÃO FORNECIDA ---
def selecionar_e_ler_imagem(mensagem="Selecione uma imagem"):
    print(mensagem)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(
        title=mensagem,
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

#------ Parte 1 --------

def limpeza_morfologica(imagem):
    # 1. Converter para tons de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # 2. Binarizar a imagem (Transformar em Preto e Branco puro)
    # O cv2.THRESH_OTSU calcula o melhor limiar sozinho
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Definir o Kernel (tamanho da máscara que desliza sobre a imagem)
    # Um kernel 5x5 costuma ser bom para ruídos médios
    kernel = np.ones((3, 3), np.uint8)

    # 4. Aplicar Abertura (Remove ruído externo / pontos brancos pequenos)
    abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    # 5. Aplicar Fechamento (Remove ruído interno / buracos pretos nas formas)
    # Geralmente aplicamos o fechamento sobre o resultado da abertura
    final = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)


    # Exibição dos resultados para o relatório
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Binária Original")
    plt.imshow(binaria, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("2. Após Abertura (Sem ruído fora)")
    plt.imshow(abertura, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("3. Após Fechamento (Sem buracos)")
    plt.imshow(final, cmap='gray')
    
    plt.tight_layout()
    plt.show()
    
    return final

# --- MÉTODO 2: COMPARAÇÃO DE SUAVIZAÇÃO (PASSA-BAIXA VS MEDIANA) ---
def comparar_suavizacao(imagem):
    # 1. Converter para tons de cinza para facilitar a visualização do ruído
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar Filtro de Média (Box Filter) - Kernel 5x5
    # Ele simplesmente soma tudo e divide pelo número de pixels
    media = cv2.blur(cinza, (5, 5))
    
    # 3. Aplicar Filtro Gaussiano - Kernel 5x5
    # O '0' final indica que o desvio padrão será calculado automaticamente
    gaussiano = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # 4. Aplicar Filtro de Mediana - Tamanho 5
    # Nota: O tamanho aqui deve ser um número ímpar
    mediana = cv2.medianBlur(cinza, 5)
    
    # Exibição para comparação
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("1. Original com Ruído")
    plt.imshow(cinza, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("2. Filtro de Média (Borra Bordas)")
    plt.imshow(media, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("3. Filtro Gaussiano (Suave)")
    plt.imshow(gaussiano, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title("4. Filtro de Mediana (Preserva Bordas)")
    plt.imshow(mediana, cmap='gray')
    
    plt.tight_layout()
    plt.show()
    
    return mediana


# --- MÉTODO 3: ANÁLISE DE FOURIER (ESPECTRO DE MAGNITUDE) ---
def analisar_fourier(imagem):
    # 1. Converter para tons de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar a Transformada de Fourier (DFT)
    # O resultado é um array de números complexos
    dft = np.fft.fft2(cinza)
    
    # 3. Centralizar as frequências (Shift)
    # Por padrão, as frequências baixas ficam nos cantos. 
    # O fftshift as coloca no centro para facilitar a leitura humana.
    dft_shift = np.fft.fftshift(dft)
    
    # 4. Calcular o Espectro de Magnitude (em escala Logarítmica)
    # Usamos o log porque os valores do centro são milhões de vezes maiores que os das bordas.
    espectro_magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    
    # Exibição
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original (Domínio do Espaço)")
    plt.imshow(cinza, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Espectro de Magnitude (Domínio da Frequência)")
    plt.imshow(espectro_magnitude, cmap='gray')
    
    # Adicionando anotações visuais para o seu entendimento
    plt.text(0, 0, "<- Frequências Altas (Bordas)", color='yellow', fontsize=10)
    plt.annotate('Frequências Baixas (Centro)', xy=(espectro_magnitude.shape[1]//2, espectro_magnitude.shape[0]//2), 
                 xytext=(10, 10), textcoords='offset points', color='cyan', arrowprops=dict(arrowstyle='->', color='cyan'))
    
    plt.show()
    
    return dft_shift


# --- MÉTODO 4: RESTAURAÇÃO (NOTCH FILTER + CANNY) ---
def restauracao_notch_canny(imagem, dft_shift):
    """
    Versão corrigida para evitar erros de tipo (Linter/Pylance).
    """
    # 1. Dimensões
    rows, cols = imagem.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # 2. Criar Máscara (1s = Branco, 0s = Preto)
    # Usamos dtype=np.float32 para evitar erros de multiplicação depois
    mask = np.ones((rows, cols), dtype=np.float32)
    
    # --- ESPAÇO PARA AJUSTE DE COORDENADAS DO RUÍDO ---
    # Exemplo: mask[crow-20:crow+20, ccol-20:ccol+20] = 0
    # --------------------------------------------------

    # 3. Aplicar Filtro e Transformada Inversa
    fshift_filtrado = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_back = np.fft.ifft2(f_ishift)
    
    # 4. Pegar a magnitude (parte real)
    img_abs = np.abs(img_back)
    
    # 5. NORMALIZAÇÃO MANUAL (SUBSTITUI O CV2.NORMALIZE PARA EVITAR ERROS)
    # Esta fórmula faz exatamente o que o normalize faz: (valor - min) / (max - min) * 255
    img_min = np.min(img_abs)
    img_max = np.max(img_abs)
    
    # Evita divisão por zero se a imagem for toda de uma cor
    if img_max > img_min:
        img_norm = (255 * (img_abs - img_min) / (img_max - img_min))
    else:
        img_norm = img_abs

    # Converte explicitamente para uint8 (Inteiro de 8 bits)
    img_restaurada = img_norm.astype(np.uint8)

    # 6. CANNY (COM CONVERSÃO EXPLÍCITA PARA SATISFAZER O LINTER)
    # Garantimos que a imagem original também esteja em tons de cinza e uint8
    if len(imagem.shape) == 3:
        cinza_original = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        cinza_original = imagem

    bordas_suja = cv2.Canny(cinza_original.astype(np.uint8), 100, 200)
    bordas_limpa = cv2.Canny(img_restaurada, 100, 200)

    # 7. EXIBIÇÃO
    plt.figure("Resultado Final: Notch e Canny", figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("1. Original com Ruído")
    plt.imshow(cinza_original, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("2. Restaurada (Frequência Limpa)")
    plt.imshow(img_restaurada, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("3. Bordas na Original (Com Ruído)")
    plt.imshow(bordas_suja, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title("4. Bordas na Restaurada (Sem Ruído)")
    plt.imshow(bordas_limpa, cmap='gray')
    
    plt.tight_layout()
    plt.show()



# --- MAIN: ORQUESTRAÇÃO DO TRABALHO ---
def main():
    print("Iniciando Parte 1: Ruído Fino e Morfologia")
    img_fina = selecionar_e_ler_imagem("Selecione imagem com ruído Sal e Pimenta")
    limpeza_morfologica(img_fina)
    comparar_suavizacao(img_fina)
    
    print("\nIniciando Parte 2: Ruído Periódico e Fourier")
    img_per = selecionar_e_ler_imagem("Selecione imagem com padrão repetitivo (listras)")
    dft_s = analisar_fourier(img_per)
    restauracao_notch_canny(img_per, dft_s)

if __name__ == "__main__":
    main()