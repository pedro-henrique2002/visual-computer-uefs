import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys

# =============================================================================
# FUNÇÃO: SELEÇÃO DE ARQUIVO
# =============================================================================
def selecionar_e_ler_imagem():
    """
    Interface gráfica simples para o usuário escolher um arquivo de imagem.
    Retorna a imagem no formato BGR (padrão do OpenCV).
    """
    root = tk.Tk()
    root.withdraw()  # Oculta a janela principal do Tkinter
    root.attributes("-topmost", True)  # Garante que a janela de seleção apareça na frente

    caminho = filedialog.askopenfilename(
        title="Selecione a imagem para o Estudo Dirigido",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()

    if not caminho:
        print("Operação cancelada pelo usuário.")
        sys.exit()

    imagem = cv2.imread(caminho)
    
    if imagem is None:
        print("Erro crítico: Não foi possível decodificar a imagem.")
        sys.exit()

    return imagem

# =============================================================================
# FUNÇÃO: ANÁLISE DO DOMÍNIO DO VALOR (ESTATÍSTICA)
# =============================================================================
def extrair_estatisticas_dominio_valor(img_cinza):
    """
    Esta função atua no 'Domínio do Valor'. Ela não olha para 'o que' está na 
    imagem, mas sim para 'como' os números (intensidades) estão distribuídos.
    
    Retorna um dicionário com: histograma, média, desvio padrão e 
    parâmetros sugeridos para binarização.
    """
    
    # 1. Cálculo do Histograma:
    # Conta a frequência de cada um dos 256 níveis de cinza.
    # [img]: imagem fonte | [0]: canal (cinza só tem 1) | None: sem máscara
    # [256]: número de barras (bins) | [0, 256]: intervalo de valores
    histograma = cv2.calcHist([img_cinza], [0], None, [256], [0, 256])

    # 2. Média Aritmética (Brilho Geral):
    # Indica se a imagem é, em média, clara ou escura.
    media = np.mean(img_cinza)

    # 3. Desvio Padrão (Contraste):
    # É a medida de quão 'espalhado' está o histograma.
    # Desvio alto = Histograma espalhado (Alto contraste).
    # Desvio baixo = Histograma espremido (Baixo contraste/Imagem lavada).
    desvio_padrao = np.std(img_cinza)

    # 4. Cálculo de Parâmetros Adaptativos baseados na Resolução:
    # Aqui unimos o Domínio Espacial (tamanho) com o Domínio do Valor (contraste).
    altura, largura = img_cinza.shape
    menor_dimensao = min(altura, largura)

    # Definimos o tamanho do bloco como uma proporção da imagem (aprox. 5%)
    # Se o desvio padrão for muito baixo, aumentamos o bloco para evitar ruído.
    proporcao = 0.05 if desvio_padrao > 30 else 0.08
    bloco = int(menor_dimensao * proporcao)
    
    # O tamanho do bloco para AdaptiveThreshold PRECISA ser ímpar e maior que 1.
    if bloco % 2 == 0: bloco += 1
    if bloco < 3: bloco = 3

    # A constante C será proporcional ao desvio padrão (sensibilidade ao ruído)
    c_calculado = int(desvio_padrao / 10)

    # Empacotamos os dados para retorno
    stats = {
        "hist": histograma,
        "media": media,
        "desvio": desvio_padrao,
        "sugestao_bloco": bloco,
        "sugestao_c": c_calculado,
        "resolucao": (largura, altura)
    }

    return stats

# =============================================================================
# FUNÇÃO PRINCIPAL (MAIN)
# =============================================================================
def main():
    print("--- Iniciando Processamento de Imagem (Estudo Dirigido) ---")

    # [PASSO 1] Carregar Imagem e Converter para Cinza
    # O processamento estatístico de brilho exige apenas um canal (intensidade).
    img_original = selecionar_e_ler_imagem()
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # [PASSO 2] Análise Inicial do Domínio do Valor
    # Extraímos os dados antes de qualquer modificação.
    stats_orig = extrair_estatisticas_dominio_valor(img_cinza)

    # [PASSO 3] Melhoria de Imagem (Equalização de Histograma)
    # Este processo redistribui as intensidades para ocupar todo o espectro 0-255.
    img_equalizada = cv2.equalizeHist(img_cinza)
    stats_equa = extrair_estatisticas_dominio_valor(img_equalizada)

    # [PASSO 4] Binarização Global (Método de Otsu)
    # O Otsu analisa o histograma e tenta achar o 'vale' entre dois 'picos'.
    # Usamos THRESH_BINARY_INV para que o objeto fique BRANCO (255).
    limiar_otsu, bin_otsu = cv2.threshold(
        img_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # [PASSO 5] Binarização Adaptativa (Local)
    # Diferente do Otsu, ele calcula o limiar para cada vizinhança de tamanho 'bloco'.
    # Usamos os parâmetros calculados automaticamente pela nossa função estatística.
    bin_adaptativa = cv2.adaptiveThreshold(
        img_cinza, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        stats_orig["sugestao_bloco"], 
        stats_orig["sugestao_c"]
    )

    # [PASSO 6] Cálculo de Métrica de Preenchimento (Área)
    # Como binarizamos com INV, o objeto é 255 (branco).
    pixels_objeto = cv2.countNonZero(bin_adaptativa)
    pixels_totais = bin_adaptativa.size
    porcentagem_area = (pixels_objeto / pixels_totais) * 100

    # =============================================================================
    # APRESENTAÇÃO DOS RESULTADOS (GRÁFICOS)
    # =============================================================================

    # JANELA 1: Comparação de Histogramas e Equalização
    plt.figure(figsize=(15, 8))
    plt.suptitle("Análise do Domínio do Valor: Antes e Depois da Equalização", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(img_cinza, cmap='gray')
    plt.title(f"Original (Desvio Padrão: {stats_orig['desvio']:.2f})")

    plt.subplot(2, 2, 2)
    plt.imshow(img_equalizada, cmap='gray')
    plt.title(f"Equalizada (Desvio Padrão: {stats_equa['desvio']:.2f})")

    plt.subplot(2, 2, 3)
    plt.plot(stats_orig["hist"], color='blue')
    plt.fill_between(range(256), stats_orig["hist"].flatten(), color='blue', alpha=0.3)
    plt.title("Histograma Original (Frequência de Tons)")
    plt.xlim([0, 256])

    plt.subplot(2, 2, 4)
    plt.plot(stats_equa["hist"], color='red')
    plt.fill_between(range(256), stats_equa["hist"].flatten(), color='red', alpha=0.3)
    plt.title("Histograma Equalizado (Distribuição Uniforme)")
    plt.xlim([0, 256])

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # JANELA 2: Comparação de Técnicas de Thresholding
    plt.figure(figsize=(15, 6))
    plt.suptitle("Segmentação: Otsu (Global) vs. Adaptativa (Local)", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.imshow(bin_otsu, cmap='gray')
    plt.title(f"Método de Otsu\nLimiar Automático: {limiar_otsu}")

    plt.subplot(1, 2, 2)
    plt.imshow(bin_adaptativa, cmap='gray')
    plt.title(f"Método Adaptativo\nBloco: {stats_orig['sugestao_bloco']} | C: {stats_orig['sugestao_c']}\nÁrea Ocupada: {porcentagem_area:.2f}%")

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # RELATÓRIO VIA CONSOLE
    print("\n" + "="*50)
    print("         RELATÓRIO TÉCNICO DE EXECUÇÃO")
    print("="*50)
    print(f"Resolução da Imagem: {stats_orig['resolucao'][0]}x{stats_orig['resolucao'][1]}")
    print(f"Média de Brilho Original: {stats_orig['media']:.2f}")
    print(f"Contraste Original (Desvio Padrão): {stats_orig['desvio']:.2f}")
    print("-" * 50)
    print(f"Parâmetros de Segmentação Calculados:")
    print(f" > Tamanho do Bloco Local: {stats_orig['sugestao_bloco']} pixels")
    print(f" > Constante de Sensibilidade (C): {stats_orig['sugestao_c']}")
    print("-" * 50)
    print(f"RESULTADO FINAL DE OCUPAÇÃO: {porcentagem_area:.2f}% da cena")
    print("="*50)

    plt.show()

# Ponto de entrada do script
if __name__ == "__main__":
    main()