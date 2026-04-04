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
    
    Heurística: Utilizamos o Desvio Padrão (contraste) para ditar o comportamento 
    dos filtros espaciais. Se o desvio é baixo, a imagem é "flat", então 
    aumentamos o tamanho do bloco para buscar variações locais mais sutis.
    
    Retorna um dicionário com: histograma, média, desvio padrão e 
    parâmetros sugeridos para binarização.
    """
    
    # 1. Cálculo do Histograma:
    # Conta a frequência de cada um dos 256 níveis de cinza.
    histograma = cv2.calcHist([img_cinza], [0], None, [256], [0, 256])

    # 2. Média Aritmética (Brilho Geral):
    media = np.mean(img_cinza)

    # 3. Desvio Padrão (Contraste):
    # Medida de dispersão. Alto desvio = histograma largo (bom contraste).
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
    img_original = selecionar_e_ler_imagem()
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # [PASSO 2] Análise Inicial do Domínio do Valor
    stats_orig = extrair_estatisticas_dominio_valor(img_cinza)

    # [PASSO 3] Melhoria de Imagem (Equalização de Histograma)
    img_equalizada = cv2.equalizeHist(img_cinza)
    stats_equa = extrair_estatisticas_dominio_valor(img_equalizada)

    # [PASSO 4] Binarização Global (Método de Otsu)
    limiar_otsu, bin_otsu = cv2.threshold(
        img_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # [PASSO 5] Binarização Adaptativa (Local)
    bin_adaptativa = cv2.adaptiveThreshold(
        img_cinza, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        stats_orig["sugestao_bloco"], 
        stats_orig["sugestao_c"]
    )

    # [PASSO 6] Cálculo de Métrica de Preenchimento (Área)
    pixels_objeto = cv2.countNonZero(bin_adaptativa)
    pixels_totais = bin_adaptativa.size
    porcentagem_area = (pixels_objeto / pixels_totais) * 100

    # =============================================================================
    # [PASSO 7] DESAFIO DE SIMILARIDADE (ROI E COMPARAÇÃO DE HISTOGRAMAS)
    # =============================================================================
    print("\n[DESAFIO] Selecione dois objetos na imagem com o mouse.")
    print("Instruções: Selecione a área do primeiro objeto e pressione ENTER. Repita para o segundo.")
    
    # Seleção de ROI usando interface do OpenCV
    roi1_box = cv2.selectROI("Selecione o Objeto 1 e pressione ENTER", img_cinza, fromCenter=False)
    x1, y1, w1, h1 = roi1_box
    obj1 = img_cinza[y1:y1+h1, x1:x1+w1]

    roi2_box = cv2.selectROI("Selecione o Objeto 2 e pressione ENTER", img_cinza, fromCenter=False)
    x2, y2, w2, h2 = roi2_box
    obj2 = img_cinza[y2:y2+h2, x2:x2+w2]
    
    cv2.destroyAllWindows()

    # Cálculo e Normalização dos Histogramas para Comparação
    hist_obj1 = cv2.calcHist([obj1], [0], None, [256], [0, 256])
    hist_obj2 = cv2.calcHist([obj2], [0], None, [256], [0, 256])
    
    # Normalizamos para que a comparação seja justa mesmo com ROIs de tamanhos diferentes
    cv2.normalize(hist_obj1, hist_obj1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_obj2, hist_obj2, 0, 1, cv2.NORM_MINMAX)
    
    # Comparação usando Correlação (1.0 = Idêntico, 0.0 = Diferente)
    similaridade = cv2.compareHist(hist_obj1, hist_obj2, cv2.HISTCMP_CORREL)

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

    # JANELA 3: Desafio de Similaridade (ROI)
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Desafio de Similaridade - Índice de Correlação: {similaridade:.4f}", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(obj1, cmap='gray')
    plt.title("Objeto 1 (Recorte)")

    plt.subplot(2, 2, 2)
    plt.imshow(obj2, cmap='gray')
    plt.title("Objeto 2 (Recorte)")

    plt.subplot(2, 2, 3)
    plt.plot(hist_obj1, color='green')
    plt.title("Histograma Normalizado - Objeto 1")

    plt.subplot(2, 2, 4)
    plt.plot(hist_obj2, color='orange')
    plt.title("Histograma Normalizado - Objeto 2")

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
    print(f"NÍVEL DE SIMILARIDADE (ROI): {similaridade:.4f}")
    print("="*50)

    plt.show()

# Ponto de entrada do script
if __name__ == "__main__":
    main()