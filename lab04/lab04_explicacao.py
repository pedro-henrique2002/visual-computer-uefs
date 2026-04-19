# =============================================================================
# TEC434 - Computação Visual | Aula 04
# Detecção de Bordas e Morfologia Matemática
# Versão LOCAL (roda no seu computador, sem Google Colab)
# =============================================================================
# Dependências: pip install opencv-python numpy matplotlib
# =============================================================================

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
    root.withdraw()
    root.attributes("-topmost", True)

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
# FUNÇÕES AUXILIARES
# =============================================================================

def para_cinza(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def gerar_ruido_sal_pimenta(img, probabilidade=0.05):
    """
    Adiciona ruído Salt & Pepper (sal e pimenta) à imagem.

    COMO FUNCIONA NO CÓDIGO?
    - np.random.rand(*img.shape) cria uma matriz de valores entre 0 e 1
      com o mesmo tamanho da imagem.
    - Pixels onde esse valor < prob/2 viram brancos (sal).
    - Pixels onde esse valor < prob/2 em outra sorteio viram pretos (pimenta).
    - O resultado é aproximadamente 'probabilidade'*100% de pixels afetados.
    """
    ruidosa = np.copy(img)  # Copia para não modificar o original
    pixels_sal     = np.random.rand(*img.shape) < (probabilidade / 2)
    pixels_pimenta = np.random.rand(*img.shape) < (probabilidade / 2)
    ruidosa[pixels_sal]     = 255
    ruidosa[pixels_pimenta] = 0
    return ruidosa


def mostrar(imagens, titulos, colunas=3, figsize=(18, 6), titulo_geral=None):
    """Exibe lista de imagens em grade com matplotlib."""
    linhas = (len(imagens) + colunas - 1) // colunas
    fig, axes = plt.subplots(linhas, colunas, figsize=figsize)
    axes = np.array(axes).flatten()
    if titulo_geral:
        fig.suptitle(titulo_geral, fontsize=14, fontweight='bold')
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titulo, fontsize=10)
        axes[i].axis('off')
    for j in range(len(imagens), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXERCÍCIO 1 — O Detetive de Bordas
# =============================================================================
# CONCEITO CENTRAL: DETECÇÃO DE BORDAS
#
# Uma "borda" em imagem é uma região onde a intensidade muda bruscamente.
# Matematicamente, isso corresponde a um alto valor de DERIVADA (gradiente).
#
# O problema: ruído também causa variações bruscas de intensidade!
# Por isso, antes de calcular bordas, precisamos SUAVIZAR a imagem.
#
# Neste exercício você vai ver:
#  1. O Sobel SEM suavização → capta bordas E ruído (resultado sujo)
#  2. O Canny COM suavização → consegue separar bordas reais do ruído
# =============================================================================

def exercicio_1(img_cinza):
    print("\n" + "="*60)
    print("EXERCÍCIO 1 — O Detetive de Bordas")
    print("="*60)

    # ── Etapa 1: Criar imagem ruidosa
    # Adicionamos ruído proposital para simular um cenário difícil.
    img_ruidosa = gerar_ruido_sal_pimenta(img_cinza, probabilidade=0.08)

    # ── Etapa 2: Aplicar o OPERADOR SOBEL
    #
    # O Sobel é um filtro de derivada. Ele detecta bordas calculando quanto
    # a intensidade muda em cada direção (horizontal e vertical).
    #
    # cv2.Sobel(imagem, profundidade_saída, derivada_x, derivada_y, ksize)
    #   - CV_64F: saída em ponto flutuante (pode ter valores negativos!)
    #   - dx=1, dy=0: calcula variação HORIZONTAL (bordas verticais)
    #   - dx=0, dy=1: calcula variação VERTICAL   (bordas horizontais)
    #   - ksize=3: janela 3x3 pixels ao redor de cada ponto
    #
    # A MAGNITUDE combina os dois gradientes: sqrt(Gx² + Gy²)
    # Isso nos dá a "força" da borda em qualquer direção.
    #
    # PROBLEMA: O Sobel não sabe distinguir uma borda real de um pixel de ruído.
    # Um pixel branco isolado (sal) gera gradiente altíssimo nos 4 vizinhos!

    sobelX = cv2.Sobel(img_ruidosa, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente horizontal
    sobelY = cv2.Sobel(img_ruidosa, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente vertical
    sobel_magnitude = cv2.magnitude(sobelX, sobelY)
    sobel_resultado = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)

    # ── Etapa 3: Aplicar o CANNY com suavização prévia ───────────────────────
    #
    # O Canny é um algoritmo de múltiplos estágios. Internamente ele faz:
    #
    # PASSO A — Suavização Gaussiana:
    #   Remove o ruído de alta frequência antes de calcular qualquer gradiente.
    #   Aqui fazemos manualmente para ter controle total do kernel.
    #   Um kernel 5x5 suaviza mais (bom para imagens muito ruidosas).
    #
    # PASSO B — Cálculo do Gradiente (igual ao Sobel internamente):
    #   Calcula Gx, Gy e a magnitude em cada pixel.
    #
    # PASSO C — Supressão de Não-Máximos (Non-Maximum Suppression):
    #   Para cada pixel, olha os vizinhos na direção do gradiente.
    #   Se o pixel atual NÃO for o valor máximo local, ele é zerado.
    #   Resultado: bordas com exatamente 1 pixel de espessura.
    #
    # PASSO D — Histerese (limiarização dupla):
    #   Define dois limiares: T_low e T_high.
    #   - Pixels com gradiente > T_high: BORDA FORTE (aceita)
    #   - Pixels com gradiente < T_low:  RUÍDO (descartado)
    #   - Pixels entre T_low e T_high:   aceitos SOMENTE se conectados a uma borda forte
    #   Isso evita bordas fragmentadas E evita incluir ruído.

    img_suavizada  = cv2.GaussianBlur(img_ruidosa, (5, 5), 0)
    canny_resultado = cv2.Canny(img_suavizada, threshold1=30, threshold2=100)

    # ── Exibição ─────────────────────────────────────────────────────────────
    mostrar(
        [img_cinza, img_ruidosa, sobel_resultado, canny_resultado],
        ["Original (sem ruído)",
         "Com Ruído Salt & Pepper\n(8% dos pixels afetados)",
         "Sobel na Imagem Ruidosa\n⚠ Ruído 'explode' no gradiente",
         "Canny Calibrado\n✓ Bordas limpas apesar do ruído"],
        colunas=4,
        figsize=(20, 5),
        titulo_geral="Exercício 1 — O Detetive de Bordas: Sobel × Canny"
    )
    print("✅ Exercício 1 concluído.")


# =============================================================================
# EXERCÍCIO 2 — "Cirurgia" Morfológica
# =============================================================================
# CONCEITO CENTRAL: MORFOLOGIA MATEMÁTICA
#
# A morfologia opera sobre imagens binárias (preto ou branco) usando um
# "elemento estruturante" — uma pequena forma (quadrado, círculo, etc.)
# que serve como "sonda" deslizando pela imagem.
#
# As duas operações base:
#
#   EROSÃO: um pixel branco SOBREVIVE apenas se TODOS os pixels sob
#           o elemento estruturante também forem brancos.
#           Efeito: objetos brancos ENCOLHEM; pequenos objetos SOMEM.
#
#   DILATAÇÃO: um pixel preto vira branco se QUALQUER pixel sob
#              o elemento estruturante for branco.
#              Efeito: objetos brancos CRESCEM; buracos pequenos FECHAM.
#
# Combinando as duas:
#
#   ABERTURA  = Erosão → Dilatação
#     Remove objetos pequenos (ruído) sem alterar muito os objetos grandes.
#     O objeto encolhe (erosão apaga ruídos e pontas finas) e depois
#     volta ao tamanho (dilatação), mas os ruídos pequenos não voltam.
#
#   FECHAMENTO = Dilatação → Erosão
#     Fecha buracos e cortes internos.
#     O objeto cresce (dilatação fecha buracos) e depois volta ao tamanho
#     (erosão), mas os buracos pequenos não reaparecem.
# =============================================================================

def exercicio_2():
    print("\n" + "="*60)
    print("EXERCÍCIO 2 — 'Cirurgia' Morfológica")
    print("="*60)

    # ── Etapa 1: Gerar imagem binária com dois problemas ─────────────────────
    #
    # Problema 1 → Pontos brancos no FUNDO (ruído salt no fundo preto)
    #              Simula uma segmentação imperfeita onde "sobrou lixo" fora do objeto.
    #
    # Problema 2 → Buracos/cortes nas LETRAS
    #              Simula letras mal segmentadas, com partes faltando.

    largura, altura = 640, 160
    img_base = np.zeros((altura, largura), dtype=np.uint8)
    cv2.putText(img_base, "MORFOLOGIA", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, 255, 8, cv2.LINE_AA)

    img_problematica = img_base.copy()

    # Adiciona ruído (pontos brancos) SOMENTE no fundo preto
    mascara_ruido = np.random.rand(altura, largura) < 0.04
    img_problematica[mascara_ruido & (img_base == 0)] = 255

    # Adiciona buracos (pontos pretos) SOMENTE dentro das letras brancas
    mascara_buracos = np.random.rand(altura, largura) < 0.10
    img_problematica[mascara_buracos & (img_base == 255)] = 0

    # ── Etapa 2: ABERTURA — elimina pontos brancos no fundo ──────────────────
    #
    # Usamos um elemento estruturante RETANGULAR 3×3.
    # Os pontos de ruído (1 ou 2 pixels) são MENORES que o elemento → a Erosão
    # os apaga completamente. As letras são muito maiores → sobrevivem à erosão
    # e voltam ao tamanho com a dilatação subsequente.
    #
    # Resumo intuitivo:
    # "Se o objeto é menor que minha sonda, ele some."

    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_apos_abertura = cv2.morphologyEx(img_problematica, cv2.MORPH_OPEN, kernel_abertura)

    # ── Etapa 3: FECHAMENTO — fecha buracos e cortes nas letras ──────────────
    #
    # Usamos um elemento estruturante maior (5×5) para cobrir os buracos.
    # A dilatação "preenche" os buracos internos; a erosão depois reduz o
    # contorno de volta ao original, mas os buracos preenchidos não reaparecem.
    #
    # Resumo intuitivo:
    # "Se o buraco é menor que minha sonda, ele some."

    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_apos_fechamento = cv2.morphologyEx(img_apos_abertura, cv2.MORPH_CLOSE, kernel_fechamento)

    # ── Exibição ─────────────────────────────────────────────────────────────
    mostrar(
        [img_problematica, img_apos_abertura, img_apos_fechamento],
        ["2.1 — Imagem Problemática\n(ruído externo + letras furadas)",
         "2.2 — Após ABERTURA (3×3)\n✓ Ruído do fundo removido",
         "2.3 — Após FECHAMENTO (5×5)\n✓ Letras restauradas"],
        colunas=3,
        figsize=(18, 5),
        titulo_geral="Exercício 2 — Cirurgia Morfológica: Abertura + Fechamento"
    )
    print("✅ Exercício 2 concluído.")


# =============================================================================
# EXERCÍCIO 3 — O Desafio do "Esqueleto" (Gradiente Morfológico)
# =============================================================================
# CONCEITO CENTRAL: GRADIENTE MORFOLÓGICO
#
# O Gradiente Morfológico é uma operação que extrai o CONTORNO de objetos
# usando apenas morfologia (sem nenhum cálculo de derivada numérica).
#
# Fórmula:  Gradiente = Dilatação(img) - Erosão(img)
#
# Intuição:
#   - A dilatação "infla" o objeto → a borda externa fica fora do original
#   - A erosão  "deflate" o objeto → a borda interna fica dentro do original
#   - A diferença entre os dois é exatamente a FAIXA de borda
#
# Comparação com Canny:
#   - Gradiente Morfológico: mais GROSSO (espessura = tamanho do kernel)
#   - Canny: mais FINO (supressão de não-máximos garante 1 pixel de largura)
#
# Quando usar cada um:
#   - Canny: quando precisa de bordas precisas e finas (ex: medir objetos)
#   - Gradiente Morfológico: quando quer contorno robusto, fácil de conectar
# =============================================================================

def exercicio_3(img_cinza):
    print("\n" + "="*60)
    print("EXERCÍCIO 3 — Gradiente Morfológico × Canny")
    print("="*60)

    # ── Etapa 1: Binarizar o objeto ──────────────────────────────────────────
    #
    # A morfologia clássica opera sobre imagens BINÁRIAS.
    # O limiar de Otsu calcula automaticamente o melhor valor de corte para
    # separar fundo (preto) de objeto (branco) sem precisarmos adivinhar.
    # cv2.THRESH_OTSU analisa o histograma e encontra o vale entre os dois picos.

    _, img_binaria = cv2.threshold(img_cinza, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Etapa 2: GRADIENTE MORFOLÓGICO ──────────────────────────────────────
    #
    # Elemento estruturante ELÍPTICO 5×5 (mais suave que retangular).
    # Quanto maior o kernel, mais grosso será o contorno resultante.
    # cv2.MORPH_GRADIENT aplica a fórmula Dilatação − Erosão diretamente.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradiente_morfologico = cv2.morphologyEx(img_binaria, cv2.MORPH_GRADIENT, kernel)

    # ── Etapa 3: CONTORNO COM CANNY ──────────────────────────────────────────
    #
    # Suavização obrigatória antes do Canny (ver questão de discussão 1).
    # Os limiares 30/100 seguem a regra prática: T_high ≈ 3 × T_low.

    img_suavizada = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    contorno_canny = cv2.Canny(img_suavizada, threshold1=30, threshold2=100)

    # ── Exibição ─────────────────────────────────────────────────────────────
    mostrar(
        [img_cinza, gradiente_morfologico, contorno_canny],
        ["3.1 — Objeto Original",
         "3.2 — Gradiente Morfológico\n⚠ Contorno mais GROSSO\n(espessura = tamanho do kernel)",
         "3.3 — Canny\n✓ Contorno mais FINO\n(supressão de não-máximos)"],
        colunas=3,
        figsize=(18, 6),
        titulo_geral="Exercício 3 — Comparação: Gradiente Morfológico × Canny"
    )
    print("✅ Exercício 3 concluído.")


# =============================================================================
# RESPOSTAS ÀS QUESTÕES DE DISCUSSÃO
# =============================================================================

def imprimir_respostas():
    texto = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              QUESTÕES DE DISCUSSÃO — Aula 04                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Q1. Por que o Blur é quase obrigatório antes do Sobel/Gradiente?           ║
║                                                                              ║
║  O Sobel calcula derivadas de 1ª ordem. Ruído gera variações bruscas de     ║
║  intensidade em pixels isolados, que aparecem como "bordas falsas" com      ║
║  gradiente altíssimo. O Blur suaviza essas variações locais, preservando    ║
║  apenas as transições amplas que correspondem a bordas reais de objetos.    ║
║  Matematicamente: derivar uma função ruidosa amplifica o ruído; suavizar    ║
║  primeiro (convolução com Gaussiana) estabiliza o resultado.                ║
║                                                                              ║
║  Q2. Objeto branco com buraco preto: qual operação morfológica remove?      ║
║                                                                              ║
║  FECHAMENTO (Closing = Dilatação → Erosão).                                 ║
║  A dilatação cresce o objeto branco para dentro do buraco preto,            ║
║  preenchendo-o. A erosão subsequente encolhe o contorno externo de          ║
║  volta ao original, mas o buraco interno (agora preenchido) não volta.      ║
║  Condição: o buraco deve ser menor que o elemento estruturante.             ║
║                                                                              ║
║  Q3. No Canny, qual é a função pedagógica da Histerese (dois limiares)?    ║
║                                                                              ║
║  A histerese resolve um dilema fundamental:                                 ║
║  • Limiar único alto → perde bordas fracas reais (bordas fragmentadas)     ║
║  • Limiar único baixo → inclui ruído como borda (excesso de detalhes)      ║
║                                                                              ║
║  Com dois limiares (T_low, T_high):                                         ║
║  1. Pixels acima de T_high → BORDA FORTE: certamente é borda               ║
║  2. Pixels abaixo de T_low → RUÍDO: certamente não é borda                 ║
║  3. Pixels entre os dois   → BORDA CANDIDATA: só aceita se estiver         ║
║     conectada (vizinha) a uma borda forte                                   ║
║                                                                              ║
║  Isso mantém a continuidade das bordas reais sem incluir ruído isolado.     ║
║  Regra prática: T_high ≈ 2 a 3 vezes T_low.                               ║
║                                                                              ║
║  Q4. No Exercício 3, qual contorno é mais grosso e por quê?                 ║
║                                                                              ║
║  O GRADIENTE MORFOLÓGICO produz contornos mais grossos.                     ║
║  A largura do contorno = 2 × raio do elemento estruturante, porque:        ║
║  • A dilatação expande o objeto ALÉM da borda real                          ║
║  • A erosão encolhe o objeto AQUÉM da borda real                            ║
║  • A diferença é uma faixa de pixels ao redor de toda a borda              ║
║                                                                              ║
║  O Canny aplica Supressão de Não-Máximos: para cada pixel de gradiente,    ║
║  compara com os vizinhos na direção do gradiente. Apenas o pixel com o     ║
║  valor máximo local sobrevive → bordas de exatamente 1 pixel de largura.   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(texto)


# =============================================================================
# PONTO DE ENTRADA — Execução Principal
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║   TEC434 - Computação Visual | Aula 04              ║")
    print("║   Detecção de Bordas e Morfologia Matemática        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\n📂 Selecione a imagem na janela que vai abrir...")

    # Carrega a imagem via Tkinter (a função do enunciado)
    img_bgr   = selecionar_e_ler_imagem()
    img_cinza = para_cinza(img_bgr)

    print(f"✅ Imagem carregada! Tamanho: {img_bgr.shape[1]}×{img_bgr.shape[0]} px")
    print("   Executando os 3 exercícios em sequência...\n")
    print("   (Feche cada janela de gráfico para avançar ao próximo exercício)\n")

    exercicio_1(img_cinza)
    exercicio_2()            # Gera o texto sintético internamente
    exercicio_3(img_cinza)
    imprimir_respostas()

    print("\n✅ Todos os exercícios concluídos!")