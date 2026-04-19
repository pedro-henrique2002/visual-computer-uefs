# =============================================================================
# Aluno: Pedro Henrique de Araújo Silva
# =============================================================================
"""
QUESTÕES DE DISCUSSÃO
 
Q1. Por que o Blur é obrigatório antes do Sobel?
    Sem o Blur, pixels de ruído isolados geram bordas falsas.
    O Blur suaviza esses pontos antes de calcular o gradiente.
 
Q2. Objeto branco com buraco preto: qual operação fecha o buraco?
    Fechamento. Ele expande o branco para dentro do buraco e depois
    encolhe de volta — o buraco some.
 
Q3. No Canny, para que servem os dois limiares (histerese)?
    Um limiar só obriga a escolher entre detectar demais ou de menos.
    Com dois limiares: pixels muito fortes são borda, pixels muito fracos
    são ruído, e os do meio só entram se estiverem ligados a uma borda forte.
 
Q4. No Exercício 3, qual contorno é mais grosso e por quê?
    O Gradiente Morfológico. Ele subtrai a imagem encolhida da expandida,
    gerando uma faixa larga. O Canny mantém só o pico do gradiente,
    resultando em 1 pixel de largura.
    """
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def selecionar_e_ler_imagem():
    """Abre janela de seleção de arquivo e retorna a imagem em BGR."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(
        title="Selecione a imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()
    if not caminho:
        print("Operação cancelada.")
        sys.exit()
    imagem = cv2.imread(caminho)
    if imagem is None:
        print("Erro: não foi possível carregar a imagem.")
        sys.exit()
    return imagem


def gerar_ruido_sal_pimenta(img, probabilidade=0.05):
    """Adiciona pixels aleatórios brancos (sal) e pretos (pimenta) à imagem."""
    ruidosa = np.copy(img)
    ruidosa[np.random.rand(*img.shape) < probabilidade / 2] = 255
    ruidosa[np.random.rand(*img.shape) < probabilidade / 2] = 0
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

def exercicio_1(img_cinza):
    # Gera imagem ruidosa para simular cenário adverso
    img_ruidosa = gerar_ruido_sal_pimenta(img_cinza, probabilidade=0.08)

    # Sobel aplicado direto na imagem ruidosa — sem suavização prévia
    # O ruído gera gradientes falsos, "sujando" o resultado
    sobelX = cv2.Sobel(img_ruidosa, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img_ruidosa, cv2.CV_64F, 0, 1, ksize=3)
    sobel_ruidoso = np.clip(cv2.magnitude(sobelX, sobelY), 0, 255).astype(np.uint8)

    # Canny com suavização prévia — o Blur elimina o ruído antes do cálculo
    # A histerese (30/100) separa bordas fortes de candidatas fracas
    img_blur = cv2.GaussianBlur(img_ruidosa, (5, 5), 0)
    canny_calibrado = cv2.Canny(img_blur, threshold1=30, threshold2=100)

    mostrar(
        [img_cinza, img_ruidosa, sobel_ruidoso, canny_calibrado],
        ["Original",
         "Com Ruído Salt & Pepper",
         "Sobel — sem suavização\n(ruído contamina o gradiente)",
         "Canny — com Blur 5×5\n(bordas limpas)"],
        colunas=4, figsize=(20, 5),
        titulo_geral="Exercício 1 — Sobel Ruidoso × Canny Calibrado"
    )


# =============================================================================
# EXERCÍCIO 2 — "Cirurgia" Morfológica
# =============================================================================

def exercicio_2():
    # Gera imagem binária de texto com dois problemas propositais:
    # 1) Pontos brancos no fundo (ruído salt-and-pepper)
    # 2) Buracos/cortes dentro das letras
    largura, altura = 640, 160
    img_base = np.zeros((altura, largura), dtype=np.uint8)
    cv2.putText(img_base, "MORFOLOGIA", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, 255, 8, cv2.LINE_AA)

    img_prob = img_base.copy()
    img_prob[(np.random.rand(altura, largura) < 0.04) & (img_base == 0).__array__()] = 255  # ruído no fundo
    img_prob[(np.random.rand(altura, largura) < 0.10) & (img_base == 255)] = 0             # buracos nas letras

    # Abertura (Erosão → Dilatação): remove ruído externo menor que o kernel
    kernel_ab = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    apos_abertura = cv2.morphologyEx(img_prob, cv2.MORPH_OPEN, kernel_ab)

    # Fechamento (Dilatação → Erosão): preenche buracos internos menores que o kernel
    kernel_fe = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    apos_fechamento = cv2.morphologyEx(apos_abertura, cv2.MORPH_CLOSE, kernel_fe)

    mostrar(
        [img_prob, apos_abertura, apos_fechamento],
        ["Imagem Problemática\n(ruído + letras furadas)",
         "Após Abertura (3×3)\n(ruído externo removido)",
         "Após Fechamento (5×5)\n(letras restauradas)"],
        colunas=3, figsize=(18, 5),
        titulo_geral="Exercício 2 — Cirurgia Morfológica"
    )


# =============================================================================
# EXERCÍCIO 3 — Gradiente Morfológico × Canny
# =============================================================================

def exercicio_3(img_cinza):
    # Binariza com limiar de Otsu para isolar o objeto do fundo
    _, img_bin = cv2.threshold(img_cinza, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Gradiente Morfológico = Dilatação − Erosão
    # Produz o contorno do objeto; espessura proporcional ao tamanho do kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradiente = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)

    # Canny com suavização: supressão de não-máximos afina bordas para 1 pixel
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    canny = cv2.Canny(img_blur, threshold1=30, threshold2=100)

    mostrar(
        [img_cinza, gradiente, canny],
        ["Objeto Original",
         "Gradiente Morfológico\n(contorno mais grosso — kernel 5×5)",
         "Canny\n(contorno fino — supressão de não-máximos)"],
        colunas=3, figsize=(18, 6),
        titulo_geral="Exercício 3 — Gradiente Morfológico × Canny"
    )


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    img_bgr   = selecionar_e_ler_imagem()
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"Imagem carregada: {img_bgr.shape[1]}×{img_bgr.shape[0]} px")
    print("Feche cada janela para avançar ao próximo exercício.\n")

    exercicio_1(img_cinza)
    exercicio_2()
    exercicio_3(img_cinza)