import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys

    # RESPOSTAS PARA OS COMENTÁRIOS NO CÓDIGO (Questões do PDF):
    # 1. Usamos cv2.absdiff em vez de subtração (-) comum porque o NumPy faz 'wrap-around'
    #    em valores negativos (ex: 10 - 20 = 246 em uint8), o que geraria falsos alarmes de movimento.
    #    O absdiff garante o valor absoluto |10 - 20| = 10.
    
    # 2. No espaço HSV, a vantagem de filtrar pelo canal H (Hue) é que a cor é isolada
    #    da sua luminosidade. No RGB, sombras ou reflexos mudam os valores de R, G e B,
    #    dificultando o filtro. No HSV, 'Azul' continua sendo 'Azul' no canal H, mesmo na sombra.

# --- FUNÇÃO AUXILIAR PARA SELEÇÃO DE ARQUIVOS ---
def selecionar_imagem(titulo="Selecione uma imagem"):
    """Abre uma janela para selecionar arquivo e retorna a imagem lida pelo OpenCV."""
    print(titulo)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(title=titulo, 
                                        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")])
    root.destroy()
    if not caminho:
        print(f"Seleção cancelada para: {titulo}")
        sys.exit()
    img = cv2.imread(caminho)
    if img is None:
        print("Erro ao carregar a imagem.")
        sys.exit()
    return img

# --- EXERCÍCIO 1: ARITMÉTICA E ESTOURO ---
def executar_exercicio_1():
    print("\n--- Iniciando Exercício 1 ---")
    img_bgr = selecionar_imagem("Exercício 1: Selecione uma imagem colorida")
    
    # O Matplotlib usa RGB, o OpenCV usa BGR. Convertemos para visualização correta.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Criando matriz de mesma dimensão preenchida com 100 (uint8 é essencial aqui)
    # np.ones cria matriz de 1s, multiplicamos por 100 para brilho cinza.
    img_100 = np.ones(img_rgb.shape, dtype="uint8") * 100

    # SOMA VIA NUMPY (Operador +): Ocorre o "Wrap-around" (módulo 256).
    # Se 200 + 100 = 300, o resultado vira 44 (300 - 256). Gera ruído visual.
    soma_numpy = img_rgb + img_100

    # SOMA VIA OPENCV (cv2.add): Ocorre a "Saturação".
    # Se 200 + 100 = 300, o resultado é limitado ao teto de 255 (branco). Apenas clareia.
    soma_opencv = cv2.add(img_rgb, img_100)

    # Exibição dos resultados
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(soma_numpy); plt.title("Soma NumPy (Ruído/Wrap)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(soma_opencv); plt.title("Soma OpenCV (Saturação)"); plt.axis('off')
    plt.show()

# --- EXERCÍCIO 2: DETETIVE DE DIFERENÇAS ---
def executar_exercicio_2():
    print("\n--- Iniciando Exercício 2 ---")
    fundo_bgr = selecionar_imagem("Exercício 2: Selecione a imagem do FUNDO (vazio)")
    objeto_bgr = selecionar_imagem("Exercício 2: Selecione a imagem com o OBJETO")

    # Conversão para tons de cinza: Reduz 3 canais para 1, facilitando a comparação de intensidade.
    cinza_fundo = cv2.cvtColor(fundo_bgr, cv2.COLOR_BGR2GRAY)
    cinza_objeto = cv2.cvtColor(objeto_bgr, cv2.COLOR_BGR2GRAY)

    # cv2.absdiff: Calcula a diferença absoluta pixel a pixel |A - B|.
    # Diferente da subtração comum, evita valores negativos e estouros de uint8.
    mascara_diff = cv2.absdiff(cinza_fundo, cinza_objeto)

    # np.mean: Calcula a média de brilho da imagem de diferença.
    # Se a média for alta, muitos pixels mudaram significativamente.
    media = np.mean(mascara_diff)
    print(f"Média de diferença detectada: {media:.2f}")

    limiar = 5.0
    if media > limiar:
        print("ALERTA: Mudança detectada no cenário!")
    else:
        print("Cenário estático.")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cinza_fundo, cmap='gray'); plt.title("Fundo (Cinza)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(cinza_objeto, cmap='gray'); plt.title("Objeto (Cinza)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(mascara_diff, cmap='gray'); plt.title("Diferença (absdiff)"); plt.axis('off')
    plt.show()

# --- EXERCÍCIO 3: FILTRO SELETIVO (SIN CITY) ---
def executar_exercicio_3():
    print("\n--- Iniciando Exercício 3 ---")
    img_bgr = selecionar_imagem("Exercício 3: Selecione imagem com objeto colorido (Feito com azul Azul)")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Definição de limites para cor AZUL no espaço HSV do OpenCV
    # H (Matiz): 100-130 | S (Saturação): 50-255 | V (Brilho): 50-255
    low_blue = np.array([100, 50, 50])
    high_blue = np.array([130, 255, 255])

    # Criação da máscara binária (branco onde é azul, preto no resto)
    mask = cv2.inRange(img_hsv, low_blue, high_blue)

    # Passo 4: Isolar o objeto colorido usando a máscara
    objeto_colorido = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # Passo 5: Criar fundo PB e converter para 3 canais (RGB) para permitir a soma
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fundo_pb_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # Inverter a máscara para "furar" o fundo onde o objeto vai entrar
    mask_inv = cv2.bitwise_not(mask)
    fundo_com_furo = cv2.bitwise_and(fundo_pb_rgb, fundo_pb_rgb, mask=mask_inv)

    # Passo 6: Combinar objeto colorido com o fundo PB "furado"
    # Como as áreas são complementares (onde um é cor, o outro é 0), a soma funciona perfeitamente.
    sin_city = cv2.add(objeto_colorido, fundo_com_furo)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1); plt.imshow(img_rgb); plt.title("1. Original"); plt.axis('off')
    plt.subplot(2, 2, 2); plt.imshow(mask, cmap='gray'); plt.title("2. Máscara (inRange)"); plt.axis('off')
    plt.subplot(2, 2, 3); plt.imshow(objeto_colorido); plt.title("3. Objeto Isolado"); plt.axis('off')
    plt.subplot(2, 2, 4); plt.imshow(sin_city); plt.title("4. Efeito Sin City Final"); plt.axis('off')
    plt.show()

# --- BLOCO PRINCIPAL (MAIN) ---
if __name__ == "__main__":
    print("Iniciando Roteiro de Laboratório - Aula 02")
    
    executar_exercicio_1()
    executar_exercicio_2()
    executar_exercicio_3()

    print("\n--- Atividade Finalizada ---")

