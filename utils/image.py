import cv2
import tkinter as tk
from tkinter import filedialog
import sys

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