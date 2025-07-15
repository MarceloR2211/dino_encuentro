import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def mostrar_imagen(imagen, titulo="", cmap=None):
    """
    Muestra una matriz NumPy como una imagen usando Matplotlib.
    """
    #plt.figure(figsize=(7, 7)) # Ajusta el tamaño de la figura
    #plt.imshow(imagen, cmap=cmap)
    plt.imshow(imagen, cmap=cmap)
    plt.title(titulo)
    # plt.colorbar(label="Intensidad del Píxel") # Muestra la barra de color
    plt.axis('off') # Oculta los ejes para una visualización más limpia
    #plt.show()


# Un píxel individual con sus componentes R, G, B
rojo = np.array( [255, 0, 0], dtype=np.uint8) # Rojo puro
verde = np.array([0, 255, 0], dtype=np.uint8) # Verde puro
azul = np.array( [0, 0, 255], dtype=np.uint8) # Azul puro
pasto = np.array([124, 252, 0], dtype=np.uint8)  # pasto
cielo = np.array([135, 206, 235], dtype=np.uint8)  # cielo
negro    = np.array([0, 0, 0], dtype=np.uint8)            # negro
#verde1   = np.array([100, 221, 23], dtype=np.uint8)       # Verde brillante
morado   = np.array([104, 58, 183], dtype=np.uint8)       # Morado
verde   = np.array([104, 159, 56], dtype=np.uint8)       # Verde más opaco
blanco = np.array([255, 255, 255], dtype=np.uint8)  # 
amarillo = np.array([255, 255, 0], dtype=np.uint8)     # Amarillo puro
naranja  = np.array([255, 165, 0], dtype=np.uint8)     # Naranja (tono estándar)
a    = np.array([0, 0, 0], dtype=np.uint8)
b    = np.array([124, 252, 0], dtype=np.uint8)
c=np.array([255, 255, 255], dtype=np.uint8) 
d=np.array([0, 0, 0], dtype=np.uint8)

fondo = np.tile(cielo,(300,900,1))
fondo[-40:]=pasto

fila_dino = 160
fila_pesardo = 0
N_pasos = 600
ancho_dino=14*10
alto_dino=14*10
alto_pesardo=13*10
ancho_pesardo=18*10
factor = (10, 10, 1)  # duplicar en alto y ancho, pero no en el canal de color
dino = np.array([
    [cielo, cielo, negro, negro, negro, negro, negro, negro, negro, negro, negro, negro, cielo, cielo],
    [cielo, negro, negro,verde,verde,verde,verde,verde,verde,verde,verde,verde, negro, cielo],
    [negro,morado, negro,verde,verde,verde, negro,verde,verde,verde,verde,verde,verde, negro],
    [negro,morado, negro,verde,verde,verde, negro,verde,verde, negro,verde,verde, negro, negro],
    [negro, negro, negro, negro,verde,verde,verde,verde,verde,verde,verde,verde,verde, negro],
    [cielo, negro,morado, negro,verde,verde,verde,verde,verde,verde,verde,verde,verde, negro],
    [cielo, negro,morado, negro,verde,verde,verde,verde,verde,verde,verde,verde,verde, negro],
    [cielo, negro, negro, negro, negro,verde,verde,verde,verde, negro, negro, negro, negro, negro],
    [cielo, cielo, negro, negro, negro,verde,verde,verde,verde,verde, negro, cielo, cielo, cielo],
    [cielo, cielo, negro, verde, negro,verde,verde,verde,verde,verde, negro, cielo, cielo, cielo],
    [pasto, pasto, pasto, negro, negro,verde,verde,verde,verde,verde, negro, pasto, b, pasto],
    [pasto, pasto, b,b, negro,negro,negro, negro,negro,negro, negro, b, b, pasto],
    [pasto, pasto, b, pasto, pasto,a, pasto, pasto, pasto,a, pasto, pasto, pasto, pasto],
    [pasto, pasto, pasto, pasto, pasto, a, a, pasto, pasto, a, a, pasto, pasto, pasto]
    ], dtype=np.uint8) # uint8 para valores de 0 a 255
dino = np.kron(dino, np.ones(factor, dtype=np.uint8))
pesardo = np.array([
    [cielo, cielo, cielo, cielo, cielo, cielo, negro, negro, negro, negro, negro, negro, negro, cielo, cielo, cielo, cielo, cielo],
    [cielo, cielo, cielo, cielo, negro, negro, negro, amarillo, amarillo, amarillo, amarillo, amarillo, negro, negro, negro, cielo, cielo, cielo],
    [cielo, cielo, cielo, negro, negro, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, c, c, c, negro, negro, cielo, cielo],
    [negro, negro, negro, negro, negro, amarillo, amarillo, amarillo, amarillo, amarillo, c, c, c, c, c, negro, cielo, cielo],
    [negro, naranja, naranja, naranja, negro, negro, negro, amarillo, amarillo, amarillo, c, c, negro, c, c, negro, cielo, cielo],
    [negro, naranja, naranja, naranja, naranja, naranja, negro, negro, amarillo, amarillo, c, c, c, c, negro, negro, negro, cielo],
    [negro, naranja, naranja, naranja, naranja, naranja, naranja, negro, amarillo, amarillo, negro, naranja, naranja, naranja, naranja, naranja, naranja, negro],
    [negro, negro, naranja, naranja, naranja, naranja, negro, negro, amarillo, amarillo, negro, negro, negro, negro, negro, negro, negro, cielo],
    [cielo, negro, negro, negro, negro, negro, negro, amarillo, amarillo, negro, naranja, naranja, naranja, naranja, naranja, naranja, negro, cielo],
    [cielo, cielo, negro, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, negro, negro, negro, negro, negro, negro, negro, cielo, cielo],
    [cielo, cielo, negro, negro, negro, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, amarillo, negro, cielo, cielo],
    [cielo, cielo, cielo, cielo, negro, negro, negro, amarillo, amarillo, amarillo, amarillo, negro, negro, negro, negro, cielo, cielo, cielo],
    [cielo, cielo, cielo, cielo, cielo, cielo, negro, negro, negro, negro, negro, negro, cielo, cielo, cielo, cielo, cielo, cielo]
    ], dtype=np.uint8)
pesardo = np.fliplr(pesardo)
pesardo = np.kron(pesardo, np.ones(factor, dtype=np.uint8))

def generar_frame(paso):
    fondo_animado = fondo.copy()

    # Movimiento Dino
    col_dino = paso
    if col_dino + ancho_dino <= fondo.shape[1]:
        fondo_animado[fila_dino:fila_dino+alto_dino, col_dino:col_dino+ancho_dino] = dino

    # Movimiento Pesardo
    col_pesardo = fondo.shape[1] - paso - ancho_pesardo
    if col_pesardo >= 0:
        fondo_animado[fila_pesardo:fila_pesardo+alto_pesardo, col_pesardo:col_pesardo+ancho_pesardo] = pesardo

    return fondo_animado

if __name__ == "__main__":
    N_pasos = 600

    with Pool(processes=cpu_count()) as pool:
        frames = pool.map(generar_frame, range(N_pasos))

    # Mostrar los frames uno por uno
    for i, frame in enumerate(frames):
        mostrar_imagen(frame, f"Frame {i}")
        plt.pause(0.5)  
        plt.close()         
        