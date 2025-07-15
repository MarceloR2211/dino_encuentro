import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def mostrar_imagen(imagen, titulo="", cmap=None):
    plt.imshow(imagen, cmap=cmap)
    plt.title(titulo)
    plt.axis('off')

# Un píxel individual con sus componentes R, G, B
pasto = np.array([124, 252, 0], dtype=np.uint8)  # pasto
cielo = np.array([135, 206, 235], dtype=np.uint8)  # cielo
negro    = np.array([0, 0, 0], dtype=np.uint8)            # negro
morado   = np.array([104, 58, 183], dtype=np.uint8)       # Morado
verde   = np.array([104, 159, 56], dtype=np.uint8)       # Verde más opaco
blanco = np.array([255, 255, 255], dtype=np.uint8)  # 
amarillo = np.array([255, 255, 0], dtype=np.uint8)     # Amarillo puro
naranja  = np.array([255, 165, 0], dtype=np.uint8)     # Naranja (tono estándar)
a    = np.array([0, 0, 0], dtype=np.uint8)
b    = np.array([124, 252, 0], dtype=np.uint8)
c=np.array([255, 255, 255], dtype=np.uint8) 
d=np.array([0, 0, 0], dtype=np.uint8)

colores_fijos = {
    "pasto": pasto,
    "cielo": cielo,
    "negro": negro,
    "morado": morado,
    "verde": verde,
    "blanco": blanco,
    "amarillo": amarillo,
    "naranja": naranja
}

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
dino_template  = np.array([
    ["cielo", "cielo", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "cielo", "cielo"],
    ["cielo", "negro", "negro", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "negro", "cielo"],
    ["negro", "morado", "negro", "verde", "verde", "verde", "negro", "verde", "verde", "verde", "verde", "verde", "verde", "negro"],
    ["negro", "morado", "negro", "verde", "verde", "verde", "negro", "verde", "verde", "negro", "verde", "verde", "negro", "negro"],
    ["negro", "negro", "negro", "negro", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "negro"],
    ["cielo", "negro", "morado", "negro", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "negro"],
    ["cielo", "negro", "morado", "negro", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "verde", "negro"],
    ["cielo", "negro", "negro", "negro", "negro", "verde", "verde", "verde", "verde", "negro", "negro", "negro", "negro", "negro"],
    ["cielo", "cielo", "negro", "negro", "negro", "verde", "verde", "verde", "verde", "verde", "negro", "cielo", "cielo", "cielo"],
    ["cielo", "cielo", "negro", "verde", "negro", "verde", "verde", "verde", "verde", "verde", "negro", "cielo", "cielo", "cielo"],
    ["pasto", "pasto", "pasto", "negro", "negro", "verde", "verde", "verde", "verde", "verde", "negro", "pasto", "b", "pasto"],
    ["pasto", "pasto", "b", "b", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "b", "b", "pasto"],
    ["pasto", "pasto", "b", "pasto", "pasto", "a", "pasto", "pasto", "pasto", "a", "pasto", "pasto", "pasto", "pasto"],
    ["pasto", "pasto", "pasto", "pasto", "pasto", "a", "a", "pasto", "pasto", "a", "a", "pasto", "pasto", "pasto"]
], dtype=object) # uint8 para valores de 0 a 255
pesardo_template  = np.array([
    ["cielo", "cielo", "cielo", "cielo", "cielo", "cielo", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "cielo", "cielo", "cielo", "cielo", "cielo"],
    ["cielo", "cielo", "cielo", "cielo", "negro", "negro", "negro", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "negro", "negro", "negro", "cielo", "cielo", "cielo"],
    ["cielo", "cielo", "cielo", "negro", "negro", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "c", "c", "c", "negro", "negro", "cielo", "cielo"],
    ["negro", "negro", "negro", "negro", "negro", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "c", "c", "c", "c", "c", "negro", "cielo", "cielo"],
    ["negro", "naranja", "naranja", "naranja", "negro", "negro", "negro", "amarillo", "amarillo", "amarillo", "c", "c", "negro", "c", "c", "negro", "cielo", "cielo"],
    ["negro", "naranja", "naranja", "naranja", "naranja", "naranja", "negro", "negro", "amarillo", "amarillo", "c", "c", "c", "c", "negro", "negro", "negro", "cielo"],
    ["negro", "naranja", "naranja", "naranja", "naranja", "naranja", "naranja", "negro", "amarillo", "amarillo", "negro", "naranja", "naranja", "naranja", "naranja", "naranja", "naranja", "negro"],
    ["negro", "negro", "naranja", "naranja", "naranja", "naranja", "negro", "negro", "amarillo", "amarillo", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "cielo"],
    ["cielo", "negro", "negro", "negro", "negro", "negro", "negro", "amarillo", "amarillo", "negro", "naranja", "naranja", "naranja", "naranja", "naranja", "naranja", "negro", "cielo"],
    ["cielo", "cielo", "negro", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "negro", "negro", "negro", "negro", "negro", "negro", "negro", "cielo", "cielo"],
    ["cielo", "cielo", "negro", "negro", "negro", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "amarillo", "negro", "cielo", "cielo"],
    ["cielo", "cielo", "cielo", "cielo", "negro", "negro", "negro", "amarillo", "amarillo", "amarillo", "amarillo", "negro", "negro", "negro", "negro", "cielo", "cielo", "cielo"],
    ["cielo", "cielo", "cielo", "cielo", "cielo", "cielo", "negro", "negro", "negro", "negro", "negro", "negro", "cielo", "cielo", "cielo", "cielo", "cielo", "cielo"]
], dtype=object)

def construir_sprite(template):
    mapa = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        **colores_fijos
    }
    sprite_rgb = np.zeros(template.shape + (3,), dtype=np.uint8)
    for clave, color in mapa.items():
        sprite_rgb[template == clave] = color
    return np.kron(sprite_rgb, np.ones(factor, dtype=np.uint8))


def intercambiar_colores():
    global a, b, c, d
    a, b = b.copy(), a.copy()
    c, d = d.copy(), c.copy()

def generar_frame(paso):
    fondo_animado = fondo.copy()
    dino_sprite = construir_sprite(dino_template)
    pesardo_sprite = np.fliplr(construir_sprite(pesardo_template))

    col_dino = paso
    col_pesardo = fondo.shape[1] - paso - ancho_pesardo
    if col_dino + ancho_dino <= fondo.shape[1]:
        fondo_animado[fila_dino:fila_dino + alto_dino, col_dino:col_dino + ancho_dino] = dino_sprite
    if col_pesardo >= 0:
        fondo_animado[fila_pesardo:fila_pesardo + alto_pesardo, col_pesardo:col_pesardo + ancho_pesardo] = pesardo_sprite
    return fondo_animado

def preparar_frames():
    with Pool(processes=cpu_count()) as pool:
        frames = pool.map(generar_frame, range(N_pasos))
    return frames

if __name__ == "__main__":
    for i in range(N_pasos):
        intercambiar_colores()
        frame = generar_frame(i)
        mostrar_imagen(frame, f"Frame {i}")
        plt.pause(0.5)
        plt.close()
