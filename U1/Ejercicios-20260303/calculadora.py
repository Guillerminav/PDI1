import cv2
import matplotlib.pyplot as plt
import numpy as np
from roipoly import RoiPoly


# Cargar la imagen desde el archivo img_calculadora.tif y mostrarla en una figura. 

img = cv2.imread('./img_calculadora.tif',cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)

# Determinar sus dimensiones y el tipo de dato con el cual se representa el valor de cada píxel. 

type(img)
img.dtype # [0, 255] uint8
img.shape # (1134, 1360)
h,w = img.shape

# Determinar el valor mínimo y máximo del nivel de grises de la imagen. 
img.min() # 12
img.max() # 255

#Hallar todos los valores de nivel de grises que tiene la imagen. ¿Cuántos son? 

pix_vals = np.unique(img)
N_pix_vals = len(np.unique(img)) # 244

# ¿Cuál es el valor de nivel de gris con menor repetitividad? ¿Cuál es el valor de nivel  de gris con mayor repetitividad? Considere que, en ambos casos, pueden ser más  de uno. En tal caso, mostrarlos todos. 

img.argmin()

cant_grises, counts = np.unique(img, return_counts=True)
indmax = np.argmax(counts)
indmin = np.argmin(counts)

nivelrepetido = cant_grises[indmax] # 71 nivel de gris
cantmax = counts[indmax] # 39933 cant de apariciones del nivel de gris 71
nivelminrepetido = cant_grises[indmin] # 12 nivel de gris
cantmin = counts[indmin] # 1 cant de apariciones del nivel de gris 12

#  ROI

sin = img[324:725,443:1335]
cos = img[324:725,443:1335]
tan = img[324:725,443:1335]
plt.figure(), plt.imshow(sin, cmap='gray'), plt.show(block=False)


# --- Cargo imagen y selecciono ROI --------------------------------------
img = cv2.imread("./img_calculadora.tif", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
sen = RoiPoly(color='r') 

# --- Muestra la ROI sobre la imagen -------------------------------------
plt.imshow(img, cmap="gray")
sen.display_roi()
plt.show()


# Supongamos que ya identificaste las coordenadas (y, x) superiores izquierdas
# Debes ajustar estos valores según tu imagen:
H, W = 100,150

def obtener_recorte(nombre_tecla):
    print(f"Selecciona el punto superior izquierdo de: {nombre_tecla}")
    plt.imshow(img, cmap="gray")
    roi = RoiPoly(color='r') 
    # Usamos el primer punto que marcaste como origen
    x_inicio, y_inicio = int(roi.x[0]), int(roi.y[0])
    return img[y_inicio : y_inicio + H, x_inicio : x_inicio + W].copy(), (y_inicio, x_inicio)

# Realizamos las 3 selecciones
rec_sin, pos_sin = obtener_recorte("SIN")
rec_cos, pos_cos = obtener_recorte("COS")
rec_tan, pos_tan = obtener_recorte("TAN")

# Mostrar en subplots
fig, axs = plt.subplots(1, 3)
axs[0].imshow(rec_sin, cmap="gray"); axs[0].set_title("SIN")
axs[1].imshow(rec_cos, cmap="gray"); axs[1].set_title("COS")
axs[2].imshow(rec_tan, cmap="gray"); axs[2].set_title("TAN")
plt.show()

## tecla ENTER
img = cv2.imread("./img_calculadora.tif", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
enter = RoiPoly(color='r') 
x_enter, y_enter = int(enter.x[0]), int(enter.y[0])
rec_enter = img[y_enter : y_enter + 125, x_enter : x_enter + 350].copy()

fig, axs = plt.subplots(1, 3)
axs[0].imshow(rec_sin, cmap="gray"); axs[0].set_title("SIN")
axs[1].imshow(rec_enter, cmap="gray"); axs[1].set_title("ENTER")
axs[2].imshow(rec_tan, cmap="gray"); axs[2].set_title("TAN")
plt.show()


# --- Muestra la ROI + info ----------------------------------------------
plt.imshow(img, cmap="gray")
sen.display_roi()
sen.display_mean(img)
plt.show()

# --- Obtengo máscara ----------------------------------------------------
mask = sen.get_mask(img)
mask
type(mask)
mask.dtype
plt.imshow(mask, cmap="gray")
plt.show()

mask2 = sen.get_mask(img)
mask2

# --- ROI info -----------------------------------------------------------
sen.x
sen.y
mean = np.mean(img[mask])
std = np.std(img[mask])