import cv2
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly 


# 1.1

image = cv2.imread("./faces.jpg") # Cargo imagen
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Paso de BGR a RGB
grayimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Paso a escala de grises
plt.figure(), plt.imshow(image), plt.title("Imagen Original"), plt.show(block=False)

type(image)
image.dtype 
image.shape
type(grayimg)
grayimg.shape
grayimg.dtype

image.max()
image.min()
grayimg.max()       
grayimg.min()

# 1.2

