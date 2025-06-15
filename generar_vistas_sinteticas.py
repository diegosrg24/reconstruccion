import cv2
import numpy as np
from pathlib import Path

# Ruta a imagen original
img_path = Path(r"C:\dataset-EdmCrack600\images\6.png")
output_dir = Path(r"C:\entrada_grieta1_parallax")
output_dir.mkdir(parents=True, exist_ok=True)

# Cargar imagen
image = cv2.imread(str(img_path))
h, w = image.shape[:2]

# Definir parámetros de proyección en perspectiva
f = 800  # distancia focal simulada
cx, cy = w / 2, h / 2

# Parámetros del arco de la cámara
num_views = 100
radius = 300  # radio del arco en píxeles
angles = np.linspace(-35, 0, num_views)  # ángulos desde -30 a +30 grados

count = 0
for theta in angles:
    # Convertir grados a radianes
    angle_rad = np.radians(theta)
    
    # Matriz de rotación alrededor del eje Y
    R = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    
    # Proyección en perspectiva simulada
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    
    Rt = np.hstack((R, np.array([[0], [0], [radius]])))  # cámara girando en arco
    P = K @ Rt  # matriz de proyección 3x4

    # Generar proyección en perspectiva
    warped = cv2.warpPerspective(image, P[:, [0,1,3]], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Guardar imagen generada
    out_path = output_dir / f"view_{count:03d}.png"
    cv2.imwrite(str(out_path), warped)
    count += 1

print(f"Se generaron {count} imágenes con parallax en: {output_dir}")
