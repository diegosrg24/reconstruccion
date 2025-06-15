import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dataset_path = Path(r"C:\dataset-EdmCrack600")
images_path = dataset_path / "images"
masks_path = dataset_path / "masks"

image_files = sorted(images_path.glob("*.png"))
mask_files = sorted(masks_path.glob("*.png"))

# Mostrar los primeros 5 pares
num = min(5, len(image_files))

plt.figure(figsize=(10, 4 * num))
for i in range(num):
    img = cv2.imread(str(image_files[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
    
    plt.subplot(num, 2, 2*i+1)
    plt.imshow(img)
    plt.title(f"Imagen {i+1}")
    plt.axis("off")
    
    plt.subplot(num, 2, 2*i+2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Máscara {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
