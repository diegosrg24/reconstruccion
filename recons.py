import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# === Ruta de imágenes
folder = Path(r"C:\entrada_grieta1_parallax")
image_files = sorted(folder.glob("view_*.png"))

# === Parámetros de cámara simulada
img_sample = cv2.imread(str(image_files[0]))
h, w = img_sample.shape[:2]
f = 800
K = np.array([[f, 0, w / 2],
              [0, f, h / 2],
              [0, 0, 1]])

# === Inicializar acumulador de puntos 3D
global_points = []

# === SIFT + matcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# === Procesar por pares cada n frames
step = 5
for i in range(0, len(image_files) - step, step):
    img1 = cv2.imread(str(image_files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image_files[i + step]), cv2.IMREAD_GRAYSCALE)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = bf.knnMatch(des1, des2, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

    if len(good) < 8:
        print(f"Par {i}-{i+step}: pocos matches ({len(good)}), se salta.")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print(f"Par {i}-{i+step}: no se pudo estimar E.")
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    pts1_un = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_un = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

    pts_4d = cv2.triangulatePoints(P1, P2, pts1_un, pts2_un)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T

    global_points.append(pts_3d)

# === Concatenar nube final
global_points = np.vstack(global_points)
print("Total puntos reconstruidos:", len(global_points))

# === Visualización
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(global_points[:, 0], global_points[:, 1], global_points[:, 2], s=1)
ax.set_title("Reconstrucción 3D básica")
plt.show()

# === Guardar en formato .ply
def save_to_ply(filename, points):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write("{} {} {}\n".format(p[0], p[1], p[2]))

save_to_ply("nube_grieta.ply", global_points)
print("Guardado como nube_grieta.ply")
