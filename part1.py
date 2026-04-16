"""
Ejercicio — Segmentación de imágenes de carretera
================================================
Métodos aplicados:
  1. Distancia Euclidiana
  2. K-means Clustering
  3. Watershed
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CARGA DE IMÁGENES DESDE DIRECTORIO LOCAL
# ---------------------------------------------------------------------------
IMG_DIR = "road_images"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

image_paths = sorted(
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if f.lower().endswith(VALID_EXTENSIONS)
)

# ---------------------------------------------------------------------------
# 2. PREPROCESAMIENTO
# ---------------------------------------------------------------------------
def preprocess(img):
    resized = cv2.resize(img, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    return resized, gray, blurred

def roi_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [0, h],
        [w, h],
        [int(w*0.85), int(h*0.55)],
        [int(w*0.15), int(h*0.55)],
    ], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

# ---------------------------------------------------------------------------
# 3. MÉTODOS DE SEGMENTACIÓN
# ---------------------------------------------------------------------------

# --- 1. Distancia Euclidiana ---
def segment_euclidean(img):
    img_f = img.astype(np.float32)
    road_color = np.array([128,128,128], dtype=np.float32)

    dist = np.sqrt(np.sum((img_f - road_color)**2, axis=2))
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(dist_norm, 60, 255, cv2.THRESH_BINARY_INV)
    return mask


# --- 2. K-means ---
def segment_kmeans(img, k=3):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img.shape)

    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, segmented


# --- 3. Watershed ---
def segment_watershed(img, gray):
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255

    return mask

# ---------------------------------------------------------------------------
# 4. PROCESAMIENTO
# ---------------------------------------------------------------------------
def process_image(path, idx):
    img = cv2.imread(path)
    resized, gray, blurred = preprocess(img)
    roi = roi_mask(resized.shape)

    euclid = cv2.bitwise_and(segment_euclidean(resized),
                             segment_euclidean(resized), mask=roi)

    k_mask, k_segmented = segment_kmeans(resized)
    k_mask = cv2.bitwise_and(k_mask, k_mask, mask=roi)

    ws_mask = segment_watershed(resized, blurred)
    ws_mask = cv2.bitwise_and(ws_mask, ws_mask, mask=roi)

    def overlay(base, mask):
        out = base.copy()
        out[mask==255] = (0,255,0)
        return cv2.addWeighted(base, 0.5, out, 0.5, 0)

    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2,4, figsize=(18,8))

    ax[0,0].imshow(img_rgb); ax[0,0].set_title("Original")
    ax[0,1].imshow(euclid, cmap='gray'); ax[0,1].set_title("Euclidiana")
    ax[0,2].imshow(k_mask, cmap='gray'); ax[0,2].set_title("K-means")
    ax[0,3].imshow(ws_mask, cmap='gray'); ax[0,3].set_title("Watershed")

    ax[1,0].imshow(cv2.cvtColor(overlay(resized, euclid), cv2.COLOR_BGR2RGB))
    ax[1,0].set_title("Overlay Euclidiana")

    ax[1,1].imshow(cv2.cvtColor(overlay(resized, k_mask), cv2.COLOR_BGR2RGB))
    ax[1,1].set_title("Overlay K-means")

    ax[1,2].imshow(cv2.cvtColor(overlay(resized, ws_mask), cv2.COLOR_BGR2RGB))
    ax[1,2].set_title("Overlay Watershed")

    ax[1,3].imshow(cv2.cvtColor(k_segmented, cv2.COLOR_BGR2RGB))
    ax[1,3].set_title("Clusters K-means")

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"result_{idx}.png"))
    plt.close()


# ---------------------------------------------------------------------------
# 5. MÉTRICAS
# ---------------------------------------------------------------------------
def compute_metrics(path):
    img = cv2.imread(path)
    resized, gray, blurred = preprocess(img)
    roi = roi_mask(resized.shape)

    total = roi.sum() // 255

    euclid = segment_euclidean(resized)
    k_mask, _ = segment_kmeans(resized)
    ws_mask = segment_watershed(resized, blurred)

    def density(mask):
        return (cv2.bitwise_and(mask, mask, mask=roi).sum() // 255) / total

    return {
        "Euclidiana": round(density(euclid),3),
        "K-means": round(density(k_mask),3),
        "Watershed": round(density(ws_mask),3)
    }

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Procesando imágenes...\n")

    all_metrics = {}

    for i, path in enumerate(image_paths):
        process_image(path, i)

        m = compute_metrics(path)
        print(path, m)

        for k,v in m.items():
            all_metrics.setdefault(k, []).append(v)

    print("\nPromedios:")
    for k,v in all_metrics.items():
        print(k, np.mean(v))