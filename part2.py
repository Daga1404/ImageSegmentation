"""
Parte 2 — Detección de carretera, carriles y líneas de división
===============================================================
Pipeline por frame:
  1. Superficie de carretera  → cromaticidad euclidiana + CLAHE
                                zona de exclusión superior + blob más grande
  2. Líneas de carril         → HLS blanco + amarillo DENTRO de la superficie
  3. Geometría de carriles    → Hough probabilístico → agrupado en 1 línea/lado

Robustez a iluminación:
  - Cromaticidad normalizada (invariante al brillo)
  - CLAHE local antes de todas las operaciones
  - Umbral de brillo mínimo (descarta ruido en sombras profundas)
  - Lane detection confinada a la máscara de carretera (sin falsos positivos
    de follaje, edificios ni señales)

Salidas por video (en road_images/):
  - <nombre>_road.mp4     : superficie de carretera
  - <nombre>_lanes.mp4    : líneas de carril
  - <nombre>_combined.mp4 : todo combinado
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SOURCE_DIR   = "road_images/source_videos"
OUTPUTS_BASE = "road_images"          # carpetas outputs_video_N se crean aquí
TARGET_SIZE  = (640, 360)
FPS_OVERRIDE = None

# — Superficie de carretera —
CHROMA_THRESHOLD = 0.12   # distancia máxima al gris neutro en cromaticidad
BRIGHT_MIN       = 25     # brillo mínimo por canal (descarta sombras ruidosas)
BRIGHT_MAX       = 220    # brillo máximo (descarta reflejos / cielo)
TOP_EXCLUDE      = 0.40   # ignorar el % superior del frame (cielo / pasos elevados)

# — Líneas de carril (HLS, escala OpenCV: H 0-180, L/S 0-255) —
WHITE_L_MIN  = 200
YELLOW_H     = (15, 35)
YELLOW_S_MIN = 90

# — Hough probabilístico —
HOUGH_THRESHOLD = 25
HOUGH_MIN_LEN   = 25
HOUGH_MAX_GAP   = 100
HOUGH_SLOPE_MIN = 0.3     # |pendiente| mínima para considerar una línea de carril

OUTPUT_NAMES = ("road", "lanes", "combined")

_clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_NEUTRAL = np.array([1/3, 1/3, 1/3], dtype=np.float32)


# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE ILUMINACIÓN
# ---------------------------------------------------------------------------
def normalize_lighting(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# MÉTODO 1: SUPERFICIE DE CARRETERA
# ---------------------------------------------------------------------------
def detect_road_surface(frame):
    h, w = frame.shape[:2]

    # 1. Normalizar iluminación
    norm  = normalize_lighting(frame)
    img_f = norm.astype(np.float32)

    # 2. Cromaticidad y distancia al gris neutro
    total  = img_f.sum(axis=2, keepdims=True).clip(min=1e-5)
    chroma = img_f / total
    dist   = np.sqrt(np.sum((chroma - _NEUTRAL) ** 2, axis=2))
    mask   = (dist < CHROMA_THRESHOLD).astype(np.uint8) * 255

    # 3. Umbral de brillo sobre frame original
    brightness       = frame.astype(np.float32).mean(axis=2)
    mask[brightness <  BRIGHT_MIN] = 0
    mask[brightness >  BRIGHT_MAX] = 0

    # 4. Excluir zona superior (cielo, pasos elevados, señales altas)
    mask[:int(h * TOP_EXCLUDE), :] = 0

    # 5. Limpieza morfológica
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9, iterations=2)

    # 6. Conservar ÚNICAMENTE el blob que contiene el punto semilla del centro-inferior.
    #    En footage de dashcam, el centro del borde inferior es casi siempre carretera.
    #    Esto excluye aceras, arcenes y estructuras grises que no están directamente
    #    conectadas a la superficie donde circula el vehículo.
    n, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    road = np.zeros_like(mask)

    # Intentar varios puntos semilla a lo largo del borde inferior (centro → lados)
    # por si el centro está tapado por el capó del coche.
    seed_y = int(h * 0.96)
    for seed_x in [w // 2, w // 3, 2 * w // 3, w // 4, 3 * w // 4]:
        road_lbl = labels[seed_y, seed_x]
        if road_lbl > 0:
            road[labels == road_lbl] = 255
            break   # el primer punto semilla válido define la carretera

    return road


# ---------------------------------------------------------------------------
# MÉTODO 2: LÍNEAS DE CARRIL (solo dentro de la superficie detectada)
# ---------------------------------------------------------------------------
def detect_lane_lines(frame, road_mask):
    norm = normalize_lighting(frame)
    hls  = cv2.cvtColor(norm, cv2.COLOR_BGR2HLS)
    _, l_ch, s_ch = cv2.split(hls)
    h_ch          = hls[:, :, 0]

    # Marcas blancas: L muy alto
    white_mask = cv2.inRange(l_ch, WHITE_L_MIN, 255)

    # Marcas amarillas: hue en rango + saturación alta
    yellow_mask = cv2.inRange(
        hls,
        np.array([YELLOW_H[0], 50,  YELLOW_S_MIN]),
        np.array([YELLOW_H[1], 255, 255])
    )

    # Confinar al área de carretera detectada → elimina follaje, señales, edificios
    white_mask  = cv2.bitwise_and(white_mask,  white_mask,  mask=road_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=road_mask)

    # Unión para Hough
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    k3       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3, iterations=1)

    return white_mask, yellow_mask, combined


# ---------------------------------------------------------------------------
# MÉTODO 3: HOUGH → agrupado en 1 línea por lado
# ---------------------------------------------------------------------------
def _extrapolate(slope, intercept, h):
    """Devuelve dos puntos (x_bottom, y_bottom, x_top, y_top) para una línea."""
    y_bot = h
    y_top = int(h * 0.55)
    if abs(slope) < 1e-5:
        return None
    x_bot = int((y_bot - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return x_bot, y_bot, x_top, y_top


def detect_hough_lines(binary, frame_shape):
    h, w  = frame_shape[:2]
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP
    )
    if lines is None:
        return []

    left_params, right_params = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        mid_x     = (x1 + x2) / 2

        if abs(slope) < HOUGH_SLOPE_MIN:
            continue                         # descartar líneas casi horizontales

        if slope < 0 and mid_x < w * 0.65:  # carril izquierdo
            left_params.append((slope, intercept))
        elif slope > 0 and mid_x > w * 0.35:# carril derecho
            right_params.append((slope, intercept))

    result = []
    for params in [left_params, right_params]:
        if not params:
            continue
        # Mediana robusta frente a outliers
        med_slope = float(np.median([p[0] for p in params]))
        med_inter = float(np.median([p[1] for p in params]))
        seg       = _extrapolate(med_slope, med_inter, h)
        if seg:
            result.append(seg)

    return result


# ---------------------------------------------------------------------------
# RENDERS
# ---------------------------------------------------------------------------
def draw_road_only(frame, road_mask):
    overlay = frame.copy()
    overlay[road_mask == 255] = (0, 200, 0)
    return cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)


def draw_lanes_only(frame, white_mask, yellow_mask, hough_lines):
    out = frame.copy()
    out[white_mask  == 255] = (255, 200, 60)   # azul claro
    out[yellow_mask == 255] = (0,   210, 255)  # amarillo
    for (x1, y1, x2, y2) in hough_lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 220), 3, cv2.LINE_AA)
    return out


def draw_combined(frame, road_mask, white_mask, yellow_mask, hough_lines):
    # Capa de carretera semitransparente
    layer = frame.copy()
    layer[road_mask == 255] = (0, 180, 0)
    out = cv2.addWeighted(frame, 0.60, layer, 0.40, 0)

    # Marcas encima
    out[white_mask  == 255] = (255, 200, 60)
    out[yellow_mask == 255] = (0,   210, 255)

    # Líneas Hough encima de todo
    for (x1, y1, x2, y2) in hough_lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 220), 3, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# PROCESAMIENTO DE UN VIDEO
# ---------------------------------------------------------------------------
def process_video(input_path, output_dir):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = FPS_OVERRIDE or cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    base   = os.path.splitext(os.path.basename(input_path))[0]

    paths   = {k: os.path.join(output_dir, f"{base}_{k}.mp4") for k in OUTPUT_NAMES}
    writers = {k: cv2.VideoWriter(p, fourcc, fps, TARGET_SIZE) for k, p in paths.items()}

    print(f"  {total} frames @ {fps:.1f} fps")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, TARGET_SIZE)

        road_mask                       = detect_road_surface(frame)
        white_mask, yellow_mask, binary = detect_lane_lines(frame, road_mask)
        hough_lines                     = detect_hough_lines(binary, frame.shape)

        writers["road"].write(draw_road_only(frame, road_mask))
        writers["lanes"].write(draw_lanes_only(frame, white_mask, yellow_mask, hough_lines))
        writers["combined"].write(draw_combined(frame, road_mask, white_mask, yellow_mask, hough_lines))

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"    frame {frame_idx}/{total}  ({pct:.0f}%)")

    cap.release()
    for w in writers.values():
        w.release()
    for k, p in paths.items():
        print(f"  -> {p}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    videos = sorted(
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )

    if not videos:
        print(f"No se encontraron videos en '{SOURCE_DIR}'.")
    else:
        for idx, v in enumerate(videos, start=1):
            output_dir = os.path.join(OUTPUTS_BASE, f"outputs_video_{idx}")
            print(f"\nVideo {idx}: {v}")
            print(f"  Salida : {output_dir}/")
            process_video(v, output_dir)

    print("Listo.")
