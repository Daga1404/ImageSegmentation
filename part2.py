"""
Parte 2 — Detección de carretera, carriles y líneas de división  (v6.1)
============================================================================
CAMBIO FUNDAMENTAL respecto a v1-v5:
  Las versiones anteriores usaban umbrales FIJOS de cromaticidad/textura
  que fallaban en distintos tipos de carretera (asfalto oscuro, claro,
  nocturno, desértico, agrietado). Cada ajuste para un video rompía otro.

  v6 usa MUESTREO ADAPTIVO: en cada frame se muestrea el color real del
  asfalto en varias zonas del centro-inferior y se construye el umbral a
  partir de esa muestra. Funciona para cualquier tipo de carretera sin
  cambiar parámetros.

ARQUITECTURA v6:
  ┌─────────────────────────────────────────────────────────┐
  │  1. MUESTREO ADAPTIVO del color de carretera            │
  │     · 3 parches en el centro-inferior del frame         │
  │     · Mediana robusta → color de referencia             │
  │     · Tolerancia dinámica por canal                     │
  ├─────────────────────────────────────────────────────────┤
  │  2. MÁSCARA DE CARRETERA                                │
  │     · Píxeles dentro de tolerancia del color muestreado │
  │     · ROI trapezoidal + exclusión de zona superior      │
  │     · Closing morfológico grande (rellena sombras)      │
  │     · Votación multi-semilla para selección de blob     │
  ├─────────────────────────────────────────────────────────┤
  │  3. LÍNEAS DE CARRIL (dentro de la máscara)             │
  │     Blanco sólido/discontinuo: L > 200 en HLS           │
  │     Amarillo sólido/discontinuo: H[15-38] + S > 70 HLS  │
  │     Gradiente Sobel-X: detecta bordes de marcas viales  │
  │     → Combinación de los 3 para máxima cobertura        │
  ├─────────────────────────────────────────────────────────┤
  │  4. HOUGH ROBUSTO                                       │
  │     · Filtro de zona: izq < 48 %, der > 52 % de ancho  │
  │       (evita que línea central se clasifique mal)       │
  │     · Mediana de parámetros (robusta a outliers)        │
  └─────────────────────────────────────────────────────────┘
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SOURCE_DIR   = "road_images/source_videos"
OUTPUTS_BASE = "road_images"
TARGET_SIZE  = (640, 360)
FPS_OVERRIDE = None

# — ROI trapezoidal —
ROI_TOP_Y        = 0.52
ROI_BOT_Y        = 0.98
ROI_TOP_LEFT     = 0.22
ROI_TOP_RIGHT    = 0.78
ROI_BOTTOM_LEFT  = 0.00
ROI_BOTTOM_RIGHT = 1.00

# — Zona superior excluida (cielo, señales altas) —
TOP_EXCLUDE = 0.43

# — Muestreo adaptivo —
# Tolerancia base por canal: max(std_muestra * TOL_SCALE, TOL_MIN)
ADAPT_TOL_SCALE = 2.8
ADAPT_TOL_MIN   = 20.0   # tolerancia mínima por canal
ADAPT_TOL_MAX   = 55.0   # tolerancia máxima por canal

# — Líneas de carril (umbrales adaptativos según brillo del frame) —
# Día (L_avg >= 120):  blancas L>200, amarillo S>70
# Crepúsculo (80-120): blancas L>170, amarillo S>55
# Noche (< 80):        blancas L>130, amarillo S>40
LANE_BRIGHT_DAY        = 120   # L_avg del frame — umbral día/crepúsculo
LANE_BRIGHT_DUSK       = 80    # L_avg del frame — umbral crepúsculo/noche
WHITE_L_DAY            = 200
WHITE_L_DUSK           = 170
WHITE_L_NIGHT          = 130
YELLOW_H_MIN           = 15
YELLOW_H_MAX           = 38
YELLOW_S_DAY           = 70
YELLOW_S_DUSK          = 55
YELLOW_S_NIGHT         = 40
# Sobel: solo píxeles MÁS BRILLANTES que la carretera (filtra vegetación)
SOBEL_THRESH           = 25    # umbral mínimo de gradiente Sobel-X normalizado
SOBEL_ROAD_OFFSET      = 22    # píxel debe tener L > road_L_median + este offset

# — Hough —
HOUGH_THRESHOLD     = 35
HOUGH_MIN_LEN       = 40
HOUGH_MAX_GAP       = 80
HOUGH_SLOPE_MIN     = 0.25
HOUGH_LEFT_MAX_MID  = 0.48   # izquierdo: mid_x < 48 %
HOUGH_RIGHT_MIN_MID = 0.52   # derecho:   mid_x > 52 %

# — Blob —
BLOB_MIN_FRAC = 0.004   # tamaño mínimo: 0.4 % del área ROI

OUTPUT_NAMES = ("road", "lanes", "combined")


# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE ILUMINACIÓN
# ---------------------------------------------------------------------------
def normalize_lighting(frame):
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# ROI TRAPEZOIDAL
# ---------------------------------------------------------------------------
def build_roi_mask(h, w):
    mask  = np.zeros((h, w), dtype=np.uint8)
    y_top = int(h * ROI_TOP_Y)
    y_bot = int(h * ROI_BOT_Y)
    pts   = np.array([
        [int(w * ROI_BOTTOM_LEFT),  y_bot],
        [int(w * ROI_BOTTOM_RIGHT), y_bot],
        [int(w * ROI_TOP_RIGHT),    y_top],
        [int(w * ROI_TOP_LEFT),     y_top],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ---------------------------------------------------------------------------
# MUESTREO ADAPTIVO DEL COLOR DE CARRETERA
# ---------------------------------------------------------------------------
def sample_road_color(norm):
    """
    Muestrea el color del asfalto desde 3 parches en el centro-inferior.
    Devuelve (mean_bgr, tolerance_bgr) como arrays float32.
    """
    h, w = norm.shape[:2]

    regions = [
        # (y1_frac, y2_frac, x1_frac, x2_frac)
        (0.84, 0.94, 0.38, 0.62),   # centro
        (0.87, 0.95, 0.18, 0.40),   # izquierda
        (0.87, 0.95, 0.60, 0.82),   # derecha
    ]

    means = []
    for y1f, y2f, x1f, x2f in regions:
        y1, y2 = int(h * y1f), int(h * y2f)
        x1, x2 = int(w * x1f), int(w * x2f)
        patch   = norm[y1:y2, x1:x2]
        if patch.size > 0:
            means.append(patch.mean(axis=(0, 1)).astype(np.float32))

    if not means:
        return np.array([100, 100, 100], dtype=np.float32), \
               np.array([30, 30, 30],   dtype=np.float32)

    means      = np.array(means)
    road_mean  = np.median(means, axis=0)
    # Tolerancia: dispersión entre parches * escala + mínimo
    road_std   = means.std(axis=0) if len(means) > 1 else np.zeros(3)
    tolerance  = np.clip(road_std * ADAPT_TOL_SCALE + ADAPT_TOL_MIN,
                         ADAPT_TOL_MIN, ADAPT_TOL_MAX)
    return road_mean.astype(np.float32), tolerance.astype(np.float32)


# ---------------------------------------------------------------------------
# DETECCIÓN DE SUPERFICIE DE CARRETERA (ADAPTIVA)
# ---------------------------------------------------------------------------
def detect_road_surface(frame, norm, roi_mask):
    h, w     = frame.shape[:2]
    roi_area = max(int(roi_mask.sum() / 255), 1)

    # 1. Muestrear color adaptivo de carretera
    road_mean, tolerance = sample_road_color(norm)

    # 2. Máscara: píxeles dentro de la tolerancia del color muestreado
    norm_f = norm.astype(np.float32)
    diff   = np.abs(norm_f - road_mean)
    mask   = np.all(diff < tolerance, axis=2).astype(np.uint8) * 255

    # 3. Excluir zona superior y aplicar ROI
    mask[:int(h * TOP_EXCLUDE), :] = 0
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # 4. Limpieza morfológica
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k17 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5,  iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k17, iterations=3)

    # 5. Componentes conectados + votación multi-semilla
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    road = np.zeros_like(mask)
    if n <= 1:
        return road

    seed_ys = [int(h * f) for f in (0.93, 0.86, 0.78)]
    seed_xs = [int(w * f) for f in (0.50, 0.42, 0.58, 0.33, 0.67)]
    votes   = {}
    for sy in seed_ys:
        for sx in seed_xs:
            if not (0 <= sy < h and 0 <= sx < w):
                continue
            lbl = labels[sy, sx]
            if lbl == 0:
                continue
            if stats[lbl, cv2.CC_STAT_AREA] < BLOB_MIN_FRAC * roi_area:
                continue
            votes[lbl] = votes.get(lbl, 0) + 1

    if votes:
        best_lbl = max(votes, key=votes.get)
        road[labels == best_lbl] = 255
    else:
        valid = [
            (stats[i, cv2.CC_STAT_AREA], i)
            for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= BLOB_MIN_FRAC * roi_area
        ]
        if valid:
            _, best_lbl = max(valid)
            road[labels == best_lbl] = 255

    return road


# ---------------------------------------------------------------------------
# DETECCIÓN DE LÍNEAS DE CARRIL  (umbrales adaptativos)
# Detecta: blancas continuas, blancas discontinuas, amarillas continuas,
#          amarillas discontinuas — todo dentro de la máscara de carretera.
#
# FIX A — POCA LUZ: umbrales de L y S se reducen según brillo promedio
#          del frame. Noche → L>130, S>40. Crepúsculo → L>170, S>55.
#
# FIX B — VEGETACIÓN: el Sobel-X ahora solo acepta píxeles más brillantes
#          que la mediana del asfalto detectado. Las hojas y el pasto no son
#          más brillantes que el asfalto → sus bordes quedan descartados.
#          Además se restringe el amarillo exigiendo que el píxel esté
#          dentro de la road_mask (vegetación amarilla queda afuera).
# ---------------------------------------------------------------------------
def detect_lane_lines(norm, road_mask):
    hls              = cv2.cvtColor(norm, cv2.COLOR_BGR2HLS)
    h_ch, l_ch, s_ch = cv2.split(hls)

    # — FIX A: seleccionar umbrales según brillo promedio del frame —
    l_avg = float(l_ch.mean())
    if l_avg >= LANE_BRIGHT_DAY:
        white_l_min   = WHITE_L_DAY
        yellow_s_min  = YELLOW_S_DAY
    elif l_avg >= LANE_BRIGHT_DUSK:
        white_l_min   = WHITE_L_DUSK
        yellow_s_min  = YELLOW_S_DUSK
    else:                                   # noche / poca luz
        white_l_min   = WHITE_L_NIGHT
        yellow_s_min  = YELLOW_S_NIGHT

    # — Marcas blancas —
    white_mask = cv2.inRange(l_ch, white_l_min, 255)

    # — Marcas amarillas —
    yellow_mask = cv2.inRange(
        hls,
        np.array([YELLOW_H_MIN, 40,  yellow_s_min]),
        np.array([YELLOW_H_MAX, 255, 255])
    )

    # — FIX B: Sobel filtrado por brillo relativo al asfalto —
    # Calcular mediana de L en la zona de carretera detectada
    road_pixels = l_ch[road_mask == 255]
    road_l_med  = float(np.median(road_pixels)) if road_pixels.size > 0 else float(l_avg)

    sobel_x   = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.abs(sobel_x)
    max_s     = abs_sobel.max()
    scaled    = (255.0 * abs_sobel / max_s).astype(np.uint8) if max_s > 0 \
                else np.zeros_like(l_ch)

    # Solo acepta gradiente donde el píxel es NOTABLEMENTE más brillante
    # que el asfalto → descarta bordes de vegetación, sombras, muros
    brighter_than_road = (l_ch.astype(np.int16) > road_l_med + SOBEL_ROAD_OFFSET) \
                         .astype(np.uint8) * 255
    grad_mask = cv2.bitwise_and(
        cv2.inRange(scaled, SOBEL_THRESH, 255),
        brighter_than_road
    )

    # — Combinar los 3 criterios —
    combined = cv2.bitwise_or(white_mask,  yellow_mask)
    combined = cv2.bitwise_or(combined,    grad_mask)

    # — Confinar a la máscara de carretera —
    white_mask  = cv2.bitwise_and(white_mask,  white_mask,  mask=road_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=road_mask)
    combined    = cv2.bitwise_and(combined,    combined,    mask=road_mask)

    # Limpieza
    k3       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3, iterations=1)

    return white_mask, yellow_mask, combined


# ---------------------------------------------------------------------------
# HOUGH → 1 línea por lado
# ---------------------------------------------------------------------------
def _extrapolate(slope, intercept, h, w):
    y_bot = h
    y_top = int(h * 0.57)
    if abs(slope) < 1e-5:
        return None
    x_bot = int(np.clip((y_bot - intercept) / slope, 0, w - 1))
    x_top = int(np.clip((y_top - intercept) / slope, 0, w - 1))
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

    left_p, right_p = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        mid_x     = (x1 + x2) / 2
        if abs(slope) < HOUGH_SLOPE_MIN:
            continue
        if slope < 0 and mid_x < w * HOUGH_LEFT_MAX_MID:
            left_p.append((slope, intercept))
        elif slope > 0 and mid_x > w * HOUGH_RIGHT_MIN_MID:
            right_p.append((slope, intercept))

    result = []
    for params in [left_p, right_p]:
        if not params:
            continue
        med_s = float(np.median([p[0] for p in params]))
        med_i = float(np.median([p[1] for p in params]))
        seg   = _extrapolate(med_s, med_i, h, w)
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
    out[white_mask  == 255] = (255, 200, 60)   # celeste para blancas
    out[yellow_mask == 255] = (0,   210, 255)  # amarillo para amarillas
    for (x1, y1, x2, y2) in hough_lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 220), 3, cv2.LINE_AA)
    return out


def draw_combined(frame, road_mask, white_mask, yellow_mask, hough_lines):
    layer = frame.copy()
    layer[road_mask == 255] = (0, 180, 0)
    out = cv2.addWeighted(frame, 0.60, layer, 0.40, 0)
    out[white_mask  == 255] = (255, 200, 60)
    out[yellow_mask == 255] = (0,   210, 255)
    for (x1, y1, x2, y2) in hough_lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 220), 4, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# VideoWriter con fallback de codec
# ---------------------------------------------------------------------------
def _open_writer(path, fps, size):
    for cc in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*cc)
        w      = cv2.VideoWriter(path, fourcc, fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError(f"No se pudo crear VideoWriter: '{path}'")


# ---------------------------------------------------------------------------
# PROCESAMIENTO DE UN VIDEO
# ---------------------------------------------------------------------------
def process_video(input_path, output_dir):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = FPS_OVERRIDE or cap.get(cv2.CAP_PROP_FPS) or 25.0
    base  = os.path.splitext(os.path.basename(input_path))[0]

    paths = {k: os.path.join(output_dir, f"{base}_{k}.mp4") for k in OUTPUT_NAMES}
    try:
        writers = {k: _open_writer(p, fps, TARGET_SIZE) for k, p in paths.items()}
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        cap.release()
        return

    roi_mask  = build_roi_mask(TARGET_SIZE[1], TARGET_SIZE[0])
    frame_idx = 0
    print(f"  {total} frames @ {fps:.1f} fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, TARGET_SIZE)
        norm  = normalize_lighting(frame)

        road_mask                       = detect_road_surface(frame, norm, roi_mask)
        white_mask, yellow_mask, binary = detect_lane_lines(norm, road_mask)
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
            out_dir = os.path.join(OUTPUTS_BASE, f"outputs_video_{idx}")
            print(f"\nVideo {idx}: {v}\n  Salida: {out_dir}/")
            process_video(v, out_dir)
    print("Listo.")
