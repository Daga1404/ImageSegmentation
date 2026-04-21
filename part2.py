"""
Parte 2 — Detección de carretera, carriles y líneas de división  (v7)
======================================================================
PROBLEMA RAÍZ de v1-v6:
  Todos usaban umbrales GLOBALES por frame (L>200, S>70, etc.).
  De noche una línea blanca tiene L=140 pero también el cielo y farolas.
  Con vegetación, el pasto amarillo tiene el mismo rango H/S que las marcas.

SOLUCIÓN v7 — DOS CAMBIOS FUNDAMENTALES:

  ① DETECCIÓN DE LÍNEAS POR UMBRAL ADAPTIVO LOCAL (cv2.adaptiveThreshold)
    Una marca vial SIEMPRE es más brillante que el asfalto inmediatamente
    a su alrededor, sin importar la luz global del frame.
    adaptiveThreshold compara cada píxel con su vecindad (bloque 61×61):
      · Día:   línea blanca L=240 vs asfalto L=160 → diferencia +80  ✓
      · Noche: línea blanca L=140 vs asfalto L= 60 → diferencia +80  ✓
      · Vegetación fuera de road_mask → no llega a Hough              ✓
    Esto captura blancas continuas, blancas discontinuas, amarillas
    continuas y discontinuas con el mismo algoritmo, día y noche.

  ② DETECCIÓN DE CARRETERA MÁS ROBUSTA DE NOCHE
    De noche se sube la tolerancia de muestreo adaptivo y se aplica
    un closing extra para unir el asfalto entre los vehículos.
    Además se añade dilación antes del closing para conectar fragmentos.
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
TOP_EXCLUDE      = 0.43   # fracción superior ignorada (cielo)

# — Muestreo adaptivo de carretera —
ADAPT_TOL_SCALE      = 2.8
ADAPT_TOL_MIN_DAY    = 20.0
ADAPT_TOL_MAX_DAY    = 55.0
ADAPT_TOL_MIN_NIGHT  = 30.0   # más permisivo de noche
ADAPT_TOL_MAX_NIGHT  = 70.0
NIGHT_FRAME_THRESH   = 80     # L_avg < 80 → modo noche

# — Líneas de carril: umbral adaptivo local —
ADAPT_BLOCK  = 61    # tamaño de ventana local (debe ser impar)
ADAPT_C      = -18   # constante: píxel debe superar media local en |C| niveles
                     # valor negativo → exige píxel MÁS BRILLANTE que vecindad

# Para amarillo se usa detección por color (el adaptivo no discrimina amarillo)
# pero el S_min se ajusta dinámicamente
YELLOW_H_MIN = 15
YELLOW_H_MAX = 38

# — Hough —
HOUGH_THRESHOLD     = 30
HOUGH_MIN_LEN       = 30
HOUGH_MAX_GAP       = 100
HOUGH_SLOPE_MIN     = 0.40   # |pendiente| mínima — evita casi horizontales
HOUGH_SLOPE_MAX     = 3.50   # |pendiente| máxima — evita casi verticales
HOUGH_LEFT_MAX_MID  = 0.50
HOUGH_RIGHT_MIN_MID = 0.50
HOUGH_MIN_SEGMENTS  = 2      # mínimo segmentos para dibujar la línea
HOUGH_SMOOTH_ALPHA  = 0.25   # EMA: 0=congelado, 1=sin suavizado

BLOB_MIN_FRAC = 0.003
OUTPUT_NAMES  = ("road", "lanes", "combined")


# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE ILUMINACIÓN (CLAHE)
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
# DETECCIÓN DE MODO (día / noche)
# ---------------------------------------------------------------------------
def get_brightness(norm):
    """L promedio del canal L en HLS."""
    hls = cv2.cvtColor(norm, cv2.COLOR_BGR2HLS)
    return float(hls[:, :, 1].mean())


# ---------------------------------------------------------------------------
# MUESTREO ADAPTIVO DEL COLOR DE CARRETERA
# ---------------------------------------------------------------------------
def sample_road_color(norm, night_mode):
    h, w = norm.shape[:2]
    regions = [
        (0.84, 0.94, 0.38, 0.62),
        (0.87, 0.95, 0.18, 0.40),
        (0.87, 0.95, 0.60, 0.82),
    ]
    means = []
    for y1f, y2f, x1f, x2f in regions:
        patch = norm[int(h*y1f):int(h*y2f), int(w*x1f):int(w*x2f)]
        if patch.size > 0:
            means.append(patch.mean(axis=(0,1)).astype(np.float32))

    if not means:
        return np.array([100,100,100], np.float32), np.array([35,35,35], np.float32)

    means     = np.array(means)
    road_mean = np.median(means, axis=0)
    road_std  = means.std(axis=0) if len(means) > 1 else np.zeros(3)

    if night_mode:
        tol_min, tol_max = ADAPT_TOL_MIN_NIGHT, ADAPT_TOL_MAX_NIGHT
    else:
        tol_min, tol_max = ADAPT_TOL_MIN_DAY, ADAPT_TOL_MAX_DAY

    tolerance = np.clip(road_std * ADAPT_TOL_SCALE + tol_min, tol_min, tol_max)
    return road_mean.astype(np.float32), tolerance.astype(np.float32)


# ---------------------------------------------------------------------------
# DETECCIÓN DE SUPERFICIE DE CARRETERA
# ---------------------------------------------------------------------------
def detect_road_surface(norm, roi_mask, night_mode):
    h, w     = norm.shape[:2]
    roi_area = max(int(roi_mask.sum() / 255), 1)

    road_mean, tolerance = sample_road_color(norm, night_mode)

    diff = np.abs(norm.astype(np.float32) - road_mean)
    mask = np.all(diff < tolerance, axis=2).astype(np.uint8) * 255

    mask[:int(h * TOP_EXCLUDE), :] = 0
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # Limpieza ligera ANTES de etiquetar componentes: sólo OPEN para quitar
    # ruido puntual. El CLOSE pesado se aplica DESPUÉS de seleccionar el blob
    # ganador — así no puede construir un puente entre el asfalto y regiones
    # vecinas de color similar (banqueta, hombro, edificios) antes de la CC.
    k5   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    road = np.zeros_like(mask)
    if n <= 1:
        return road

    # Votación multi-semilla
    if night_mode:
        seed_ys = [int(h * f) for f in (0.80, 0.87, 0.93)]
    else:
        seed_ys = [int(h * f) for f in (0.93, 0.86, 0.78)]
    seed_xs = [int(w * f) for f in (0.50, 0.42, 0.58, 0.33, 0.67)]

    votes = {}
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
        # Aceptar TODOS los componentes con voto: si un vehículo/sombra
        # parte la carretera en varios fragmentos, las 15 semillas (3×5)
        # reparten votos entre ellos y se conservan todos los trozos.
        for lbl in votes:
            road[labels == lbl] = 255
    else:
        valid = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= BLOB_MIN_FRAC * roi_area]
        if valid:
            road[labels == max(valid)[1]] = 255

    # CLOSE pesado SÓLO sobre el blob ganador: rellena huecos internos
    # (vehículos, sombras, marcas viales) sin poder filtrar hacia regiones
    # vecinas porque el resto de la máscara ya está en negro.
    if road.any():
        if night_mode:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            road    = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k_close, iterations=4)
        else:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            road    = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k_close, iterations=3)
        # Re-confinar al ROI por si el CLOSE expandió el contorno más allá
        road = cv2.bitwise_and(road, road, mask=roi_mask)

    return road


# ---------------------------------------------------------------------------
# DETECCIÓN DE LÍNEAS DE CARRIL — UMBRAL ADAPTIVO LOCAL
# ---------------------------------------------------------------------------
def detect_lane_lines(norm, road_mask, night_mode):
    """
    Usa cv2.adaptiveThreshold sobre el canal L (HLS):
      Cada píxel se compara con la media de su vecindad ADAPT_BLOCK×ADAPT_BLOCK.
      Si es más brillante por más de |ADAPT_C| → marca vial.
    Funciona de día y de noche porque la comparación es RELATIVA al entorno.

    El amarillo se detecta por color (H+S) ya que el adaptivo no discrimina tono.
    Ambas máscaras se confinan a road_mask para eliminar vegetación y señales.
    """
    hls              = cv2.cvtColor(norm, cv2.COLOR_BGR2HLS)
    h_ch, l_ch, s_ch = cv2.split(hls)
    hsv              = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    s_hsv            = hsv[:, :, 1]

    # ① Marcas por brillo local relativo (blancas día+noche, amarillas día)
    adaptive_bright = cv2.adaptiveThreshold(
        l_ch,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPT_BLOCK,
        ADAPT_C        # negativo: requiere ser MÁS brillante que la media
    )

    # ② Marcas amarillas por color (umbral S adaptivo según saturación media)
    s_mean = float(s_hsv[road_mask == 255].mean()) if road_mask.any() else 50.0
    # De noche la saturación global baja → bajar el umbral proporcionalmente
    yellow_s_min = max(30, int(s_mean * 0.55))
    yellow_mask = cv2.inRange(
        hls,
        np.array([YELLOW_H_MIN, 30,  yellow_s_min]),
        np.array([YELLOW_H_MAX, 255, 255])
    )

    # ③ Extraer blancas: brillo adaptivo AND no-amarillo
    #    (evita que el amarillo pase como "blanco" por ser brillante)
    white_mask  = cv2.bitwise_and(adaptive_bright, cv2.bitwise_not(yellow_mask))

    # ④ Filtrar amarillo: exigir que también sea localmente brillante
    #    (elimina vegetación/pasto seco que comparte rango de tono amarillo
    #     pero NO es más brillante que su entorno como lo es una marca vial)
    yellow_mask = cv2.bitwise_and(yellow_mask, adaptive_bright)

    # ⑤ Combinar
    combined = cv2.bitwise_or(white_mask, yellow_mask)

    # ⑥ Confinar a carretera → elimina vegetación, señales, edificios
    white_mask  = cv2.bitwise_and(white_mask,  white_mask,  mask=road_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=road_mask)
    combined    = cv2.bitwise_and(combined,    combined,    mask=road_mask)

    # ⑦ Limpieza morfológica leve
    k3       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3, iterations=1)

    return white_mask, yellow_mask, combined


# ---------------------------------------------------------------------------
# HOUGH → 1 línea por lado  (robusto)
# ---------------------------------------------------------------------------

# Estado de suavizado temporal (EMA) — persistente entre frames
_lane_smooth = {"left": None, "right": None}   # (slope, intercept) suavizados


def _iqr_filter(values):
    """Elimina outliers fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]."""
    if len(values) < 4:
        return values
    arr  = np.array(values)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr  = q3 - q1
    mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
    return arr[mask].tolist() if mask.sum() > 0 else values


def _extrapolate(slope, intercept, h, w):
    """Extrapola la línea entre y_bot y y_top con coordenadas clipeadas."""
    y_bot = h
    y_top = int(h * 0.57)
    if abs(slope) < 1e-5:
        return None
    x_bot = int(np.clip((y_bot - intercept) / slope, 0, w - 1))
    x_top = int(np.clip((y_top - intercept) / slope, 0, w - 1))
    return x_bot, y_bot, x_top, y_top


def _smooth_lane(side, slope, intercept):
    """Aplica EMA al par (slope, intercept) para suavizar entre frames."""
    prev = _lane_smooth[side]
    if prev is None:
        _lane_smooth[side] = (slope, intercept)
    else:
        a = HOUGH_SMOOTH_ALPHA
        _lane_smooth[side] = (
            a * slope     + (1 - a) * prev[0],
            a * intercept + (1 - a) * prev[1],
        )
    return _lane_smooth[side]


def detect_hough_lines(binary, frame_shape, road_mask):
    h, w  = frame_shape[:2]
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP
    )

    left_slopes, left_ints   = [], []
    right_slopes, right_ints = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope     = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            mid_x     = (x1 + x2) / 2
            abs_slope = abs(slope)

            if abs_slope < HOUGH_SLOPE_MIN or abs_slope > HOUGH_SLOPE_MAX:
                continue

            # --- NUEVO: validar que el segmento esté sobre la carretera ---
            # Muestrear ~12 puntos a lo largo del segmento y verificar
            # que la mayoría caen dentro de road_mask.
            n_pts    = 12
            xs = np.linspace(x1, x2, n_pts).astype(int)
            ys = np.linspace(y1, y2, n_pts).astype(int)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            on_road  = road_mask[ys, xs].sum() / 255
            if on_road / n_pts < 0.60:   # < 60 % del segmento sobre carretera → descartar
                continue
            # -----------------------------------------------------------

            if slope < 0 and mid_x < w * HOUGH_LEFT_MAX_MID:
                left_slopes.append(slope)
                left_ints.append(intercept)
            elif slope > 0 and mid_x > w * HOUGH_RIGHT_MIN_MID:
                right_slopes.append(slope)
                right_ints.append(intercept)

    result = []
    for side, slopes, ints in [
        ("left",  left_slopes,  left_ints),
        ("right", right_slopes, right_ints),
    ]:
        if len(slopes) < HOUGH_MIN_SEGMENTS:
            if _lane_smooth[side] is not None:
                s, i = _lane_smooth[side]
                seg  = _extrapolate(s, i, h, w)
                if seg:
                    result.append(seg)
            continue

        slopes_f = _iqr_filter(slopes)
        ints_f   = _iqr_filter(ints)

        med_s = float(np.median(slopes_f))
        med_i = float(np.median(ints_f))

        med_s, med_i = _smooth_lane(side, med_s, med_i)

        seg = _extrapolate(med_s, med_i, h, w)
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
    out[white_mask  == 255] = (255, 200, 60)
    out[yellow_mask == 255] = (0,   210, 255)
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

    # Resetear suavizado temporal al inicio de cada video
    _lane_smooth["left"]  = None
    _lane_smooth["right"] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame      = cv2.resize(frame, TARGET_SIZE)
        norm       = normalize_lighting(frame)
        night_mode = get_brightness(norm) < NIGHT_FRAME_THRESH

        road_mask                       = detect_road_surface(norm, roi_mask, night_mode)
        white_mask, yellow_mask, binary = detect_lane_lines(norm, road_mask, night_mode)
        hough_lines                     = detect_hough_lines(binary, frame.shape, road_mask)

        writers["road"].write(draw_road_only(frame, road_mask))
        writers["lanes"].write(draw_lanes_only(frame, white_mask, yellow_mask, hough_lines))
        writers["combined"].write(draw_combined(frame, road_mask, white_mask, yellow_mask, hough_lines))

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct  = frame_idx / total * 100 if total > 0 else 0
            mode = "NOCHE" if night_mode else "DÍA  "
            print(f"    frame {frame_idx}/{total}  ({pct:.0f}%)  [{mode}]")

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
