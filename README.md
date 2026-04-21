# Road Image Segmentation

Computer vision project for road detection, segmentation, and lane line identification using OpenCV. The project is split into two independent parts: **Part 1** segments road surfaces from static images using three classical methods, and **Part 2** processes dashcam videos to detect the road surface, lane markings (white and yellow), and extrapolated lane lines in real time.

---

## Project Structure

```
ImageSegmentation/
├── part1.py                     # Static image segmentation (3 methods)
├── part2.py                     # Video lane & road detection (v7)
├── road_images/
│   ├── road0.jpeg … road4.jpeg  # Input images for Part 1
│   ├── result_0.png … result_4.png  # Output composites from Part 1
│   ├── source_videos/           # Input dashcam videos for Part 2
│   │   ├── road-video.mp4
│   │   ├── road-video-dubai.mp4
│   │   ├── road-video-forest.mp4
│   │   ├── road-video-russia.mp4
│   │   └── road-video-yellow-solid.mp4
│   └── outputs_video_1/ … outputs_video_5/  # Part 2 output videos
└── .gitignore
```

## Dependencies

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib (Part 1 only)

```bash
pip install opencv-python numpy matplotlib
```

---

## Part 1 — Static Image Segmentation (`part1.py`)

### What It Does

Takes road images from `road_images/` and segments the road surface using three different methods, then compares them side by side. For each image it produces an 8-panel composite saved as `result_N.png`.

### Segmentation Methods

#### 1. Euclidean Distance

Computes the per-pixel Euclidean distance in BGR space from a reference gray `[128, 128, 128]` (typical asphalt color). Pixels within a threshold distance of 60 are classified as road.

#### 2. K-Means Clustering

Clusters all pixels into `k=3` color groups using OpenCV's K-means. The resulting cluster image is converted to grayscale and binarized via Otsu's threshold to separate road from non-road regions.

#### 3. Watershed

Applies Otsu's threshold on the grayscale image, then uses distance transform to find "sure foreground" seeds. Connected components become markers, and OpenCV's Watershed algorithm grows regions from those markers to segment the image.

### Pipeline

1. **Load** all images from `road_images/` (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`).
2. **Preprocess** — resize to 640x360, convert to grayscale, apply Gaussian blur (5x5).
3. **ROI mask** — a trapezoid covering the lower portion of the frame (bottom edge to 55% height, inset 15% on each side at top) to focus on the road area.
4. **Segment** with each method, then `bitwise_and` with the ROI mask.
5. **Overlay** — green highlight blended at 50% opacity onto the original image.
6. **Metrics** — for each method, compute road density (fraction of ROI classified as road) and print per-image and averaged results.

### Running

```bash
python part1.py
```

**Output:** `road_images/result_0.png` through `result_4.png` (one per input image), plus per-image and average density metrics printed to the console.

---

## Part 2 — Video Road & Lane Detection (`part2.py`)

### What It Does

Processes dashcam videos frame by frame to detect:
- **Road surface** (green overlay)
- **White lane markings** (cyan highlight)
- **Yellow lane markings** (yellow highlight)
- **Extrapolated left/right lane lines** (red lines via Hough transform)

Produces three output videos per input: `*_road.mp4`, `*_lanes.mp4`, and `*_combined.mp4`.

### Core Algorithm (v7)

Previous versions (v1–v6) used global thresholds that failed at night (low brightness makes lane markings look similar to sky/streetlights) and with vegetation (yellow grass matches yellow lane markings). Version 7 solves this with two fundamental changes:

#### 1. Local Adaptive Thresholding for Lane Lines

Instead of a global brightness cutoff, each pixel is compared against the **mean of its 61x61 neighborhood**. A lane marking is always brighter than the asphalt immediately around it regardless of global illumination:

| Condition | Lane L | Asphalt L | Difference |
|-----------|--------|-----------|------------|
| Day       | 240    | 160       | +80        |
| Night     | 140    | 60        | +80        |

This single mechanism captures white continuous, white dashed, yellow continuous, and yellow dashed markings in both day and night.

#### 2. Adaptive Road Detection with Day/Night Mode

Brightness is measured per-frame (mean L channel in HLS). If `L_avg < 80`, the frame switches to **night mode** which:
- Increases color tolerance for road sampling (`30–70` vs day's `20–55`)
- Applies stronger morphological closing (21x21 kernel, 4 iterations) to bridge gaps between vehicles
- Adds dilation before closing to connect fragmented asphalt patches
- Uses additional seed points at higher frame positions (80%, 87%, 93%) for multi-seed voting

### Pipeline (per frame)

1. **Resize** to 640x360.
2. **CLAHE normalization** — equalizes the L channel in LAB color space to handle uneven lighting.
3. **Day/night classification** — mean L in HLS < 80 triggers night mode.
4. **Road surface detection:**
   - Sample road color from three bottom-center patches of the frame.
   - Compute per-pixel color distance from the sampled mean.
   - Apply tolerance band (scaled from inter-patch standard deviation).
   - Morphological open + close to clean up.
   - Connected components + multi-seed voting to pick the largest road blob.
5. **Lane line detection:**
   - `cv2.adaptiveThreshold` on L channel (Gaussian, block=61, C=-18) for bright markings.
   - Separate HSV-based detection for yellow lines (H: 15–38, dynamic S minimum).
   - White = adaptive AND NOT yellow. Combined = white OR yellow.
   - Confined to `road_mask` so vegetation/signs outside the road are eliminated.
6. **Hough line fitting:**
   - Canny edge detection on the combined lane mask.
   - `HoughLinesP` with slope filtering (0.40–3.50) to reject horizontal/vertical noise.
   - Each segment is validated: >=60% of sampled points must fall on `road_mask`.
   - Segments are split into left (negative slope, left half) and right (positive slope, right half).
   - IQR outlier removal on slopes/intercepts per side.
   - Median slope+intercept extrapolated from bottom of frame to 57% height.
   - **EMA temporal smoothing** (alpha=0.25) across frames for stable lines.
7. **Render** three output videos: road-only, lanes-only, and combined overlay.

### Configuration

Key parameters at the top of `part2.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_SIZE` | 640x360 | Frame processing resolution |
| `NIGHT_FRAME_THRESH` | 80 | L_avg below this triggers night mode |
| `ADAPT_BLOCK` | 61 | Local threshold window size (must be odd) |
| `ADAPT_C` | -18 | Pixel must be this much brighter than local mean |
| `HOUGH_THRESHOLD` | 30 | Minimum votes for a Hough line segment |
| `HOUGH_MIN_LEN` | 30 | Minimum segment length in pixels |
| `HOUGH_SMOOTH_ALPHA` | 0.25 | EMA smoothing factor (0=frozen, 1=no smoothing) |

### Running

Place dashcam videos in `road_images/source_videos/`, then:

```bash
python part2.py
```

**Output:** For each input video, three `.mp4` files are written to `road_images/outputs_video_N/`:
- `<name>_road.mp4` — road surface highlighted in green
- `<name>_lanes.mp4` — white and yellow lane markings highlighted, Hough lines in red
- `<name>_combined.mp4` — all detections overlaid together

Progress is printed per 50 frames with percentage and current day/night mode.